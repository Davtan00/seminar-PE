from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from app.chains.sentiment_chain import SentimentAnalysisChain
from app.config import get_settings
from app.utils.cost_tracker import track_cost
from app.chains.generation_chain import DataGenerationChain, MAX_RECORDS
from app.chains.text_analysis_chain import TextAnalysisChain
from concurrent.futures import ThreadPoolExecutor
from app.routes import local_inference,hf_inference,generation 
from app.utils import cache_manager
from app.analysis.pdf_generator import SentimentAnalysisReport
from app.utils.auth import verify_api_key
from app.routes.hf_inference import (
    analyze_text_batch_hf, 
    analyze_text_batch_hf_quick, 
    analyze_text_batch_hf_large,
    HuggingFaceRequest,  
    LargeAnalysisRequest
)
from app.models.hf_models import HuggingFaceRequest, LargeAnalysisRequest
from app.models.advanced_config import AdvancedConfig
from app.utils.encryption import decrypt_api_key
from app.models.advanced_config import AdvancedGenerationRequest
import asyncio
from typing import List, Dict
import random
from app.chains.gen_sen_chain import GenSenChain
from fastapi.middleware.cors import CORSMiddleware
import logging
import uuid
import asyncio
import os
from fastapi.responses import FileResponse
from cachetools import TTLCache

pdf_cache = TTLCache(maxsize=100, ttl=3600)  # 1 hour TTL
analysis_status = TTLCache(maxsize=100, ttl=3600)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentiment Data Handling & Analysis")

settings = get_settings()

# Configure CORS with settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=False, 
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

app.include_router(local_inference.router)
app.include_router(hf_inference.router)
app.include_router(cache_manager.router)
app.include_router(generation.router) 


class TextInput(BaseModel):
    text: str
    domain: str


class BatchRequest(BaseModel):
    texts: List[TextInput]


@app.post("/analyze")
async def analyze_texts(request: BatchRequest):
    chain = SentimentAnalysisChain()
    results = []
    total_cost = 0.0

    for idx, item in enumerate(request.texts):
        try:
            with track_cost() as cost:
                result = await chain.analyze(item.text, item.domain, index=idx)
                total_cost += cost.get_costs()["total_cost"]
            results.append(result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {
        "results": results,
        "batch_metrics": {
            "total_cost_usd": round(total_cost, 4),
            "processed_items": len(results)
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cors_config": {
            "origins": settings.CORS_ORIGINS,
            "methods": settings.CORS_METHODS,
            "headers": settings.CORS_HEADERS
        }
    }


class RawTextInput(BaseModel):
    text: str
    domain: str = "general"


class DataCleaningRequest(BaseModel):
    raw_texts: List[RawTextInput]


@app.post("/clean-and-analyze")
async def clean_and_analyze(
        request: DataCleaningRequest,
        _: str = Depends(verify_api_key)  # Changed x_api_key to _ so that the IDE doesnt complain
):
    chain = SentimentAnalysisChain()
    cleaned_results = []
    total_cost = 0.0

    for idx, item in enumerate(request.raw_texts):
        try:
            with track_cost() as cost:
                result = await chain.analyze(item.text, item.domain, index=idx)
                total_cost += cost.get_costs()["total_cost"]
            cleaned_results.append(result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {
        "cleaned_data": cleaned_results,
        "summary": {
            "total_processed": len(cleaned_results),
            "sentiment_distribution": {
                "positive": sum(1 for x in cleaned_results if x["sentiment"] == "positive"),
                "neutral": sum(1 for x in cleaned_results if x["sentiment"] == "neutral"),
                "negative": sum(1 for x in cleaned_results if x["sentiment"] == "negative")
            },
            "total_cost_usd": round(total_cost, 4)
        }
    }


#Should be used to indicate how poorly a response is without a good prompt
@app.post("/bad-clean")
async def naive_analyze(
        request: DataCleaningRequest,
        _: str = Depends(verify_api_key)
):
    chain = SentimentAnalysisChain()
    results = []
    total_cost = 0.0
    errors = 0

    for idx, item in enumerate(request.raw_texts):
        try:
            with track_cost() as cost:
                result = await chain.analyze(item.text, item.domain, use_bad_prompt=True, index=idx)
                total_cost += cost.get_costs()["total_cost"]
            results.append(result)
            if "error" in result:
                errors += 1
        except Exception as e:
            errors += 1
            results.append({
                "id": idx + 1,
                "text": item.text,
                "error": str(e),
                "prompt_type": "bad"
            })

    return {
        "analyzed_data": results,
        "summary": {
            "total_processed": len(results),
            "successful_analyses": len(results) - errors,
            "failed_analyses": errors,
            "total_cost_usd": round(total_cost, 4)
        }
    }


class GenerationRequest(BaseModel):
    domain: str
    count: int = 10
    sentiment_distribution: Optional[Dict[str, float]] = None  # e.g., {"positive": 0.4, "neutral": 0.2, "negative": 0.4}
    output_format: str = "json"  # Could add support for CSV or other formats later
    verbose: bool = True 


@app.post("/generate-data")
async def generate_data(
        request: GenerationRequest,
        _: str = Depends(verify_api_key)
):
    count = min(request.count, MAX_RECORDS)
    chain = DataGenerationChain()
    total_cost = 0.0

    try:
        with track_cost() as cost:
            result = await chain.generate(
                domain=request.domain,
                count=count,
                sentiment_distribution=request.sentiment_distribution,
                verbose=request.verbose  # Pass the verbose parameter
            )
            total_cost = cost.get_costs()["total_cost"]

        return {
            "generated_data": result["data"],
            "summary": {
                "total_generated": len(result["data"]),
                "requested_count": count,
                "domain": request.domain,
                "sentiment_distribution": {
                    "positive": sum(1 for x in result["data"] if x["sentiment"] == "positive"),
                    "neutral": sum(1 for x in result["data"] if x["sentiment"] == "neutral"),
                    "negative": sum(1 for x in result["data"] if x["sentiment"] == "negative")
                },
                "warnings": result["warnings"],
                "total_cost_usd": round(total_cost, 4)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bad-generate-data")
async def generate_data_basic(
        request: GenerationRequest,
        _: str = Depends(verify_api_key)
):
    chain = DataGenerationChain()
    total_cost = 0.0

    try:
        with track_cost() as cost:
            results = await chain.generate(request.domain, request.count, use_bad_prompt=True)
            total_cost = cost.get_costs()["total_cost"]

        return {
            "generated_data": [
                {"id": i + 1, **item} for i, item in enumerate(results)
            ],
            "summary": {
                "total_generated": len(results),
                "domain": request.domain,
                "total_cost_usd": round(total_cost, 4)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SimpleGenerationRequest(BaseModel):
    domain: str
    count: int = Field(gt=0, le=MAX_RECORDS)

@app.post("/generate-simple")
async def generate_simple_data(
    request: SimpleGenerationRequest,
    _: str = Depends(verify_api_key)
):
    chain = DataGenerationChain()
    total_cost = 0.0

    try:
        with track_cost() as cost:
            results = await chain.generate_simple(request.domain, request.count)
            total_cost = cost.get_costs()["total_cost"]

        return {
            "generated_data": results["data"],
            "summary": {
                "total_generated": len(results["data"]),
                "requested_count": request.count,
                "domain": request.domain,
                "total_cost_usd": round(total_cost, 4)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



class TextAnalysisRequest(BaseModel):
    generated_data: List[Dict[str, Any]]
    summary: Optional[Dict[str, Any]]

@app.post("/analyze-text-batch")
async def analyze_text_batch(
    request: TextAnalysisRequest,
    _: str = Depends(verify_api_key)
):
    chain = TextAnalysisChain()
    results = []
    total_cost = 0.0
    
    for item in request.generated_data:
        try:
            with track_cost() as cost:
                result = await chain.analyze_text(item["text"], index=item["id"]-1)
                total_cost += cost.get_costs()["total_cost"]
            results.append(result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    sentiment_counts = {
        "positive": sum(1 for x in results if x["sentiment"] == "positive"),
        "neutral": sum(1 for x in results if x["sentiment"] == "neutral"),
        "negative": sum(1 for x in results if x["sentiment"] == "negative")
    }
    
    return {
        "analyzed_data": results,
        "summary": {
            "total_analyzed": len(results),
            "sentiment_distribution": sentiment_counts,
            "total_cost_usd": round(total_cost, 4)
        }
    }





#Huggingface inference API , PRO version, gives you 20k rate limit per day
@app.post("/analyze-text-batch-hf")
async def analyze_text_batch_hf_endpoint(request: HuggingFaceRequest, _: str = Depends(verify_api_key)):
    return await analyze_text_batch_hf(request, _)

@app.post("/analyze-text-batch-hf-quick")
async def analyze_text_batch_hf_quick_endpoint(request: HuggingFaceRequest, _: str = Depends(verify_api_key)):
    return await analyze_text_batch_hf_quick(request, _)

@app.post("/analyze-text-batch-hf-large")
async def analyze_text_batch_hf_large_endpoint(request: LargeAnalysisRequest, _: str = Depends(verify_api_key)):
    return await analyze_text_batch_hf_large(request, _)

# Format results with robust response handling
async def format_sentiment_result(result) -> Tuple[bool, float]:
    try:
        if isinstance(result, str):
            # Handle unexpected string response
            return False, 0.0
            
        if isinstance(result, list):
            if not result:  # Empty list
                return False, 0.0
            sentiment_result = result[0]
        else:
            sentiment_result = result
            
        
        label = sentiment_result.get("label", "")
        score = sentiment_result.get("score", 0.0)
        
        return label == "POSITIVE", score
    except Exception as e:
        print(f"Error formatting result: {str(e)}, raw result: {result}")
        return False, 0.0
## Not used
async def generate_sentiment_batch(
    openai_key: str,
    sentiment: str,
    count: int,
    config: AdvancedConfig
) -> List[Dict]:
    chain = DataGenerationChain(api_key=openai_key)
    
    prompt_config = f"""
    Generate a {sentiment} review for the {config.domain} domain.
    Consider the following parameters:
    - Realism: {config.realism}
    - Domain relevance: {config.domainRelevance}
    - Formality: {config.formality}
    - Lexical complexity: {config.lexicalComplexity}
    - Cultural sensitivity: {config.culturalSensitivity}
    
    The response should be natural and maintain consistent sentiment.
    """
    
    results = []
    for _ in range(count):
        result = await chain.generate_single(
            prompt_config,
            temperature=config.temperature,
            top_p=config.topP,
            max_tokens=config.maxTokens,
            frequency_penalty=config.frequencyPenalty,
            presence_penalty=config.presencePenalty
        )
        results.append({
            "text": result["text"],
            "sentiment": sentiment
        })
    
    return results
### Used and I will take apart the code and not just leave it all in api.py
@app.post("/generate-advanced")
async def generate_advanced_data(
    request: AdvancedGenerationRequest,
    strict: bool = Query(False, description="Ensure exact distribution and count matching")
):
    try:
        logger.info("🚀 Starting advanced generation request")
        logger.info(f"Domain: {request.config.domain}")
        logger.info(f"Row Count: {request.config.rowCount}")
        logger.info(f"Strict Mode: {strict}")
        logger.info(f"Sentiment Distribution: positive={request.config.sentimentDistribution.positive}%, "
                   f"negative={request.config.sentimentDistribution.negative}%, "
                   f"neutral={request.config.sentimentDistribution.neutral}%")

        openai_key = decrypt_api_key(request.encryptedKey)
        logger.info("✅ API key decrypted successfully")
        
        chain = GenSenChain(api_key=openai_key)
        logger.info("✅ GenSenChain initialized")
        
        distribution = {
            "positive": int(request.config.rowCount * request.config.sentimentDistribution.positive / 100),
            "negative": int(request.config.rowCount * request.config.sentimentDistribution.negative / 100),
            "neutral": int(request.config.rowCount * request.config.sentimentDistribution.neutral / 100)
        }
        logger.info(f"📊 Calculated distribution: {distribution}")
        
        config = {
            "realism": request.config.realism,
            "domainRelevance": request.config.domainRelevance,
            "formality": request.config.formality,
            "lexicalComplexity": request.config.lexicalComplexity,
            "culturalSensitivity": request.config.culturalSensitivity,
            "temperature": request.config.temperature,
            "topP": request.config.topP,
            "maxTokens": request.config.maxTokens,
            "frequencyPenalty": request.config.frequencyPenalty,
            "presencePenalty": request.config.presencePenalty,
            "strict": strict
        }
        logger.info("✅ Configuration prepared")
        
        logger.info("🔄 Starting batch generation...")
        result = await chain.batch_generate(
            domain=request.config.domain,
            distribution=distribution,
            config=config
        )
        
        if not result or not result.get("data"):
            raise HTTPException(status_code=500, detail="Failed to generate data")
            
        logger.info(f"✅ Generation complete. Generated {result['summary']['total']} items")
        
        # Generate unique ID
        request_id = str(uuid.uuid4())
        
        # Start analysis and set initial status
        analysis_status[request_id] = "processing"
        asyncio.create_task(generate_analysis_pdf(request_id, result["data"]))
        
        response_data = {
            "request_id": request_id,
            "generated_data": result["data"],
            "summary": {
                "total_generated": result["summary"]["total"],
                "sentiment_distribution": result["summary"]["distribution"]
            }
        }
        
        logger.info("🏁 Request completed successfully")
        return response_data
        
    except Exception as e:
        logger.error(f"❌ Error in generate_advanced_data: {str(e)}")
        logger.exception("Detailed error trace:")
        # Ensure status is set to error and cleanup is triggered
        if 'request_id' in locals():
            analysis_status[request_id] = "error"
            pdf_cache.pop(request_id, None)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis/status/{request_id}")
async def get_analysis_status(request_id: str):
    """Check the status of PDF generation"""
    status = analysis_status.get(request_id)
    if not status:
        raise HTTPException(status_code=404, detail="Analysis not found or expired")
    
    return {"status": status}

@app.get("/analysis/download/{request_id}")
async def download_analysis(request_id: str):
    """Download the generated PDF analysis"""
    try:
        status = analysis_status.get(request_id)
        if not status:
            raise HTTPException(status_code=404, detail="Analysis not found or expired")
            
        if status == "processing":
            raise HTTPException(status_code=202, detail="Analysis still processing")
            
        if status == "error":
            raise HTTPException(status_code=500, detail="Error generating analysis")
            
        pdf_path = pdf_cache.get(request_id)
        if not pdf_path or not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        # Schedule cleanup after 10 minutes
        asyncio.create_task(cleanup_after_delay(request_id, pdf_path))
        
        return FileResponse(
            path=pdf_path,
            media_type='application/pdf',
            filename=f'synthetic_data_analysis_{request_id}.pdf'
        )
        
    except Exception as e:
        logger.error(f"Error downloading analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def cleanup_after_delay(request_id: str, pdf_path: str, delay: int = 600):
    """Cleanup cache and PDF file after delay"""
    await asyncio.sleep(delay)
    try:
        # Get the analyzer instance if it exists
        if request_id in analysis_status:
            analyzer = SentimentAnalysisReport(data=[], request_id=request_id)  # Empty data as we're just cleaning up
            analyzer.cleanup()
        
        # Clean up cache entries
        pdf_cache.pop(request_id, None)
        analysis_status.pop(request_id, None)
    except Exception as e:
        logger.error(f"Error during cleanup for request {request_id}: {str(e)}")

async def generate_analysis_pdf(request_id: str, data: list):
    """Background task to generate PDF analysis"""
    analyzer = None
    try:
        analyzer = SentimentAnalysisReport(data=data, request_id=request_id)
        pdf_path = analyzer.generate_report()
        pdf_cache[request_id] = pdf_path
        analysis_status[request_id] = "ready"
    except Exception as e:
        logger.error(f"Error generating PDF analysis: {str(e)}")
        analysis_status[request_id] = "error"
        # Ensure cleanup happens on error
        if analyzer:
            analyzer.cleanup()
        # Remove from cache
        pdf_cache.pop(request_id, None)
    finally:
        # Set a timeout for cleanup even if status is 'ready'
        asyncio.create_task(
            cleanup_after_delay(request_id, analyzer.output_path if analyzer else None)
        )      