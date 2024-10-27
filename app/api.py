from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List, Optional, Dict
from app.chains.sentiment_chain import SentimentAnalysisChain
from app.config import get_settings
from app.utils.cost_tracker import track_cost
from app.chains.generation_chain import DataGenerationChain, MAX_RECORDS

app = FastAPI(title="Sentiment Data Handling & Analysis")


async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != get_settings().API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return x_api_key


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
    return {"status": "healthy"}


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
    verbose: bool = True  # New field


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
