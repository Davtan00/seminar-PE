from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from app.chains.sentiment_chain import SentimentAnalysisChain
from app.config import get_settings, get_security_manager
from app.utils.cost_tracker import track_cost
from app.chains.generation_chain import DataGenerationChain, MAX_RECORDS
from app.chains.text_analysis_chain import TextAnalysisChain
from datetime import datetime, timedelta
from langchain.chat_models import ChatOpenAI
from app.chains.optimized_chain import OptimizedGenerationChain
from datetime import datetime
import hmac
import hashlib
import base64
import json
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import os
from fastapi.middleware.cors import CORSMiddleware
import binascii

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

# Add this new request model
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

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != get_settings().API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return x_api_key


class GenerationConfig(BaseModel):
    sentimentDistribution: Dict[str, int]
    rowCount: int
    domain: str
    temperature: float
    topP: float
    maxTokens: int
    frequencyPenalty: float
    presencePenalty: float
    model: str
    privacyLevel: float
    biasControl: float
    sentimentIntensity: float
    realism: float
    domainRelevance: float
    diversity: float
    temporalRelevance: float
    noiseLevel: float
    culturalSensitivity: float
    formality: float
    lexicalComplexity: float

class SecureGenerationRequest(BaseModel):
    encryptedKey: str
    config: GenerationConfig
    timestamp: int

ENCRYPTION_KEY_HEX = "1d443b0800609907c95819d255084b2341c955020ccb6a04"  # Example: 32 hexadecimal chars for a 16-byte key (128 bits)
ENCRYPTION_KEY = binascii.unhexlify(ENCRYPTION_KEY_HEX)

def decrypt_api_key(encrypted_key: str, encryption_key: bytes) -> str:
    try:
        # Decode the encrypted key from Base64
        encrypted_data = base64.b64decode(encrypted_key)

        # Extract the IV (first 16 bytes) and the ciphertext
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]

        # Initialize AES cipher in CBC mode with the provided IV
        cipher = AES.new(encryption_key, AES.MODE_CBC, iv)

        # Decrypt and unpad the ciphertext
        decrypted_bytes = unpad(cipher.decrypt(ciphertext), AES.block_size)
        return decrypted_bytes.decode('utf-8')
    except (ValueError, KeyError) as e:
        # Handle padding errors or incorrect key errors
        raise HTTPException(status_code=400, detail="Decryption failed: Incorrect key or padding.")


@app.post("/secure-generate")
async def secure_generate(
    request: SecureGenerationRequest,
    x_api_key: str = Header(...)
):
    current_time = int(datetime.now().timestamp() * 1000)
    if current_time - request.timestamp > 900000:  # 15 minutes validity
        raise HTTPException(status_code=401, detail="Request expired")

    try:
        decrypted_api_key = decrypt_api_key(request.encryptedKey, ENCRYPTION_KEY)

        return {
            "message": "Request processed successfully",
            "decrypted_api_key": decrypted_api_key,
            "config": request.config
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Couldnt fix it on friday so I will leave this on a branch and try using the huggingface PRO interference api and see how that goes
@app.post("/mocked-secure-generate")
async def mocked_secure_generate(
    request: SecureGenerationRequest,
    x_api_key: str = Depends(verify_api_key)
):
    security_manager = get_security_manager()
    
    

    try:
        # Mocked API key instead of decrypting
        api_key = os.getenv('OPENAI_API_KEY')  # Replace with your actual OpenAI key
        
        # Initialize the OptimizedGenerationChain with the mocked API key and request configuration
        chain = OptimizedGenerationChain(api_key, request.config.dict())
        
        # Generate and analyze data
        result = await chain.generate_and_analyze()
        
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))