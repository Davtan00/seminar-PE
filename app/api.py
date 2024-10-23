from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List, Optional, Dict
from app.chains.sentiment_chain import SentimentAnalysisChain
from app.config import get_settings

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
    
    for item in request.texts:
        try:
            result = await chain.analyze(item.text, item.domain)
            total_cost += result["cost_info"]["total_cost"]
            results.append(result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return {
        "results": results,
        "batch_metrics": {
            "total_cost_usd": total_cost,
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
    
    for item in request.raw_texts:
        try:
            result = await chain.analyze(item.text, item.domain)
            total_cost += result["cost_info"]["total_cost"]
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
            "total_cost_usd": total_cost
        }
    }
