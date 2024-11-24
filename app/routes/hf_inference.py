from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, AsyncGenerator, Tuple
import requests
import asyncio
import math
from datetime import datetime
import json
import os
from app.config import get_settings
from app.utils.auth import verify_api_key
from app.models.hf_models import HuggingFaceRequest, LargeAnalysisRequest

router = APIRouter(
    prefix="/hf",
    tags=["huggingface-inference"],
)

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
            
        # Use .get() with defaults for safety
        label = sentiment_result.get("label", "")
        score = sentiment_result.get("score", 0.0)
        
        return label == "POSITIVE", score
    except Exception as e:
        print(f"Error formatting result: {str(e)}, raw result: {result}")
        return False, 0.0

@router.post("/analyze-batch")
async def analyze_text_batch_hf(
    request: HuggingFaceRequest,
    _: str = Depends(verify_api_key)
):
    BATCH_SIZE = 16 
    DAILY_LIMIT = 20000  # PRO account limit,If you guys use normal account you can use 1000

    headers = {"Authorization": f"Bearer {get_settings().HUGGINGFACE_API_KEY}"}
    
    texts = [item["text"] for item in request.generated_data]
    total_texts = len(texts)
    total_batches = (total_texts + BATCH_SIZE - 1) // BATCH_SIZE

    if total_batches > DAILY_LIMIT:
        return {
            "message": "Batch too large for daily limit",
            "batch_info": {
                "total_texts": total_texts,
                "batch_size": BATCH_SIZE,
                "required_api_calls": total_batches,
                "daily_limit": DAILY_LIMIT,
                "recommended_splits": math.ceil(total_batches / DAILY_LIMIT)
            }
        }

    async def process_batch(batch: List[str]) -> List[Dict]:
        try:
            response = requests.post(get_settings().TC_API_URL, headers=headers, json={"inputs": batch})
            results = response.json()
            
            if isinstance(results, dict) and "error" in results:
                raise HTTPException(status_code=500, detail=results["error"])
                
            return results
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Process texts in batches of 8 for optimal performance
    BATCH_SIZE = 8
    batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    
    all_results = []
    for batch in batches:
        batch_results = await process_batch(batch)
        all_results.extend(batch_results)
    
    analyzed_data = []
    for idx, (item, result) in enumerate(zip(request.generated_data, all_results)):
        is_positive = result[0]["label"] == "POSITIVE" if isinstance(result, list) else result["label"] == "POSITIVE"
        score = result[0]["score"] if isinstance(result, list) else result["score"]
        
        analyzed_data.append({
            "id": idx + 1,
            "text": item["text"],
            "sentiment": "positive" if is_positive else "negative",
            "confidence": score
        })
    
    sentiment_counts = {
        "positive": sum(1 for x in analyzed_data if x["sentiment"] == "positive"),
        "negative": sum(1 for x in analyzed_data if x["sentiment"] == "negative")
    }
    
    return {
        "analyzed_data": analyzed_data,
        "summary": {
            "total_analyzed": len(analyzed_data),
            "sentiment_distribution": sentiment_counts
        }
    }

@router.post("/analyze-batch-quick")
async def analyze_text_batch_hf_quick(
    request: HuggingFaceRequest,
    _: str = Depends(verify_api_key)
):
    BATCH_SIZE = 32
    MAX_CONCURRENT_REQUESTS = 5
    headers = {"Authorization": f"Bearer {get_settings().HUGGINGFACE_API_KEY}"}
    
    texts = [item["text"] for item in request.generated_data]
    total_texts = len(texts)
    
    if total_texts > 10_000:
        raise HTTPException(
            status_code=400,
            detail="This endpoint is limited to 10,000 texts. Use /analyze-batch-large for larger datasets"
        )

    async def process_batch(batch: List[str], batch_idx: int) -> List[Dict]:
        try:
            response = requests.post(get_settings().TC_API_URL, headers=headers, json={"inputs": batch})
            if response.status_code == 429:  # Rate limit
                await asyncio.sleep(1)
                return await process_batch(batch, batch_idx)
                
            results = response.json()
            
            print(f"Batch {batch_idx}: Input size: {len(batch)}, Output size: {len(results) if isinstance(results, list) else 1}")
            
            if isinstance(results, dict):
                return [results] * len(batch)
            return results
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            return [{"label": "POSITIVE", "score": 0.0}] * len(batch)

    batches = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batches.append((batch, i // BATCH_SIZE))
    
    all_results = []
    for i in range(0, len(batches), MAX_CONCURRENT_REQUESTS):
        current_batches = batches[i:i + MAX_CONCURRENT_REQUESTS]
        batch_results = await asyncio.gather(*[
            process_batch(batch, idx)
            for batch, idx in current_batches
        ])
        
        for batch_result in batch_results:
            all_results.extend(batch_result)
            
        if i + MAX_CONCURRENT_REQUESTS < len(batches):
            await asyncio.sleep(0.5)
    
    if len(all_results) != total_texts:
        print(f"Warning: Results mismatch. Expected {total_texts}, got {len(all_results)}")
        while len(all_results) < total_texts:
            all_results.append({"label": "POSITIVE", "score": 0.0})
    
    analyzed_data = []
    for idx, (item, result) in enumerate(zip(request.generated_data, all_results)):
        is_positive, score = await format_sentiment_result(result)
        
        analyzed_data.append({
            "id": idx + 1,
            "text": item["text"],
            "sentiment": "positive" if is_positive else "negative",
            "confidence": score,
            "processing_info": {
                "raw_response_type": type(result).__name__,
                "batch_index": idx // BATCH_SIZE,
                "was_processed": True
            }
        })
    
    sentiment_counts = {
        "positive": sum(1 for x in analyzed_data if x["sentiment"] == "positive"),
        "negative": sum(1 for x in analyzed_data if x["sentiment"] == "negative")
    }
    
    return {
        "analyzed_data": analyzed_data,
        "summary": {
            "total_analyzed": len(analyzed_data),
            "sentiment_distribution": sentiment_counts,
            "processing_info": {
                "batch_size": BATCH_SIZE,
                "concurrent_requests": MAX_CONCURRENT_REQUESTS,
                "total_batches": len(batches),
                "input_size": total_texts,
                "output_size": len(analyzed_data),
                "batches_processed": len(batches)
            }
        }
    }

@router.post("/analyze-batch-large")
async def analyze_text_batch_hf_large(
    request: LargeAnalysisRequest,
    _: str = Depends(verify_api_key)
):
    BATCH_SIZE = 16
    CHUNK_SIZE = 1000
    headers = {"Authorization": f"Bearer {get_settings().HUGGINGFACE_API_KEY}"}
    
    texts = [item["text"] for item in request.generated_data]
    total_texts = len(texts)
    
    batch_id = request.batch_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_file = f"progress_{batch_id}.json"
    
    start_index = 0
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            start_index = progress['processed_count']
    
    async def process_chunk(chunk_texts: List[str], chunk_start: int) -> List[Dict]:
        chunk_results = []
        batches = [chunk_texts[i:i + BATCH_SIZE] for i in range(0, len(chunk_texts), BATCH_SIZE)]
        
        for batch in batches:
            try:
                response = requests.post(get_settings().TC_API_URL, headers=headers, json={"inputs": batch})
                if response.status_code == 429:
                    await asyncio.sleep(1)
                    continue
                
                results = response.json()
                print(f"Chunk {chunk_start}, Batch size: {len(batch)}, Response size: {len(results) if isinstance(results, list) else 1}")
                
                if isinstance(results, dict):
                    if "error" in results:
                        print(f"Error in batch response: {results['error']}")
                        chunk_results.extend([{"label": "POSITIVE", "score": 0.0}] * len(batch))
                    else:
                        chunk_results.extend([results] * len(batch))
                else:
                    chunk_results.extend(results)
                
                expected_length = len(batch)
                actual_length = len(chunk_results) - (len(chunk_results) - expected_length)
                if actual_length != expected_length:
                    print(f"Warning: Batch size mismatch. Expected {expected_length}, got {actual_length}")
                    while len(chunk_results) < expected_length:
                        chunk_results.append({"label": "POSITIVE", "score": 0.0})
                
                if request.save_progress:
                    progress = {
                        'batch_id': batch_id,
                        'processed_count': chunk_start + len(chunk_results),
                        'total_count': total_texts,
                        'last_update': datetime.now().isoformat(),
                        'failed_batches': []
                    }
                    with open(progress_file, 'w') as f:
                        json.dump(progress, f)
                        
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                chunk_results.extend([{"label": "POSITIVE", "score": 0.0}] * len(batch))
                if request.save_progress:
                    progress['failed_batches'].append({
                        'start_idx': chunk_start,
                        'texts': batch,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                continue
                
        return chunk_results

    all_results = []
    for chunk_start in range(start_index, total_texts, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, total_texts)
        chunk_texts = texts[chunk_start:chunk_end]
        chunk_results = await process_chunk(chunk_texts, chunk_start)
        all_results.extend(chunk_results)
        await asyncio.sleep(1)
    
    analyzed_data = []
    for idx, (item, result) in enumerate(zip(request.generated_data[start_index:], all_results)):
        is_positive, score = await format_sentiment_result(result)
        
        analyzed_data.append({
            "id": start_index + idx + 1,
            "text": item["text"],
            "sentiment": "positive" if is_positive else "negative",
            "confidence": score,
            "processing_info": {
                "raw_response_type": type(result).__name__,
                "was_processed": True
            }
        })
    
    sentiment_counts = {
        "positive": sum(1 for x in analyzed_data if x["sentiment"] == "positive"),
        "negative": sum(1 for x in analyzed_data if x["sentiment"] == "negative")
    }
    
    if len(analyzed_data) == total_texts and os.path.exists(progress_file):
        os.remove(progress_file)
    
    summary = {
        "total_analyzed": len(analyzed_data),
        "sentiment_distribution": sentiment_counts,
        "processing_info": {
            "batch_id": batch_id,
            "total_processed": len(analyzed_data),
            "total_remaining": total_texts - len(analyzed_data),
            "is_complete": len(analyzed_data) == total_texts
        }
    }
    
    if request.stream_progress:
        return StreamingResponse(
            generate_progress_stream(analyzed_data, summary),
            media_type="application/x-ndjson"
        )
    
    return {
        "analyzed_data": analyzed_data,
        "summary": summary
    }

async def generate_progress_stream(analyzed_data: List[Dict], summary: Dict) -> AsyncGenerator[str, None]:
    yield json.dumps({
        "type": "progress",
        "data": {
            "processed": 0,
            "total": len(analyzed_data),
            "current_status": "Starting analysis..."
        }
    }) + "\n"
    
    PROGRESS_UPDATE_FREQUENCY = 1000
    for i in range(0, len(analyzed_data), PROGRESS_UPDATE_FREQUENCY):
        yield json.dumps({
            "type": "progress",
            "data": {
                "processed": i,
                "total": len(analyzed_data),
                "current_status": f"Processing items {i} to {min(i + PROGRESS_UPDATE_FREQUENCY, len(analyzed_data))}",
                "sentiment_distribution": {
                    "positive": sum(1 for x in analyzed_data[:i] if x["sentiment"] == "positive"),
                    "negative": sum(1 for x in analyzed_data[:i] if x["sentiment"] == "negative")
                }
            }
        }) + "\n"
        await asyncio.sleep(0.01)
