from fastapi import APIRouter, HTTPException
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from concurrent.futures import ThreadPoolExecutor
import asyncio
import math
from app.config import get_settings

router = APIRouter(
    prefix="/local",
    tags=["local-inference"],
    responses={404: {"description": "Not found"}},
)

# Initialize model with specific cache directory
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
cache_dir = get_settings().MODEL_CACHE_DIR

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    cache_dir=cache_dir
)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    cache_dir=cache_dir
)

# Check if CUDA is available and move model to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Request/Response models
class LocalAnalysisRequest(BaseModel):
    generated_data: List[Dict[str, Any]]
    batch_size: Optional[int] = 32
    max_workers: Optional[int] = 8  

class AnalysisResponse(BaseModel):
    analyzed_data: List[Dict]
    summary: Dict

def process_batch(batch: List[str]) -> List[Dict]:
    try:
        # Calculate optimal batch size based on available memory
        max_sequence_length = 512
        batch_size = min(
            len(batch),
            math.ceil(torch.cuda.get_device_properties(0).total_memory * 0.7 / (max_sequence_length * 4)) if torch.cuda.is_available() else 32
        )
        
        results = []
        # Process in optimal sub-batches if needed
        for i in range(0, len(batch), batch_size):
            sub_batch = batch[i:i + batch_size]
            sub_results = sentiment_analyzer(sub_batch, truncation=True, max_length=max_sequence_length)
            results.extend(sub_results)
            
            # Clear CUDA cache if using GPU (to prevent memory leak)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return results
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        return [{"label": "POSITIVE", "score": 0.0}] * len(batch)

@router.post("/analyze-batch")
async def analyze_text_batch_local(request: LocalAnalysisRequest):
    texts = [item["text"] for item in request.generated_data]
    total_texts = len(texts)
    batch_size = request.batch_size
    
    # Create batches
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    # Process batches using ThreadPoolExecutor
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=request.max_workers) as executor:
        # Process all batches concurrently
        batch_futures = [
            loop.run_in_executor(executor, process_batch, batch)
            for batch in batches
        ]
        all_results = []
        for future in asyncio.as_completed(batch_futures):
            batch_result = await future
            all_results.extend(batch_result)
    
    # Format results
    analyzed_data = []
    for idx, (item, result) in enumerate(zip(request.generated_data, all_results)):
        analyzed_data.append({
            "id": idx + 1,
            "text": item["text"],
            "sentiment": "positive" if result["label"] == "POSITIVE" else "negative",
            "confidence": result["score"],
            "processing_info": {
                "batch_index": idx // batch_size,
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
                "batch_size": batch_size,
                "max_workers": request.max_workers,
                "total_batches": len(batches),
                "input_size": total_texts,
                "output_size": len(analyzed_data)
            }
        }
    }

# For using the hardware hosting endpoints and verifying the hardware
@router.get("/device-info")
async def get_device_info():
    return {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_info": {
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "memory_allocated": f"{torch.cuda.memory_allocated(0)/1024**2:.2f}MB" if torch.cuda.is_available() else None,
            "memory_cached": f"{torch.cuda.memory_reserved(0)/1024**2:.2f}MB" if torch.cuda.is_available() else None
        } if torch.cuda.is_available() else None,
        "cpu_threads": torch.get_num_threads(),
        "optimal_batch_size": math.ceil(torch.cuda.get_device_properties(0).total_memory * 0.7 / (512 * 4)) if torch.cuda.is_available() else 32
    }