from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Optional, List, Literal
from app.chains.local_generation_chain import LocalGenerationChain
from app.chains.hf_generation_chain import HFGenerationChain
from app.routes.local_inference import analyze_text_batch_local, LocalAnalysisRequest
from app.routes.hf_inference import analyze_text_batch_hf 

router = APIRouter(
    prefix="/generate",
    tags=["generation"],
)

class GenerationRequest(BaseModel):
    domain: str
    count: int
    batch_size: Optional[int] = 32
    max_workers: Optional[int] = 4

@router.post("/local")
async def generate_and_analyze_local(
    request: GenerationRequest,
    analyze: bool = True
):
    try:
        # Step 1: Generate reviews
        generation_chain = LocalGenerationChain()
        generation_result = await generation_chain.generate(
            domain=request.domain,
            count=request.count,
            max_workers=request.max_workers
        )

        # If analyze=False, return only the generated data
        if not analyze:
            return {
                "generated_data": generation_result["generated_data"],
                "summary": generation_result["summary"]
            }

        # Step 2: Analyze sentiment using existing endpoint
        analysis_request = LocalAnalysisRequest(
            generated_data=generation_result["generated_data"],
            batch_size=request.batch_size,
            max_workers=request.max_workers
        )
        
        analysis_result = await analyze_text_batch_local(analysis_request)
        
        return {
            "generated_and_analyzed_data": analysis_result["analyzed_data"],
            "summary": {
                **generation_result["summary"],
                **analysis_result["summary"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

@router.post("/huggingface")
async def generate_and_analyze_hf(
    request: GenerationRequest,
    analyze: bool = True,
    model: str = "gpt2"  # Allow model selection
):
    try:
        # Step 1: Generate reviews using HF
        generation_chain = HFGenerationChain()
        generation_result = await generation_chain.generate(
            domain=request.domain,
            count=request.count,
            batch_size=request.batch_size
        )

        # If analyze=False, return only the generated data
        if not analyze:
            return {
                "generated_data": generation_result["generated_data"],
                "summary": generation_result["summary"]
            }

        # Step 2: Analyze sentiment using existing HF endpoint
        analysis_result = await analyze_text_batch_hf(
            texts=[item["text"] for item in generation_result["generated_data"]],
            batch_size=request.batch_size
        )
        
        # Combine the results
        analyzed_data = []
        for gen_item, analysis in zip(generation_result["generated_data"], analysis_result["analyzed_data"]):
            analyzed_data.append({
                **gen_item,
                "sentiment": analysis["sentiment"],
                "confidence": analysis["confidence"]
            })

        return {
            "generated_and_analyzed_data": analyzed_data,
            "summary": {
                **generation_result["summary"],
                **analysis_result["summary"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Add an endpoint that allows choosing the generation and analysis providers
@router.post("/hybrid")
async def generate_and_analyze_hybrid(
    request: GenerationRequest,
    generator: Literal["local", "huggingface"] = "local",
    analyzer: Literal["local", "huggingface"] = "local",
    analyze: bool = True
):
    try:
        # Step 1: Generate using selected provider
        generation_chain = LocalGenerationChain() if generator == "local" else HFGenerationChain()
        generation_result = await generation_chain.generate(
            domain=request.domain,
            count=request.count,
            batch_size=request.batch_size,
            max_workers=request.max_workers if generator == "local" else None
        )

        if not analyze:
            return {
                "generated_data": generation_result["generated_data"],
                "summary": generation_result["summary"]
            }

        # Step 2: Analyze using selected provider
        if analyzer == "local":
            analysis_result = await analyze_text_batch_local(LocalAnalysisRequest(
                generated_data=generation_result["generated_data"],
                batch_size=request.batch_size,
                max_workers=request.max_workers
            ))
        else:
            analysis_result = await analyze_text_batch_hf(
                texts=[item["text"] for item in generation_result["generated_data"]],
                batch_size=request.batch_size
            )

        return {
            "generated_and_analyzed_data": analysis_result["analyzed_data"],
            "summary": {
                **generation_result["summary"],
                **analysis_result["summary"],
                "providers": {
                    "generator": generator,
                    "analyzer": analyzer
                }
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 