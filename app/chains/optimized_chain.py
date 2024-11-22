from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from openai import OpenAIError
import asyncio
from concurrent.futures import ThreadPoolExecutor

class GenerationResult(BaseModel):
    text: str = Field(description="Generated feedback text")

class AnalysisResult(BaseModel):
    sentiment: str = Field(description="Sentiment classification")
    confidence: float = Field(description="Confidence score")

class OptimizedGenerationChain:
    def __init__(self, api_key: str, config: Dict[str, Any]):
        self.config = config  
        # Core LLM parameters
        self.llm = ChatOpenAI(
            temperature=config["temperature"],
            model=config["model"],
            top_p=config["topP"],
            max_tokens=config["maxTokens"],
            presence_penalty=config["presencePenalty"],
            frequency_penalty=config["frequencyPenalty"],
            openai_api_key=api_key,
            request_timeout=60,
            max_retries=3
        )
        
        # Store normalized parameters for prompts
        self.generation_params = {
            "domain": config["domain"],
            "realism": config["realism"],
            "domain_relevance": config["domainRelevance"],
            "cultural_sensitivity": config["culturalSensitivity"],
            "formality": config["formality"],
            "lexical_complexity": config["lexicalComplexity"],
            "diversity": config["diversity"],
            "privacy_level": config["privacyLevel"]
        }
        
        self.analysis_params = {
            "sentiment_intensity": config["sentimentIntensity"],
            "bias_control": config["biasControl"],
            "temporal_relevance": config["temporalRelevance"],
            "noise_level": config["noiseLevel"],
            "positive": config["sentimentDistribution"]["positive"],
            "neutral": config["sentimentDistribution"]["neutral"],
            "negative": config["sentimentDistribution"]["negative"]
        }
        
        self.generation_parser = JsonOutputParser(pydantic_object=List[GenerationResult])
        self.analysis_parser = JsonOutputParser(pydantic_object=AnalysisResult)
        self.batch_size = 50
        self.max_concurrent_batches = 3

    async def _process_generation_batch(self, batch_size: int, start_id: int) -> List[Dict[str, Any]]:
        retries = 3
        backoff = 1
        
        while retries > 0:
            try:
                response = await self.llm.ainvoke(
                    self.generation_prompt.format(
                        count=batch_size,
                        domain=self.config["domain"],
                        privacy_level=self.config["privacyLevel"],
                        realism=self.config["realism"],
                        domain_relevance=self.config["domainRelevance"],
                        cultural_sensitivity=self.config["culturalSensitivity"],
                        formality=self.config["formality"],
                        lexical_complexity=self.config["lexicalComplexity"],
                        diversity=self.config["diversity"]
                    )
                )
                
                parsed = self.generation_parser.parse(response.content)
                return [{"id": start_id + i, "text": item.text} for i, item in enumerate(parsed)]
                
            except OpenAIError as e:
                retries -= 1
                if retries > 0:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    raise 