from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from app.prompts.generation_prompts import create_generation_prompt, create_bad_generation_prompt, create_simple_generation_prompt
from app.config import get_settings
from fastapi import HTTPException
from typing import Dict, Any, List, Optional
from openai import OpenAIError
import asyncio


MAX_RECORDS = 10000  
#gpt-3.5-turbo-16k or gpt-4o-mini seems to produce similar results tbh.
class DataGenerationChain:
    def __init__(self):
        settings = get_settings()
        self.llm = ChatOpenAI(
            temperature=0.7,  
            model="gpt-4o-mini",  # Good price/performance model choice with 128k context window
            request_timeout=900,  # Increased timeout
            max_retries=5,  # Increased max retries
            # Can set higher max_tokens due to 128k context window
            max_tokens=16384,  # Increased significantly, max allowed
            presence_penalty=0.1,    # Reduce repetition
            frequency_penalty=0.1,   # Encourage diversity
            openai_api_key=settings.OPENAI_API_KEY
        )
   
        # Create a more robust parser setup
        base_parser = JsonOutputParser(pydantic_object=List[SentimentRecord])
        self.parser = OutputFixingParser.from_llm(
            parser=base_parser,
            llm=self.llm
        )

    async def generate(
        self,
        domain: str,
        count: int,
        sentiment_distribution: Optional[Dict[str, float]] = None,
        use_bad_prompt: bool = False, # Add this parameter if you want bad data
        verbose: bool = True # Add this param as false to maximise output content
    ) -> Dict[str, Any]:
        count = min(count, MAX_RECORDS)
        # Increase batch size significantly
        batch_size = 100 if not verbose else 50  # Handle more records per batch

        try:
            results = []
            warnings = []
            current_id = 1  # Track the current ID
            
            # Use fewer, larger batches
            remaining_count = count
            while remaining_count > 0:
                current_batch_size = min(batch_size, remaining_count)
                
                batch_result = await self._generate_batch(
                    domain, 
                    current_batch_size,
                    sentiment_distribution, 
                    use_bad_prompt,
                    verbose,
                    start_id=current_id
                )
                
                if "data" in batch_result:
                    results.extend(batch_result["data"])
                    current_id += len(batch_result["data"])
                    remaining_count -= len(batch_result["data"])
                
                if "warnings" in batch_result:
                    warnings.extend(batch_result["warnings"])
                
                # Break if we're not making progress
                if not batch_result.get("data"):
                    break

            return {
                "data": results,
                "warnings": warnings
            }

        except OpenAIError as e:
            await self._handle_rate_limits(e)
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating data: {str(e)}")

    async def _handle_rate_limits(self, error: OpenAIError):
        if "rate_limit" in str(error).lower():
            await asyncio.sleep(20)  # Basic backoff
        raise error

    async def _generate_batch(
        self,
        domain: str,
        batch_count: int,
        sentiment_distribution: Optional[Dict[str, float]],
        use_bad_prompt: bool,
        verbose: bool,
        start_id: int  # Add start_id parameter
    ) -> Dict[str, Any]:
        retries = 3
        while retries > 0:
            try:
                batch_result = await self._process_batch(
                    domain,
                    batch_count,
                    sentiment_distribution,
                    use_bad_prompt,
                    verbose,
                    start_id  # Pass start_id to _process_batch
                )
                return batch_result
            except OpenAIError as e:
                retries -= 1
                if retries == 0:
                    raise
                await asyncio.sleep(2 ** (3 - retries))

    async def _process_batch(
        self,
        domain: str,
        batch_count: int,
        sentiment_distribution: Optional[Dict[str, float]],
        use_bad_prompt: bool,
        verbose: bool,
        start_id: int  # Add start_id parameter
    ) -> Dict[str, Any]:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if use_bad_prompt:
                    prompt = create_bad_generation_prompt()
                else:
                    prompt = (create_generation_prompt if verbose else create_simple_generation_prompt)(
                        domain=domain,
                        count=batch_count,
                        sentiment_distribution=sentiment_distribution
                    )
                
                # Add explicit instructions about batch size
                messages = prompt.format_messages(
                    domain=domain,
                    count=batch_count,
                    sentiment_distribution=sentiment_distribution
                )
                
                chain = (
                    {
                        "domain": RunnablePassthrough(),
                        "count": lambda x: batch_count,
                        "sentiment_distribution": lambda x: sentiment_distribution
                    }
                    | prompt
                    | self.llm
                    | self.parser
                )
                
                batch_results = await chain.ainvoke(domain)
                
                # Guard against empty results
                if not batch_results:
                    if attempt < max_retries - 1:
                        continue
                    return {
                        "data": [],
                        "warnings": [f"Failed to generate any results after {max_retries} attempts"]
                    }
                
                # Validate batch size
                if len(batch_results) != batch_count:
                    if attempt < max_retries - 1:
                        continue
                
                return {
                    "data": [{"id": start_id + i, **item} for i, item in enumerate(batch_results)],
                    "warnings": self._validate_results(batch_results, batch_count, sentiment_distribution)
                }
                
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing batch: {str(e)}"
                )

    def _validate_results(
        self,
        results: List[Dict[str, Any]],
        count: int,
        sentiment_distribution: Optional[Dict[str, float]] = None
    ) -> List[str]:
        warnings = []
       
        # Guard against empty results
        if not results:
            warnings.append(f"No results generated: Expected {count} records, got 0")
            return warnings
       
        # Validate count
        if len(results) != count:
            warnings.append(f"Incomplete results: Expected {count} records, got {len(results)}")
       
        # Validate sentiment distribution if specified
        if sentiment_distribution and results:  # Only check if we have results
            actual_distribution = {
                "positive": sum(1 for x in results if x["sentiment"] == "positive") / len(results),
                "neutral": sum(1 for x in results if x["sentiment"] == "neutral") / len(results),
                "negative": sum(1 for x in results if x["sentiment"] == "negative") / len(results)
            }
           
            # Allow for a 5% margin of error in distribution
            margin = 0.05
            for sentiment, expected_ratio in sentiment_distribution.items():
                actual_ratio = actual_distribution.get(sentiment, 0)
                if abs(actual_ratio - expected_ratio) > margin:
                    warnings.append(
                        f"Sentiment distribution mismatch for {sentiment}: "
                        f"expected {expected_ratio:.1%}, got {actual_ratio:.1%}"
                    )
       
        return warnings

# Define Pydantic model for structured output
class SentimentRecord(BaseModel):
    text: str = Field(description="The generated review/feedback text")
    sentiment: str = Field(description="One of: positive, neutral, negative")
