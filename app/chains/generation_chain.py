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
            max_tokens=8000,  
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
        use_bad_prompt: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        count = min(count, MAX_RECORDS)
        batch_size = 400 if not verbose else 200  # Larger batches with 128k context

        try:
            results = []
            warnings = []
            tasks = []

            # Process batches concurrently
            for i in range(0, count, batch_size):
                batch_count = min(batch_size, count - i)
                tasks.append(self._generate_batch(
                    domain, 
                    batch_count, 
                    sentiment_distribution, 
                    use_bad_prompt,  # Add this parameter
                    verbose
                ))
            
            batch_results = await asyncio.gather(*tasks)
            
            # Process results
            for batch in batch_results:
                if "data" in batch:
                    results.extend(batch["data"])
                if "warnings" in batch:
                    warnings.extend(batch["warnings"])

            return {
                "data": [{"id": i + 1, **item} for i, item in enumerate(results)],
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
        use_bad_prompt: bool,  # Add this parameter
        verbose: bool
    ) -> Dict[str, Any]:
        retries = 3
        while retries > 0:
            try:
                return await self._process_batch(
                    domain, 
                    batch_count, 
                    sentiment_distribution,
                    use_bad_prompt,  # Add this parameter 
                    verbose
                )
            except OpenAIError as e:
                retries -= 1
                if retries == 0:
                    raise
                await asyncio.sleep(2 ** (3 - retries))  # Exponential backoff

    async def _process_batch(
        self,
        domain: str,
        batch_count: int,
        sentiment_distribution: Optional[Dict[str, float]],
        use_bad_prompt: bool,  # Add this parameter
        verbose: bool
    ) -> Dict[str, Any]:
        if use_bad_prompt:
            prompt = create_bad_generation_prompt()
        else:
            prompt = (create_generation_prompt if verbose else create_simple_generation_prompt)(
                domain=domain,
                count=batch_count,
                sentiment_distribution=sentiment_distribution
            )
       
        # Print the formatted prompt
        formatted_messages = prompt.format_messages(
            domain=domain,
            count=batch_count,
            sentiment_distribution=sentiment_distribution
        )
        # Print the formatted prompt for playground
        print("\n=== FORMATTED PROMPT ===")
        for message in formatted_messages:
            print(f"\n--- {message.type.upper()} MESSAGE ---")
            print(message.content)
        print("\n==================\n")
       
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
        
        try:
            batch_results = await chain.ainvoke(domain)
            return {
                "data": [{"id": i + 1, **item} for i, item in enumerate(batch_results)],
                "warnings": self._validate_results(batch_results, batch_count, sentiment_distribution)
            }
        except Exception as e:
            # Add more specific error handling
            if "Invalid json output" in str(e):
                # Try to clean and parse the output
                try:
                    # Get the raw output before parsing
                    raw_output = str(e).split("Invalid json output: ")[1]
                    # Clean up common JSON formatting issues
                    cleaned_output = raw_output.replace("'", '"').replace(
                        'sentiment:', '"sentiment":').replace(
                        '\n', '').replace('    ', '')
                    import json
                    batch_results = [
                        SentimentRecord(**item) 
                        for item in json.loads(cleaned_output)
                    ]
                    return {
                        "data": [{"id": i + 1, **item.dict()} 
                                for i, item in enumerate(batch_results)],
                        "warnings": self._validate_results(
                            batch_results, batch_count, sentiment_distribution
                        )
                    }
                except Exception as parse_error:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to parse output: {str(parse_error)}"
                    )
            raise

    def _validate_results(
        self,
        results: List[Dict[str, Any]],
        count: int,
        sentiment_distribution: Optional[Dict[str, float]] = None
    ) -> List[str]:
        warnings = []
       
        # Validate count
        if len(results) != count:
            warnings.append(f"Incomplete results: Expected {count} records, got {len(results)}")
       
        # Validate sentiment distribution if specified
        if sentiment_distribution:
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
