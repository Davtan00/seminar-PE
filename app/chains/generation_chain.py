from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from app.prompts.generation_prompts import create_generation_prompt, create_bad_generation_prompt
from app.config import get_settings
from fastapi import HTTPException
from typing import Dict, Any, List, Optional
from openai import OpenAIError

MAX_RECORDS = 1000  # Safety limit since still not project API key
#gpt-3.5-turbo-16k or gpt-4o-mini seems to produce similar results tbh.
class DataGenerationChain:
    def __init__(self):
        settings = get_settings()
        self.llm = ChatOpenAI(
            temperature=0.5,   # Higher temperature for more variety, to make the results a bit more interesting
            model="gpt-4o-mini",
            request_timeout=300,
            max_retries=1,
            openai_api_key=settings.OPENAI_API_KEY
        )
    
    async def generate(
        self,
        domain: str,
        count: int,
        sentiment_distribution: Optional[Dict[str, float]] = None,
        use_bad_prompt: bool = False
    ) -> Dict[str, Any]:  # Changed return type to include status
        count = min(count, MAX_RECORDS)  # Ensure count doesn't exceed limit
        
        try:
            prompt = create_bad_generation_prompt() if use_bad_prompt else create_generation_prompt(
                domain=domain,
                count=count,
                sentiment_distribution=sentiment_distribution
            )
            
            # Print the formatted prompt
            formatted_messages = prompt.format_messages(
                domain=domain,
                count=count,
                sentiment_distribution=sentiment_distribution
            )
            
            print("\n=== FORMATTED PROMPT ===")
            for message in formatted_messages:
                print(f"\n--- {message.type.upper()} MESSAGE ---")
                print(message.content)
            print("\n==================\n")
            
            chain = (
                {
                    "domain": RunnablePassthrough(),
                    "count": lambda x: count,
                    "sentiment_distribution": lambda x: sentiment_distribution
                }
                | prompt
                | self.llm
                | JsonOutputParser(pydantic_object=None)  # More permissive parsing
            )
            
            results = await chain.ainvoke(domain)
            
            # Get validation warnings instead of raising errors
            validation_warnings = self._validate_results(results, count, sentiment_distribution)
            
            return {
                "data": [{"id": i + 1, **item} for i, item in enumerate(results)],
                "warnings": validation_warnings
            }
        except OpenAIError as e:
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating data: {str(e)}")

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
