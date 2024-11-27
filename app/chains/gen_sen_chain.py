from fastapi import HTTPException
import aiohttp
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
import time

logger = logging.getLogger(__name__)
### Only so fast because of rate limit, but we aren't pushing it so hard so perhaps T2,T1 would be okay performance wise as well.
class RateLimiter:
    def __init__(self):
        self.requests_limit = 5000
        self.tokens_limit = 4000000
        self.remaining_requests = 5000
        self.remaining_tokens = 4000000
        self.last_update = time.time()

    def update_limits(self, headers: Dict[str, str]):
        self.remaining_requests = int(headers.get('x-ratelimit-remaining-requests', self.remaining_requests))
        self.remaining_tokens = int(headers.get('x-ratelimit-remaining-tokens', self.remaining_tokens))
        self.last_update = time.time()

    async def wait_if_needed(self):
        if self.remaining_requests < 10 or self.remaining_tokens < 1000:
            await asyncio.sleep(0.1)  # Small delay to allow for rate limit reset

class GenSenChain:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.rate_limiter = RateLimiter()
        
        self.system_prompt = """You are a specialized content generator that creates realistic reviews or comments.
            Your output must be a JSON object with a 'reviews' array containing the generated items.
            Each item must have 'text' and 'sentiment' fields."""

    async def process_batch(
        self,
        session: aiohttp.ClientSession,
        batch_params: Dict[str, Any],
        retries: int = 3
    ) -> Optional[List[Dict[str, Any]]]:
        while retries > 0:
            try:
                await self.rate_limiter.wait_if_needed()
                
                # Log the batch parameters for debugging
                logger.debug(f"Processing batch with parameters: {batch_params}")
                
                user_prompt = f"""Generate {batch_params['count']} {batch_params['sentiment']} reviews for the {batch_params['domain']} domain.
                    Apply these style parameters:
                    - Realism: {batch_params['realism']}
                    - Domain relevance: {batch_params['domainrelevance']}
                    - Formality: {batch_params['formality']}
                    - Lexical complexity: {batch_params['lexicalcomplexity']}
                    - Cultural sensitivity: {batch_params['culturalsensitivity']}
                    
                    Respond with a JSON object containing an array of reviews."""

                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "messages": messages,
                        "response_format": {"type": "json_object"},
                        "temperature": batch_params.get("temperature", 0.7),
                        "max_tokens": batch_params.get("maxTokens", 4000),
                        "top_p": batch_params.get("topP", 1.0),
                        "frequency_penalty": batch_params.get("frequencyPenalty", 0.0),
                        "presence_penalty": batch_params.get("presencePenalty", 0.0)
                    }
                ) as response:
                    self.rate_limiter.update_limits(response.headers)
                    
                    if response.status == 429:
                        logger.warning("Rate limit hit, retrying after delay...")
                        retries -= 1
                        await asyncio.sleep(2)
                        continue
                        
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Unexpected response status: {response.status}, response: {error_text}")
                        retries -= 1
                        await asyncio.sleep(1)
                        continue
                        
                    response_data = await response.json()
                    logger.debug(f"API response data: {response_data}")
                    
                    content = response_data["choices"][0]["message"]["content"]
                    parsed_data = json.loads(content)
                    
                    if "reviews" not in parsed_data:
                        raise ValueError("Response missing 'reviews' array")
                    
                    return parsed_data["reviews"]
                    
            except json.JSONDecodeError as e:
                logger.error(
                    f"JSON decode error in batch processing:\n"
                    f"Error: {str(e)}\n"
                    f"Batch params: {batch_params}\n"
                    f"Retries left: {retries}",
                    exc_info=True
                )
                retries -= 1
                await asyncio.sleep(1)
            except aiohttp.ClientError as e:
                logger.error(
                    f"HTTP client error in batch processing:\n"
                    f"Error type: {type(e).__name__}\n"
                    f"Error: {str(e)}\n"
                    f"Batch params: {batch_params}\n"
                    f"Retries left: {retries}",
                    exc_info=True
                )
                retries -= 1
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(
                    f"Unexpected error in batch processing:\n"
                    f"Error type: {type(e).__name__}\n"
                    f"Error: {str(e)}\n"
                    f"Batch params: {batch_params}\n"
                    f"Retries left: {retries}",
                    exc_info=True
                )
                # Log the full context of the error
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                retries -= 1
                await asyncio.sleep(1)
                
        logger.error(f"All retries exhausted for batch with parameters: {batch_params}")
        return None

    async def batch_generate(
        self,
        domain: str,
        distribution: Dict[str, int],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            all_reviews = []
            tasks = []
            
            # Log the initial parameters
            logger.info(f"Starting batch generation with distribution: {distribution}")
            logger.info(f"Domain: {domain}")
            logger.info(f"Config: {config}")
            
            async with aiohttp.ClientSession() as session:
                for sentiment, count in distribution.items():
                    if count <= 0:
                        continue
                        
                    batch_size = min(5, count)
                    num_batches = (count + batch_size - 1) // batch_size
                    
                    logger.info(f"Creating {num_batches} batches of size {batch_size} for {sentiment} sentiment")
                    
                    for i in range(num_batches):
                        current_batch_size = min(batch_size, count - i * batch_size)
                        
                        batch_params = {
                            "count": current_batch_size,
                            "sentiment": sentiment,
                            "domain": domain,
                            **{k.lower(): v for k, v in config.items()}
                        }
                        
                        tasks.append(self.process_batch(session, batch_params))
                
                # Process all batches concurrently with rate limiting and better error handling
                for batch_results in asyncio.as_completed(tasks):
                    try:
                        reviews = await batch_results
                        if reviews:
                            all_reviews.extend(reviews)
                            logger.info(f"Successfully processed batch. Total reviews so far: {len(all_reviews)}")
                        else:
                            logger.warning("Batch processing returned None")
                    except Exception as e:
                        logger.error(
                            f"Error processing batch result:\n"
                            f"Error type: {type(e).__name__}\n"
                            f"Error: {str(e)}",
                            exc_info=True
                        )

            # Add IDs and calculate distribution
            for i, review in enumerate(all_reviews):
                review["id"] = i + 1
            
            final_distribution = {
                "positive": sum(1 for r in all_reviews if r["sentiment"] == "positive"),
                "negative": sum(1 for r in all_reviews if r["sentiment"] == "negative"),
                "neutral": sum(1 for r in all_reviews if r["sentiment"] == "neutral")
            }
            
            return {
                "data": all_reviews,
                "summary": {
                    "total": len(all_reviews),
                    "distribution": final_distribution
                }
            }
            
        except Exception as e:
            logger.error(
                f"Critical error in batch_generate:\n"
                f"Error type: {type(e).__name__}\n"
                f"Error: {str(e)}\n"
                f"Distribution: {distribution}\n"
                f"Domain: {domain}\n"
                f"Config: {config}",
                exc_info=True
            )
            raise HTTPException(
                status_code=500,
                detail=f"Error in batch generation: {str(e)}"
            )