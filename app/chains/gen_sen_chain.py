from fastapi import HTTPException
import aiohttp
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass
from math import ceil
from app.promptConfig.domain_configs import DOMAIN_CONFIGS
from app.classes.models import ReviewParameters
from app.promptConfig.domain_prompt import create_review_prompt_detailed
from app.classes.concurrencyManager import ConcurrencyManager
import tiktoken



logger = logging.getLogger(__name__)
### Only so fast because of rate limit, but we aren't pushing it so hard so perhaps T2,T1 would be okay performance wise as well.
max_tokens_per_request = 8000

@dataclass
class TokenEstimates:
    base_prompt_tokens: int = 260  # Actual measured #With old prompt 140
    tokens_per_review: int = 50    # ~33 tokens + small safety margin
    json_overhead: int = 15        # JSON structure overhead

    def get_total_tokens_for_batch(self, batch_size: int) -> int:
        """Calculate total tokens needed for a batch, including safety margin"""
        return (
            self.base_prompt_tokens +  # 140 tokens
            (self.tokens_per_review * batch_size) +  # 50 tokens per review
            self.json_overhead +  # 15 tokens
            25  # Smaller safety margin
        )

@dataclass
class RateLimits:
    requests_per_minute: int
    tokens_per_minute: int
    tokens_per_hour: int
    min_quality_threshold: int = 25  # Capped at 25 reviews per batch

class RateLimiter:
    def __init__(self, rate_limits: RateLimits):
        self.requests_limit = rate_limits.requests_per_minute
        self.tokens_limit = rate_limits.tokens_per_minute
        self.remaining_requests = rate_limits.requests_per_minute
        self.remaining_tokens = rate_limits.tokens_per_minute
        self.last_update = time.time()

    def update_limits(self, headers: Dict[str, str]):
        self.remaining_requests = int(headers.get('x-ratelimit-remaining-requests', self.remaining_requests))
        self.remaining_tokens = int(headers.get('x-ratelimit-remaining-tokens', self.remaining_tokens))
        self.last_update = time.time()

    async def wait_if_needed(self):
        if self.remaining_requests < 100:  # Higher threshold
            await asyncio.sleep(1)  # Longer sleep, less frequent
        return self.remaining_requests > 0 and self.remaining_tokens > 0

class BatchOptimizer:
    def __init__(
        self,
        total_reviews: int,
        rate_limits: RateLimits,
        token_estimates: TokenEstimates = TokenEstimates()
    ):
        self.total_reviews = total_reviews
        self.rate_limits = rate_limits
        self.token_estimates = token_estimates

    def calculate_optimal_batch_size(self) -> dict:
        def get_tokens_per_batch(batch_size: int) -> int:
            return self.token_estimates.get_total_tokens_for_batch(batch_size)

        is_small_job = self.total_reviews <= 10000
        
        # For small jobs, we can be more aggressive with concurrency
          # Increased but won't hit this due to review cap
        theoretical_max_reviews = min(
            25,  # Hard cap at 25 reviews
            (max_tokens_per_request - self.token_estimates.base_prompt_tokens - self.token_estimates.json_overhead) // self.token_estimates.tokens_per_review
        )
        
        # Calculate theoretical max based on rate limits
        theoretical_max_by_tokens = (self.rate_limits.tokens_per_minute // 60) // self.token_estimates.get_total_tokens_for_batch(1)
        
        max_batch_size = min(
            50,  # Double the batch size
            theoretical_max_reviews,
            theoretical_max_by_tokens
        )

        # Test batch sizes from 15 to max_batch_size for small jobs
        # This ensures we don't waste time testing very small batch sizes
        min_batch_size = 15 if is_small_job else 5
        optimal_config = None
        max_throughput = 0

        for batch_size in range(min_batch_size, max_batch_size + 1):
            tokens_per_batch = get_tokens_per_batch(batch_size)
            
            # Calculate maximum concurrent requests possible
            max_concurrent_by_rpm = self.rate_limits.requests_per_minute // 60
            max_concurrent_by_tpm = (self.rate_limits.tokens_per_minute // 60) // tokens_per_batch
            max_concurrent_by_tph = (self.rate_limits.tokens_per_hour // 3600) // tokens_per_batch
            
            # Much more aggressive concurrency for small jobs
            max_concurrent_cap = 150 if is_small_job else 50  # Increased from 75 to 150
            safety_factor = 0.95 if is_small_job else 0.8    # Increased from 0.9 to 0.95
            
            concurrent_tasks = min(
                max_concurrent_by_rpm,
                max_concurrent_by_tpm,
                max_concurrent_by_tph,
                max_concurrent_cap
            )
            concurrent_tasks = max(1, int(concurrent_tasks * safety_factor))
            
            # Calculate throughput (reviews per second)
            throughput = concurrent_tasks * batch_size
            
            # More aggressive completion time for small jobs
            completion_time = self.total_reviews / (throughput * 60)
            max_completion_time = 15 if is_small_job else 120  # 15 mins for small jobs
            
            if (throughput > max_throughput and 
                completion_time < max_completion_time and 
                tokens_per_batch < max_tokens_per_request):
                max_throughput = throughput
                optimal_config = {
                    'batch_size': batch_size,
                    'concurrent_tasks': concurrent_tasks,
                    'tokens_per_batch': tokens_per_batch,
                    'reviews_per_second': throughput,
                    'estimated_completion_time': completion_time,
                    'total_batches': ceil(self.total_reviews / batch_size),
                    'estimated_total_tokens': (
                        ceil(self.total_reviews / batch_size) * tokens_per_batch
                    )
                }

        logger.info(f"Calculated optimal configuration: {optimal_config}")
        return optimal_config

    # ## TOO MUCH UNNECESSARY INFORMATION FOR PROMPT
    # def get_prompt_description(self) -> str:
    #     return f"""Review Generation Parameters:
    #     Content Control:
    #     - Privacy: {self.privacy_level} (higher means more anonymized content)
    #     - Cultural Sensitivity: {self.cultural_sensitivity} (higher means more inclusive language)
    #     - Bias Control: {self.bias_control} (higher means more neutral perspective)
        
    #     Style Control:
    #     - Authenticity: {self.realism} (higher means more realistic language)
    #     - Formality: {self.formality} (higher means more professional tone)
    #     - Language Complexity: {self.lexical_complexity} (higher means more sophisticated vocabulary)
    #     - Emotional Intensity: {self.sentiment_intensity} (higher means stronger sentiment expression)
    #     - Contemporary Relevance: {self.temporal_relevance} (higher means more current context)
    #     - Detail Level: {self.noise_level} (higher means more supplementary information)"""



class GenSenChain:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.domain_configs = DOMAIN_CONFIGS
        rate_limits = RateLimits(
            requests_per_minute=5000,
            tokens_per_minute=4_000_000,
            tokens_per_hour=100_000_000
        )
        self.rate_limiter = RateLimiter(rate_limits)
        self.token_estimates = TokenEstimates()
        self.concurrency_manager = ConcurrencyManager(
            initial=200,
            min_limit=100,
            max_limit=300,
            increment=10,
            decrement=20
        )

    async def get_token_count(self, messages: List[Dict[str, str]]) -> int:
        """Estimate token count before making API call"""
        try:
            encoding = tiktoken.encoding_for_model("gpt-4o-mini")
            total_tokens = 0
            for message in messages:
                total_tokens += len(encoding.encode(message["role"]))
                total_tokens += len(encoding.encode(message["content"]))
            return total_tokens
        except Exception as e:
            logger.warning(f"Failed to count tokens: {e}")
            return 0 

    async def process_batch(
        self,
        session: aiohttp.ClientSession,
        batch_params: Dict[str, Any],
        retries: int = 3
    ) -> Optional[List[Dict[str, Any]]]:
        async with self.concurrency_manager:
            try:
                await self.rate_limiter.wait_if_needed()
                
                parameters = ReviewParameters(
                    privacyLevel=batch_params.get('privacyLevel', 0.8),
                    culturalSensitivity=batch_params.get('culturalSensitivity', 0.8),
                    biasControl=batch_params.get('biasControl', 0.7),
                    realism=batch_params.get('realism', 0.7),
                    domainRelevance=batch_params.get('domainRelevance', 0.8),
                    formality=batch_params.get('formality', 0.5),
                    lexicalComplexity=batch_params.get('lexicalComplexity', 0.5),
                    sentimentIntensity=batch_params.get('sentimentIntensity', 0.5),
                    temporalRelevance=batch_params.get('temporalRelevance', 0.5),
                    noiseLevel=batch_params.get('noiseLevel', 0.3),
                    diversity=batch_params.get('diversity', 0.6)
                )

                # Create user prompt BEFORE creating messages
                user_prompt = create_review_prompt_detailed(
                    domain=batch_params['domain'],
                    parameters=parameters,
                    batch_params=batch_params,
                    use_compact_prompt=True
                )

                
                dynamic_system_prompt = f"""Generate authentic {batch_params['domain']} reviews with {batch_params['sentiment']} sentiment.
Output: {{"reviews":[{{"text":"content","sentiment":"{batch_params['sentiment']}"}}]}}"""

                messages = [
                    {"role": "system", "content": dynamic_system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                print(messages)
                token_count = await self.get_token_count(messages)
                logger.info(f"Estimated token count for request: {token_count}")

                await self.rate_limiter.wait_if_needed()
                
                total_requested = batch_params['count']
                collected_reviews = []
                
                if batch_params['domain'].lower() not in self.domain_configs:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid domain. Please choose from: {', '.join(self.domain_configs.keys())}"
                    )
                
                while retries > 0 and total_requested > 0:
                    try:
                        await self.rate_limiter.wait_if_needed()
                        
                        async with session.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": batch_params.get("model", "gpt-4o-mini"),
                                "messages": messages,
                                "response_format": {"type": "json_object"},
                                "temperature": batch_params.get("temperature", 0.7),
                                "max_tokens": batch_params.get(max_tokens_per_request),
                                "top_p": batch_params.get("topP", 1.0),
                                "frequency_penalty": batch_params.get("frequencyPenalty", 0.0),
                                "presence_penalty": batch_params.get("presencePenalty", 0.)
                            }
                        ) as response:
                            self.rate_limiter.update_limits(response.headers)
                            
                            if response.status == 429:
                                logger.warning("Rate limit hit, retrying after delay...")
                                retries -= 1
                                await asyncio.sleep(1)
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
                            
                            reviews = parsed_data["reviews"]
                            for review in reviews:
                                if "sentiment" not in review:
                                    review["sentiment"] = batch_params["sentiment"]
                                if "text" not in review:
                                    logger.warning(f"Review missing text field, skipping: {review}")
                                    continue
                            
                            collected_reviews.extend(reviews)
                            
                            # Calculate remaining reviews
                            total_requested -= len(reviews)
                            
                            if total_requested <= 0:
                                return collected_reviews
                            
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
                return collected_reviews if collected_reviews else None
            except aiohttp.ClientError as e:
                if "429" in str(e):
                    logger.warning("Rate limit hit, backing off...")
                raise  # The context manager will handle the decrease_limit

    async def batch_generate(
        self,
        domain: str,
        distribution: Dict[str, int],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            # Add domain validation
            valid_domains = ['ecommerce', 'social_media', 'software', 'restaurant', 'hotel', 'technology', 'education','healthcare']  
            if domain.lower() not in valid_domains:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid domain. Please choose from: {', '.join(valid_domains)}"
                )

            total_reviews = sum(distribution.values())
            rate_limits = RateLimits(
                requests_per_minute=5000,
                tokens_per_minute=4_000_000,
                tokens_per_hour=100_000_000
            )
            
            optimizer = BatchOptimizer(total_reviews, rate_limits)
            optimal_config = optimizer.calculate_optimal_batch_size()
            
            batch_size = optimal_config['batch_size']
            all_reviews = []

            # Session pool with automatic cleanup
            connector = aiohttp.TCPConnector(
                limit=100,  # Max concurrent connections
                ttl_dns_cache=300,  # DNS cache TTL in seconds
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(
                total=300,  # Total timeout
                connect=10,  # Connection timeout
                sock_read=60  # Socket read timeout
            )
            
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                raise_for_status=True
            ) as session:
                tasks = []
                for sentiment, count in distribution.items():
                    if count <= 0:
                        continue
                    
                    num_batches = (count + batch_size - 1) // batch_size
                    for i in range(num_batches):
                        current_batch_size = min(batch_size, count - i * batch_size)
                        batch_params = {
                            "count": current_batch_size,
                            "sentiment": sentiment,
                            "domain": domain,
                            **{k.lower(): v for k, v in config.items()}
                        }
                        tasks.append(self.process_batch(session, batch_params))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results and handle any exceptions
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Batch processing error: {result}")
                        continue
                    if result:
                        all_reviews.extend(result)

            # Add IDs and calculate distribution outside the session
            for i, review in enumerate(all_reviews):
                review["id"] = i + 1
            
            final_distribution = {
                "positive": sum(1 for r in all_reviews if r.get("sentiment") == "positive"),
                "negative": sum(1 for r in all_reviews if r.get("sentiment") == "negative"),
                "neutral": sum(1 for r in all_reviews if r.get("sentiment") == "neutral")
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
