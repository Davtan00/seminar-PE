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
    base_prompt_tokens: int = 80
    tokens_per_review: int = 50
    json_overhead: int = 10

    def get_total_tokens_for_batch(self, batch_size: int) -> int:
        return (
            self.base_prompt_tokens +
            (self.tokens_per_review * batch_size) +
            self.json_overhead +
            25  # Safety margin
        )

@dataclass
class RateLimits:
    def __init__(self):
        self.requests_per_minute = 5000
        self.tokens_per_minute = 4_000_000
        self.tokens_per_hour = 100_000_000
        self.min_quality_threshold = 100  # Increased significantly

class RateLimiter:
    def __init__(self, rate_limits: RateLimits):
        self.requests_limit = rate_limits.requests_per_minute
        self.tokens_limit = rate_limits.tokens_per_minute
        self.remaining_requests = rate_limits.requests_per_minute
        self.remaining_tokens = rate_limits.tokens_per_minute
        self.last_update = time.time()
        self.request_times = []  # Track request timestamps
        self.token_usage = []    # Track token usage

    def update_limits(self, headers: Dict[str, str]):
        self.remaining_requests = int(headers.get('x-ratelimit-remaining-requests', self.remaining_requests))
        self.remaining_tokens = int(headers.get('x-ratelimit-remaining-tokens', self.remaining_tokens))
        self.last_update = time.time()

    async def wait_if_needed(self):
        current_time = time.time()
        # Clean up old entries
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        self.token_usage = [t for t in self.token_usage if current_time - t[1] < 60]
        
        requests_last_minute = len(self.request_times)
        tokens_last_minute = sum(tokens for tokens, _ in self.token_usage)
        
        if requests_last_minute >= self.requests_limit * 0.95:  # 95% of limit
            await asyncio.sleep(0.5)
        elif tokens_last_minute >= self.tokens_limit * 0.95:
            await asyncio.sleep(0.5)
        
        return True

    def record_usage(self, tokens_used: int):
        current_time = time.time()
        self.request_times.append(current_time)
        self.token_usage.append((tokens_used, current_time))

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

        is_small_job = self.total_reviews <= 50000  # Increased threshold
        
        theoretical_max_reviews = min(
            100,  # Increased from 50 to 100 reviews per batch
            (max_tokens_per_request - self.token_estimates.base_prompt_tokens - self.token_estimates.json_overhead) // self.token_estimates.tokens_per_review
        )
        
        # More aggressive batch sizes
        min_batch_size = 50 if is_small_job else 25  # Increased minimum batch sizes
        max_concurrent_cap = 400 if is_small_job else 200  # Increased concurrency
        safety_factor = 0.98 if is_small_job else 0.95  # More aggressive safety factors
        
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
        rate_limits = RateLimits()
        self.rate_limiter = RateLimiter(rate_limits)
        self.token_estimates = TokenEstimates()
        self.concurrency_manager = ConcurrencyManager(
            initial=400,    # Increased for better throughput
            min_limit=250,  
            max_limit=600,  # Higher max limit
            increment=30,   
            decrement=50    
        )
        self.request_semaphore = asyncio.Semaphore(500)
        self.base_timeout = aiohttp.ClientTimeout(
            total=300,    # Total timeout
            connect=30,   # Connection timeout
            sock_read=90  # Socket read timeout
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
        async with self.request_semaphore:
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
Return a JSON object with the following schema:
{{
  "reviews": [
    {{
      "text": string,  // The review content
      "sentiment": "{batch_params['sentiment']}"  // The sentiment of the review
    }}
  ]
}}"""

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
                
                retry_count = 0
                while retry_count < retries and total_requested > 0:
                    try:
                        # Exponential backoff for retries
                        if retry_count > 0:
                            wait_time = min(32, (2 ** retry_count))
                            logger.info(f"Retry {retry_count}/{retries}, waiting {wait_time} seconds")
                            await asyncio.sleep(wait_time)

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
                                "max_tokens": max_tokens_per_request,
                                "top_p": batch_params.get("topp", 1.0),
                                "frequency_penalty": batch_params.get("frequencypenalty", 0),
                                "presence_penalty": batch_params.get("presencepenalty", 0)
                            },
                            timeout=self.base_timeout
                        ) as response:
                            if response.status == 429:  # Rate limit
                                retry_after = int(response.headers.get('Retry-After', 1))
                                logger.warning(f"Rate limit hit, waiting {retry_after} seconds")
                                await asyncio.sleep(retry_after)
                                retry_count += 1
                                continue

                            if response.status != 200:
                                error_text = await response.text()
                                logger.error(
                                    f"OpenAI API error:\n"
                                    f"Status: {response.status}\n"
                                    f"Response: {error_text}"
                                )
                                raise aiohttp.ClientError(f"API error: {error_text}")

                            response_data = await response.json()
                            logger.debug(f"API response data: {response_data}")
                            
                            try:
                                content = response_data["choices"][0]["message"]["content"]
                                parsed_data = json.loads(content)
                                
                                if "reviews" not in parsed_data:
                                    logger.error(f"Invalid JSON structure received: {content}")
                                    raise ValueError("Response missing 'reviews' array")
                                
                                reviews = parsed_data["reviews"]
                                processed_reviews = []
                                
                                for review in reviews:
                                    if not isinstance(review, dict):
                                        logger.warning(f"Skipping invalid review format: {review}")
                                        continue
                                        
                                    if "text" not in review:
                                        logger.warning(f"Review missing text field: {review}")
                                        continue
                                        
                                    # Ensure review is a proper dictionary with all required fields
                                    processed_review = {
                                        "text": str(review["text"]),
                                        "sentiment": review.get("sentiment", batch_params["sentiment"]),
                                    }
                                    processed_reviews.append(processed_review)
                                
                                collected_reviews.extend(processed_reviews)
                                
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON decode error: {str(e)}\nContent: {content}")
                                raise
                            
                            # Calculate remaining reviews
                            total_requested -= len(reviews)
                            
                            if total_requested <= 0:
                                return collected_reviews
                            
                    except (aiohttp.ClientError, json.JSONDecodeError) as e:
                        logger.error(
                            f"Error in batch processing:\n"
                            f"Error type: {type(e).__name__}\n"
                            f"Error: {str(e)}\n"
                            f"Batch params: {batch_params}\n"
                            f"Retries left: {retries - retry_count}"
                        )
                        retry_count += 1
                        if retry_count >= retries:
                            if collected_reviews:  # Return partial results if we have any
                                return collected_reviews
                            raise

                    except Exception as e:
                        logger.error(f"Unexpected error in batch processing: {str(e)}", exc_info=True)
                        raise

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
            rate_limits = RateLimits()
            
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
