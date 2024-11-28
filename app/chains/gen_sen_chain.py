from fastapi import HTTPException
import aiohttp
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional,Tuple
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
max_tokens_per_request = 12000

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
        self._last_check = time.time()
        self._window_size = 0.5  # Reduced from 1.0 second

    async def wait_if_needed(self):
        current_time = time.time()
        if current_time - self._last_check < 0.05:  # Reduced sleep threshold
            return True
        self._last_check = current_time
        return True

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

        is_small_job = self.total_reviews <= 50000
        
        # Calculate theoretical maximums
        theoretical_max_reviews = min(
            25,  # Reduced from 100 to 15
            (max_tokens_per_request - self.token_estimates.base_prompt_tokens - self.token_estimates.json_overhead) // self.token_estimates.tokens_per_review
        )
        
        theoretical_max_by_tokens = (max_tokens_per_request - self.token_estimates.base_prompt_tokens - self.token_estimates.json_overhead) // self.token_estimates.tokens_per_review
        
        min_batch_size = 10 if is_small_job else 5  # Reduced from 50/25
        max_concurrent_cap = 800 if is_small_job else 400  # Increased from 400/200
        safety_factor = 0.98 if is_small_job else 0.95
        
        max_batch_size = min(
            25,  
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
        
        # Keep token estimates but make optional
        self.token_estimates = TokenEstimates()
        
        # Simplified concurrency settings
        self.request_semaphore = asyncio.Semaphore(1000)  # Increased from 500
        self.base_timeout = aiohttp.ClientTimeout(
            total=300,
            connect=10,   # Reduced from 30
            sock_read=30  # Reduced from 90
        )
        
        # Simplified system prompt
        self.system_prompt = """Generate authentic reviews with specified sentiment.
Return a JSON object with the following schema:
{
  "reviews": [
    {
      "text": string,
      "sentiment": string
    }
  ]
}"""

    # Keep token counting but make it optional
    # async def get_token_count(self, messages: List[Dict[str, str]]) -> int:
    #     try:
    #         encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    #         total_tokens = 0
    #         for message in messages:
    #             total_tokens += len(encoding.encode(message["role"]))
    #             total_tokens += len(encoding.encode(message["content"]))
    #         return total_tokens
    #     except Exception as e:
    #         logger.warning(f"Failed to count tokens: {e}")
    #         return 0

    async def process_batch(
        self,
        session: aiohttp.ClientSession,
        batch_params: Dict[str, Any],
        retries: int = 3
    ) -> Tuple[List[Dict[str, Any]], float]:
        start_time = time.perf_counter()
        
        # Precompute the request payload
        payload = {
            "model": batch_params.get("model", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": create_review_prompt_detailed(
                    domain=batch_params['domain'],
                    parameters=ReviewParameters(**{
                        k: batch_params.get(k, 0.7) 
                        for k in ReviewParameters.__annotations__
                    }),
                    batch_params=batch_params,
                    use_compact_prompt=True
                )}
            ],
            "response_format": {"type": "json_object"},
            "temperature": batch_params.get("temperature", 0.7),
            "max_tokens": 4000,
            "top_p": batch_params.get("topP", 1.0),
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        for attempt in range(retries):
            try:
                async with self.request_semaphore:
                    await self.rate_limiter.wait_if_needed()
                    
                    async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=self.base_timeout
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            reviews = json.loads(data["choices"][0]["message"]["content"])["reviews"]
                            return ([{
                                "text": str(review["text"]),
                                "sentiment": review.get("sentiment", batch_params["sentiment"])
                            } for review in reviews if isinstance(review, dict) and "text" in review],
                            time.perf_counter() - start_time)
                        
                        if response.status == 429:  # Rate limit
                            await asyncio.sleep(0.5 * (attempt + 1))  # Reduced backoff
                            continue
                            
                        if response.status >= 500:  # Server error
                            await asyncio.sleep(0.5 * (attempt + 1))  # Reduced backoff
                            continue
                        
                        raise aiohttp.ClientError(f"API error: {await response.text()}")
                        
            except Exception as e:
                if attempt == retries - 1:
                    logger.error(f"Batch processing error after {retries} attempts: {str(e)}")
                    return [], time.perf_counter() - start_time
                await asyncio.sleep(min(2 ** attempt, 8))

    async def batch_generate(
        self,
        domain: str,
        distribution: Dict[str, int],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            # Keep domain validation for safety
            valid_domains = ['ecommerce', 'social_media', 'software', 'restaurant', 'hotel', 'technology', 'education', 'healthcare']
            if domain.lower() not in valid_domains:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid domain. Please choose from: {', '.join(valid_domains)}"
                )

            # Simplified batch size configuration
            BATCH_SIZE = 8  # Slightly larger than 5, but not too large
            MAX_CONCURRENT_TOTAL = 250  # Reduced from 400 but still aggressive
            all_reviews: List[Dict[str, Any]] = []
            
            # Optimized session configuration
            connector = aiohttp.TCPConnector(
                limit=300,          # Reduced from 400 but still high
                limit_per_host=100, # Reduced from 200
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
                force_close=True
            )
            
            timeout = aiohttp.ClientTimeout(
                total=300,
                connect=15,    # Slightly increased from 10
                sock_read=45   # Slightly increased from 30
            )
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                # Track pending tasks
                pending_tasks = []
                sentiment_counts = {sentiment: 0 for sentiment in distribution.keys()}
                remaining_distribution = distribution.copy()
                # Initialize stats dictionary for each sentiment
                stats = {sentiment: BatchStats() for sentiment in distribution.keys()}
                
                # Create initial tasks
                for sentiment, count in distribution.items():
                    batches_needed = (count + BATCH_SIZE - 1) // BATCH_SIZE
                    concurrent_limit = min(batches_needed, MAX_CONCURRENT_TOTAL // len(distribution) * 2)
                    
                    for i in range(concurrent_limit):
                        remaining = count - (i * BATCH_SIZE)
                        if remaining <= 0:
                            break
                        
                        current_batch_size = min(BATCH_SIZE, remaining)
                        batch_params = {
                            "count": current_batch_size,
                            "sentiment": sentiment,
                            "domain": domain,
                            **config
                        }
                        task = asyncio.create_task(self.process_batch(session, batch_params))
                        pending_tasks.append((sentiment, task))
                
                # Process tasks until all reviews are generated
                while pending_tasks:
                    done, _ = await asyncio.wait(
                        [task for _, task in pending_tasks],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    for completed_task in done:
                        task_index = next(i for i, (_, task) in enumerate(pending_tasks) if task == completed_task)
                        sentiment, _ = pending_tasks.pop(task_index)
                        
                        try:
                            reviews, processing_time = await completed_task
                            if reviews:
                                current_count = sentiment_counts[sentiment]
                                target_count = distribution[sentiment]
                                remaining_needed = max(0, target_count - current_count)
                                
                                reviews_to_add = reviews[:remaining_needed]
                                sentiment_counts[sentiment] += len(reviews_to_add)
                                remaining_distribution[sentiment] -= len(reviews_to_add)
                                all_reviews.extend(reviews_to_add)
                                
                                # Update stats for this sentiment
                                stats[sentiment].update(processing_time, len(reviews), len(reviews_to_add))
                        except Exception as e:
                            logger.error(f"Error processing batch: {str(e)}")
                            # Recalculate exactly how many reviews we still need
                            current_count = sentiment_counts[sentiment]
                            target_count = distribution[sentiment]
                            remaining_needed = max(0, target_count - current_count)
                            
                            if remaining_needed > 0:
                                # Create a replacement task with exactly the number we need
                                new_batch_size = min(BATCH_SIZE, remaining_needed)
                                batch_params = {
                                    "count": new_batch_size,
                                    "sentiment": sentiment,
                                    "domain": domain,
                                    **config
                                }
                                new_task = asyncio.create_task(self.process_batch(session, batch_params))
                                pending_tasks.append((sentiment, new_task))
                        
                        # Create new tasks only if needed (moved outside try/except)
                        if remaining_distribution[sentiment] > 0:
                            new_batch_size = min(BATCH_SIZE, remaining_distribution[sentiment])
                            batch_params = {
                                "count": new_batch_size,
                                "sentiment": sentiment,
                                "domain": domain,
                                **config
                            }
                            new_task = asyncio.create_task(self.process_batch(session, batch_params))
                            pending_tasks.append((sentiment, new_task))
                        
                        # Log progress
                        LOG_FREQUENCY = 500  # Increased from 250
                        if len(all_reviews) % LOG_FREQUENCY == 0:
                            logger.info(f"\nGeneration progress ({len(all_reviews)} reviews):")
                            for sent, stat in stats.items():
                                if stat.batches_completed > 0:
                                    logger.info(
                                        f"{sent}:"
                                        f"\n - Generated: {stat.reviews_generated}"
                                        f"\n - Remaining: {remaining_distribution[sent]}"
                                        f"\n - Avg time/batch: {stat.avg_time_per_batch:.2f}s"
                                    )

                # Add IDs to reviews
                for i, review in enumerate(all_reviews):
                    review["id"] = i + 1
                
                # Calculate final distribution
                final_distribution = {
                    "positive": sum(1 for r in all_reviews if r.get("sentiment") == "positive"),
                    "negative": sum(1 for r in all_reviews if r.get("sentiment") == "negative"),
                    "neutral": sum(1 for r in all_reviews if r.get("sentiment") == "neutral")
                }
                
                # Log distribution mismatches
                for sentiment, target in distribution.items():
                    actual = final_distribution[sentiment]
                    if actual != target:
                        logger.warning(
                            f"Distribution mismatch for {sentiment}: "
                            f"target={target}, actual={actual}"
                        )
                
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

@dataclass
class BatchStats:
    __slots__ = ['total_time', 'successful_reviews', 'failed_reviews', 
                 'min_time_per_review', 'max_time_per_review', 'batches_completed']
    
    def __init__(self):
        self.total_time = 0.0
        self.successful_reviews = 0
        self.failed_reviews = 0
        self.min_time_per_review = float('inf')
        self.max_time_per_review = 0.0
        self.batches_completed = 0
    
    @property
    def reviews_generated(self) -> int:
        return self.successful_reviews  # Alias for successful_reviews
    
    def update(self, processing_time: float, requested: int, received: int):
        self.total_time += processing_time
        self.successful_reviews += received
        self.failed_reviews += (requested - received)
        self.batches_completed += 1
        
        if received > 0:
            time_per_review = processing_time / received
            self.min_time_per_review = min(self.min_time_per_review, time_per_review)
            self.max_time_per_review = max(self.max_time_per_review, time_per_review)
    
    @property
    def avg_time_per_batch(self) -> float:
        return self.total_time / self.batches_completed if self.batches_completed > 0 else 0
