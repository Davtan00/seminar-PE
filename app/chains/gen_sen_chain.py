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
        # GPT-4o-mini limits
        self.requests_per_minute = 5000  # 5000 RPM
        self.tokens_per_minute = 4_000_000  # 4M TPM
        self.min_quality_threshold = 80

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
        self.rate_limiter = RateLimiter(RateLimits())
        self.token_estimates = TokenEstimates()
        
        # Aggressive concurrency settings for high throughput
        self.concurrency_manager = ConcurrencyManager(
            initial=200,     # Start aggressive
            min_limit=100,   # Still maintain reasonable minimum
            max_limit=400,   # Can go very high with 5000 RPM
            increment=10,    # Aggressive scaling up
            decrement=20     # Quick backoff if needed
        )
        
        # Shorter timeouts since we expect fast responses
        self.base_timeout = aiohttp.ClientTimeout(
            total=30,      # Reduced from 60
            connect=5,     # Reduced from 10
            sock_read=15   # Reduced from 30
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
        retries: int = 3  # Increased from 2
    ) -> Tuple[List[Dict[str, Any]], float]:
        start_time = time.perf_counter()
        last_error = None
        
        for attempt in range(retries):
            try:
                async with self.concurrency_manager:
                    await self.rate_limiter.wait_if_needed()
                    
                    async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "gpt-4o-mini",
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
                            "temperature": 0.7,
                            "max_tokens": 4000
                        },
                        timeout=self.base_timeout
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            reviews = json.loads(data["choices"][0]["message"]["content"])["reviews"]
                            self.concurrency_manager.increase_limit()
                            return reviews, time.perf_counter() - start_time
                        
                        if response.status == 429:
                            self.concurrency_manager.decrease_limit()
                            await asyncio.sleep(1 * (attempt + 1))
                            continue
                            
                        if response.status >= 500:
                            await asyncio.sleep(2 * (attempt + 1))
                            continue
                        
                        response_text = await response.text()
                        last_error = f"API error: {response.status} - {response_text}"
                        raise aiohttp.ClientError(last_error)
                        
            except Exception as e:
                last_error = str(e)
                if attempt == retries - 1:
                    logger.error(f"Batch processing error after {retries} attempts: {last_error}")
                    return [], time.perf_counter() - start_time
                await asyncio.sleep(min(2 ** attempt, 8))

    async def batch_generate(
        self,
        domain: str,
        distribution: Dict[str, int],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            # Aggressive batching settings
            BATCH_SIZE = 20  # Increased from 10
            MAX_CONCURRENT_TOTAL = 200  # Increased from 50
            
            connector = aiohttp.TCPConnector(
                limit=400,           # Increased from 100
                limit_per_host=200,  # Increased from 50
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
                force_close=True
            )
            
            all_reviews: List[Dict[str, Any]] = []
            tasks_to_cancel = set()
            
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=60, connect=10, sock_read=30)
            ) as session:
                pending_tasks = []
                remaining_distribution = distribution.copy()
                stats = {sentiment: BatchStats() for sentiment in distribution.keys()}
                total_reviews_needed = sum(distribution.values())
                
                while sum(remaining_distribution.values()) > 0:
                    # Create new tasks if needed
                    while len(pending_tasks) < MAX_CONCURRENT_TOTAL:
                        for sentiment, remaining in remaining_distribution.items():
                            if remaining <= 0:
                                continue
                                
                            current_batch_size = min(BATCH_SIZE, remaining)
                            batch_params = {
                                "count": current_batch_size,
                                "sentiment": sentiment,
                                "domain": domain,
                                **config
                            }
                            task = asyncio.create_task(self.process_batch(session, batch_params))
                            tasks_to_cancel.add(task)
                            pending_tasks.append((sentiment, task))
                            
                            if len(pending_tasks) >= MAX_CONCURRENT_TOTAL:
                                break
                    
                    if not pending_tasks:
                        break
                    
                    # Wait for any task to complete
                    done, _ = await asyncio.wait(
                        [task for _, task in pending_tasks],
                        timeout=30,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    if not done:
                        logger.warning("Task timeout occurred, restarting failed tasks...")
                        # Cancel timed out tasks
                        for _, task in pending_tasks:
                            if not task.done():
                                task.cancel()
                        pending_tasks = []
                        continue
                    
                    # Process completed tasks
                    new_pending = []
                    for sentiment, task in pending_tasks:
                        if task in done:
                            try:
                                if task.cancelled():
                                    continue
                                    
                                reviews, processing_time = await task
                                if reviews:
                                    remaining_needed = remaining_distribution[sentiment]
                                    reviews_to_add = reviews[:remaining_needed]
                                    
                                    remaining_distribution[sentiment] -= len(reviews_to_add)
                                    all_reviews.extend(reviews_to_add)
                                    
                                    stats[sentiment].update(processing_time, len(reviews), len(reviews_to_add))
                            except Exception as e:
                                logger.error(f"Error processing batch: {str(e)}")
                                # Don't retry immediately, let the next loop handle it
                            finally:
                                tasks_to_cancel.discard(task)
                        else:
                            new_pending.append((sentiment, task))
                    
                    pending_tasks = new_pending
                    
                    # Log progress
                    if len(all_reviews) % 100 == 0:  # More frequent updates
                        logger.info(f"\nGeneration progress ({len(all_reviews)}/{total_reviews_needed} reviews):")
                        for sent, stat in stats.items():
                            if stat.batches_completed > 0:
                                logger.info(
                                    f"{sent}:"
                                    f"\n - Generated: {stat.reviews_generated}"
                                    f"\n - Remaining: {remaining_distribution[sent]}"
                                    f"\n - Avg time/batch: {stat.avg_time_per_batch:.2f}s"
                                )
            
        finally:
            # Cleanup any remaining tasks
            for task in tasks_to_cancel:
                if not task.done():
                    task.cancel()
            
            # Wait for cancellations
            if tasks_to_cancel:
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        
        # Add IDs and return results
        for i, review in enumerate(all_reviews):
            review["id"] = i + 1
        
        final_distribution = {
            "positive": sum(1 for r in all_reviews if r.get("sentiment") == "positive"),
            "negative": sum(1 for r in all_reviews if r.get("sentiment") == "negative"),
            "neutral": sum(1 for r in all_reviews if r.get("sentiment") == "neutral")
        }
        
        # Verify all requested reviews were generated
        for sentiment, target in distribution.items():
            actual = final_distribution[sentiment]
            if actual != target:
                logger.warning(
                    f"Distribution mismatch for {sentiment}: "
                    f"target={target}, actual={actual}"
                )
                raise ValueError(f"Failed to generate all requested reviews. Missing {target - actual} {sentiment} reviews.")
        
        return {
            "data": all_reviews,
            "summary": {
                "total": len(all_reviews),
                "distribution": final_distribution
            }
        }

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
