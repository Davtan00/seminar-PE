from fastapi import HTTPException
import aiohttp
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional,Tuple,Iterator 
import time
from dataclasses import dataclass
from math import ceil
from app.promptConfig.domain_configs import DOMAIN_CONFIGS
from app.classes.models import ReviewParameters
from app.promptConfig.domain_prompt import create_review_prompt_detailed
from app.classes.concurrencyManager import ConcurrencyManager
import tiktoken
from itertools import islice



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

        # Adjust based on job size
        if self.total_reviews > 5000:
            min_batch_size = 15
            max_concurrent_cap = 600
            safety_factor = 0.90
        elif self.total_reviews > 1000:
            min_batch_size = 10
            max_concurrent_cap = 400
            safety_factor = 0.95
        else:
            min_batch_size = 8
            max_concurrent_cap = 250
            safety_factor = 0.98

        # Calculate theoretical maximums
        theoretical_max_reviews = min(
            35,  # Increased from 25
            (max_tokens_per_request - self.token_estimates.base_prompt_tokens - self.token_estimates.json_overhead) 
            // self.token_estimates.tokens_per_review
        )
        
        max_batch_size = min(
            35,  # Increased from 25
            theoretical_max_reviews
        )

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
            
            throughput = concurrent_tasks * batch_size
            completion_time = self.total_reviews / (throughput * 60)
            
            if (throughput > max_throughput and 
                tokens_per_batch < max_tokens_per_request):
                max_throughput = throughput
                optimal_config = {
                    'batch_size': batch_size,
                    'concurrent_tasks': concurrent_tasks,
                    'tokens_per_batch': tokens_per_batch,
                    'reviews_per_second': throughput,
                    'estimated_completion_time': completion_time,
                    'total_batches': ceil(self.total_reviews / batch_size)
                }

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
        
        # Replace semaphore with ConcurrencyManager and enforces limits
        self.concurrency_manager = ConcurrencyManager(
            initial=300,          # Start aggressive
            min_limit=100,        # Don't go below this
            max_limit=400,        # Don't go above this
            success_threshold=5,  # Increase limit after 5 successes
            failure_threshold=2   # Decrease limit after 2 failures
        )
    
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
        retries: int = 4
    ) -> Tuple[List[Dict[str, Any]], float]:
        start_time = time.perf_counter()
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
            "max_tokens": 8000,
            "top_p": batch_params.get("topP", 1.0),
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        for attempt in range(retries):
            try:
                async with self.concurrency_manager:
                    timeout = aiohttp.ClientTimeout(
                        total=90,  # Increased from 60
                        connect=10 + (attempt * 3),  # Less aggressive scaling
                        sock_read=30 + (attempt * 3)  # Less aggressive scaling
                    )
                    
                    async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json=payload,
                        timeout=timeout
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
                            await asyncio.sleep(0.3 * (attempt + 1))  # Reduced backoff
                            continue
                            
                        if response.status >= 500:  # Server error
                            await asyncio.sleep(0.3 * (attempt + 1))  # Reduced backoff
                            continue
                        
                        raise aiohttp.ClientError(f"API error: {await response.text()}")
            except Exception as e:
                if attempt == retries - 1:
                    logger.error(f"Batch processing error after {retries} attempts: {str(e)}")
                    return [], time.perf_counter() - start_time
                await asyncio.sleep(min(1.5 ** attempt, 4))  # Less aggressive backoff
        return [], time.perf_counter() - start_time

    def _create_batch_tasks(
        self,
        session: aiohttp.ClientSession,
        sentiment: str,
        remaining: int,
        max_concurrent: int,
        batch_size: int,
        domain: str,
        config: Dict[str, Any]
    ) -> List[Tuple[str, asyncio.Task]]:
        tasks = []
        batches_needed = (remaining + batch_size - 1) // batch_size
        concurrent_batches = min(batches_needed, max_concurrent)
        
        for i in range(concurrent_batches):
            current_size = min(batch_size, remaining - (i * batch_size))
            if current_size <= 0:
                break
            
            batch_params = {
                "count": current_size,
                "sentiment": sentiment,
                "domain": domain,
                **config
            }
            task = asyncio.create_task(self.process_batch(session, batch_params))
            tasks.append((sentiment, task))
        
        return tasks

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
                
                # Initialize temporary storage for reviews
                temp_reviews: List[Dict[str, Any]] = []
                total_reviews = sum(distribution.values())
                chunk_size = 1000 if total_reviews > 5000 else 2500
                
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
                
                last_progress_time = time.time()
                last_review_count = 0
                
                # Process tasks until all reviews are generated
                while pending_tasks:
                    # Log active tasks status every 30 seconds
                    current_time = time.time()
                    if current_time - last_progress_time > 30:
                        current_review_count = len(all_reviews)
                        reviews_since_last = current_review_count - last_review_count
                        
                        logger.info(
                            f"Status Update [{time.strftime('%H:%M:%S')}]:\n"
                            f"• Active Tasks: {len(pending_tasks)}\n"
                            f"• Reviews in last 30s: {reviews_since_last}\n"
                            f"• Total Reviews: {current_review_count}\n"
                            f"• Tasks Status:\n" +
                            "\n".join([
                                f"  - {sent}: {remaining_distribution[sent]} remaining"
                                for sent in distribution.keys()
                            ])
                        )
                        
                        # If no progress in 30 seconds, might be stuck
                        if reviews_since_last == 0:
                            logger.warning("⚠️ No progress in last 30 seconds - might be stuck!")
                        
                        last_progress_time = current_time
                        last_review_count = current_review_count

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
                                
                                # Process reviews in memory-efficient way
                                temp_reviews.extend(reviews_to_add)
                                
                                # When we hit chunk size, process and clear temporary storage
                                if len(temp_reviews) >= chunk_size:
                                    # Process chunk
                                    for i, review in enumerate(temp_reviews):
                                        review["id"] = sentiment_counts["positive"] + sentiment_counts["negative"] + sentiment_counts["neutral"] + i + 1
                                    
                                    # Extend all_reviews with processed chunk
                                    all_reviews.extend(temp_reviews)
                                    # Clear temporary storage
                                    temp_reviews = []
                                    
                                    # Force garbage collection after processing large chunks
                                    if len(all_reviews) % (chunk_size * 5) == 0:
                                        import gc
                                        gc.collect()
                                
                                # Update stats for this sentiment
                                stats[sentiment].update(processing_time, len(reviews), len(reviews_to_add))

                            # If we have all the reviews we need
                            if all(count <= 0 for count in remaining_distribution.values()):
                                # Cancel all remaining tasks
                                for _, task in pending_tasks:
                                    task.cancel()
                                pending_tasks.clear()
                                break
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
                        LOG_FREQUENCY = 750  # Increased from 250
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

                # Process any remaining reviews in temp storage
                if temp_reviews:
                    start_id = len(all_reviews) + 1
                    for i, review in enumerate(temp_reviews):
                        review["id"] = start_id + i
                    all_reviews.extend(temp_reviews)
                
                # Calculate final distribution using chunks for memory efficiency
                final_distribution = {sentiment: 0 for sentiment in distribution.keys()}
                for chunk in chunk_reviews(all_reviews, chunk_size):
                    for review in chunk:
                        sentiment = review.get("sentiment")
                        if sentiment in final_distribution:
                            final_distribution[sentiment] += 1
                
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
                        "distribution": final_distribution,
                        "memory_efficient": True,
                        "chunk_size_used": chunk_size
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

def chunk_reviews(reviews: List[Dict[str, Any]], chunk_size: int = 1000) -> Iterator[List[Dict[str, Any]]]:
    """Process reviews in chunks to manage memory for large requests."""
    iterator = iter(reviews)
    return iter(lambda: list(islice(iterator, chunk_size)), [])
