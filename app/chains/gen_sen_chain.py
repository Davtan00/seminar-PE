from fastapi import HTTPException
import aiohttp
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass
from math import ceil

logger = logging.getLogger(__name__)
### Only so fast because of rate limit, but we aren't pushing it so hard so perhaps T2,T1 would be okay performance wise as well.
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
        if self.remaining_requests < 10 or self.remaining_tokens < 1000:
            await asyncio.sleep(0.1)  # Small delay to allow for rate limit reset


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
        max_tokens_per_request = 4000  # Increased but won't hit this due to review cap
        theoretical_max_reviews = min(
            25,  # Hard cap at 25 reviews
            (max_tokens_per_request - self.token_estimates.base_prompt_tokens - self.token_estimates.json_overhead) // self.token_estimates.tokens_per_review
        )
        
        # Calculate theoretical max based on rate limits
        theoretical_max_by_tokens = (self.rate_limits.tokens_per_minute // 60) // self.token_estimates.get_total_tokens_for_batch(1)
        
        max_batch_size = min(
            25,  # Fixed max batch size
            theoretical_max_reviews,
            theoretical_max_by_tokens,
            self.rate_limits.min_quality_threshold
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

@dataclass
class ReviewParameters:
    # Primary parameters
    privacy_level: float
    domain_relevance: float
    cultural_sensitivity: float
    bias_control: float
    
    # Style parameters
    realism: float
    formality: float
    lexical_complexity: float
    sentiment_intensity: float
    temporal_relevance: float
    noise_level: float
    
    def get_prompt_description(self) -> str:
        return f"""Primary Parameters:
        - Privacy level: {self.privacy_level} (ensure maximum anonymization; avoid any specific identifiable details)
        - Domain relevance: {self.domain_relevance} (maintain strong focus on domain-specific content and terminology)
        - Cultural sensitivity: {self.cultural_sensitivity} (use highly inclusive language; be considerate of diverse backgrounds)
        - Bias control: {self.bias_control} (actively minimize demographic, cultural, and personal biases)
        
        Style Parameters:
        - Realism: {self.realism} (produce authentic-sounding content matching real user behavior)
        - Formality: {self.formality} (adjust language formality from casual to professional)
        - Lexical complexity: {self.lexical_complexity} (control vocabulary complexity and sentence structure)
        - Sentiment intensity: {self.sentiment_intensity} (modulate the strength of emotional expression)
        - Temporal relevance: {self.temporal_relevance} (include current trends and contemporary references)
        - Noise level: {self.noise_level} (control amount of tangential or supplementary information)"""



class GenSenChain:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        rate_limits = RateLimits(
            requests_per_minute=5000,
            tokens_per_minute=4_000_000,
            tokens_per_hour=100_000_000
        )
        self.rate_limiter = RateLimiter(rate_limits)
        self.token_estimates = TokenEstimates()
        
        self.system_prompt = """You are a specialized content generator that creates synthetic reviews or comments.
            Your output must be a JSON object with a 'reviews' array containing the generated items.
            Each item must have 'text' and 'sentiment' fields."""

    async def process_batch(
        self,
        session: aiohttp.ClientSession,
        batch_params: Dict[str, Any],
        retries: int = 3
    ) -> Optional[List[Dict[str, Any]]]:
        total_requested = batch_params['count']
        collected_reviews = []

        # Add domain validation
        if batch_params['domain'].lower() == 'all':
            raise HTTPException(
                status_code=400,
                detail="Please specify a concrete domain for review generation"
            )
        
        while retries > 0 and total_requested > 0:
            try:
                await self.rate_limiter.wait_if_needed()
                
                # Create ReviewParameters instance from batch_params
                parameters = ReviewParameters(
                    privacy_level=batch_params.get('privacy_level', 0.8),
                    domain_relevance=batch_params.get('domainrelevance', 0.8), ##Maybe remove, conflicts with system prompt
                    cultural_sensitivity=batch_params.get('culturalsensitivity', 0.8),
                    bias_control=batch_params.get('bias_control', 0.7),
                    
                    realism=batch_params.get('realism', 0.7),
                    formality=batch_params.get('formality', 0.5),
                    lexical_complexity=batch_params.get('lexicalcomplexity', 0.5),
                    sentiment_intensity=batch_params.get('sentiment_intensity', 0.5),
                    temporal_relevance=batch_params.get('temporal_relevance', 0.5),
                    noise_level=batch_params.get('noise_level', 0.3)
                )
                logger.debug(f"Processing batch with parameters: {parameters}")
                # Construct the new user prompt
                user_prompt = f"""Generate {total_requested} {batch_params['sentiment']} reviews specifically about {batch_params['domain']}.
                    Focus on concrete aspects, features, or experiences related to {batch_params['domain']}.
                    Avoid generic statements that could apply to any domain.

                    {parameters.get_prompt_description()}

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
                    
                    reviews = parsed_data["reviews"]
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
            max_concurrent = optimal_config['concurrent_tasks']
            all_reviews = []
            semaphore = asyncio.Semaphore(max_concurrent)

            async with aiohttp.ClientSession() as session:
                async def bounded_process_batch(batch_params):
                    async with semaphore:
                        return await self.process_batch(session, batch_params)

           
            
          
                tasks = []
                for sentiment, count in distribution.items():
                    if count <= 0:
                        continue
                    
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
                        tasks.append(bounded_process_batch(batch_params))

                        
                for batch_results in asyncio.as_completed(tasks):
                    try:
                        reviews = await batch_results
                        if reviews:
                            all_reviews.extend(reviews)
                            logger.info(f"Successfully processed batch. Total reviews so far: {len(all_reviews)}")
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