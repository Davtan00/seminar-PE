from langchain_community.callbacks import get_openai_callback
from contextlib import contextmanager
from typing import Generator, Dict, Any

@contextmanager
def track_cost() -> Generator[Dict[str, Any], None, None]:
    """Track the cost of OpenAI API calls"""
    with get_openai_callback() as callback:
        yield {
            "total_tokens": callback.total_tokens,
            "prompt_tokens": callback.prompt_tokens,
            "completion_tokens": callback.completion_tokens,
            "total_cost": callback.total_cost,
        }
