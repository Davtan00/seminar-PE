from langchain_community.callbacks import get_openai_callback
from contextlib import contextmanager
from typing import Generator, Dict, Any

class CostTracker:
    def __init__(self, callback):
        self.callback = callback

    def get_costs(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.callback.total_tokens,
            "prompt_tokens": self.callback.prompt_tokens,
            "completion_tokens": self.callback.completion_tokens,
            "total_cost": self.callback.total_cost,
        }

@contextmanager
def track_cost() -> Generator[CostTracker, None, None]:
    """Track the cost of OpenAI API calls"""
    with get_openai_callback() as callback:
        yield CostTracker(callback)
