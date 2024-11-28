import asyncio
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class ConcurrencyManager:
    def __init__(
        self,
        initial: int = 200,
        min_limit: int = 100,
        max_limit: int = 300,
        increment: int = 10,
        decrement: int = 20
    ):
        self.current_limit = initial
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.increment = increment
        self.decrement = decrement
        self._semaphore = asyncio.Semaphore(initial)

    async def acquire(self):
        """Acquire the semaphore"""
        await self._semaphore.acquire()

    def release(self):
        """Release the semaphore"""
        self._semaphore.release()

    def increase_limit(self):
        """Increase concurrency limit after successful operations"""
        if self.current_limit < self.max_limit:
            self.current_limit = min(self.max_limit, self.current_limit + self.increment)
            self._update_semaphore()

    def decrease_limit(self):
        """Decrease concurrency limit after failures or rate limits"""
        if self.current_limit > self.min_limit:
            self.current_limit = max(self.min_limit, self.current_limit - self.decrement)
            self._update_semaphore()

    def _update_semaphore(self):
        """Update the semaphore with the new limit"""
        new_semaphore = asyncio.Semaphore(self.current_limit)
        self._semaphore = new_semaphore

    async def __aenter__(self):
        """Context manager support"""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.release()
        if exc_type is None:
            self.increase_limit()
        elif "429" in str(exc_val):  # Rate limit error
            self.decrease_limit()