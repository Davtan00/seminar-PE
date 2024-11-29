import asyncio
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class ConcurrencyManager:
    def __init__(
        self,
        initial: int = 400,
        min_limit: int = 200,
        max_limit: int = 600,
        success_threshold: int = 3,
        failure_threshold: int = 3
    ):
        self.current_limit = initial
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.success_streak = 0
        self.failure_streak = 0
        self._semaphore = asyncio.Semaphore(initial)
        self._lock = asyncio.Lock()
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold
        self._adjustment_factor = 1.5  # More aggressive scaling

    async def acquire(self):
        """Acquire the semaphore"""
        await self._semaphore.acquire()

    def release(self, success: bool = True):
        """Release the semaphore and update stats"""
        self._semaphore.release()
        if success:
            asyncio.create_task(self.handle_success())  # Create task for async handling, avoids blocking the main thread
        else:
            asyncio.create_task(self.handle_failure())  # Create task for async handling, avoids blocking the main thread

    async def handle_success(self):
        """Handle successful operations with more aggressive scaling"""
        async with self._lock:
            self.success_streak += 1
            self.failure_streak = 0
            if self.success_streak >= self.success_threshold:
                increase = int(30 * self._adjustment_factor)  # More aggressive increase
                self.current_limit = min(self.current_limit + increase, self.max_limit)
                self._update_semaphore()
                self.success_streak = 0
                logger.debug(f"Increased concurrency limit to {self.current_limit}")

    async def handle_failure(self):
        """Handle failed operations with gentler reduction"""
        async with self._lock:
            self.failure_streak += 1
            self.success_streak = 0
            if self.failure_streak >= self.failure_threshold:
                decrease = int(20 * self._adjustment_factor)  # Smaller decrease
                self.current_limit = max(self.current_limit - decrease, self.min_limit)
                self._update_semaphore()
                self.failure_streak = 0
                logger.debug(f"Decreased concurrency limit to {self.current_limit}")

    def _update_semaphore(self):
        """Update the semaphore with the new limit"""
        self._semaphore = asyncio.Semaphore(self.current_limit)

    async def __aenter__(self):
        """Context manager support"""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        success = exc_type is None and exc_val is None
        self.release(success=success)