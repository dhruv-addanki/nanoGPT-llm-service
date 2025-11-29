"""Simple in-process queue and concurrency limiter."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass


@dataclass
class QueueStats:
    waiting: int
    active: int


class GenerationQueue:
    """A lightweight queue that caps concurrent generations and exposes metrics."""

    def __init__(self, max_concurrent: int):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._waiting = 0
        self._active = 0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self):
        async with self._lock:
            self._waiting += 1
        await self.semaphore.acquire()
        async with self._lock:
            self._waiting -= 1
            self._active += 1
        try:
            yield
        finally:
            async with self._lock:
                self._active -= 1
            self.semaphore.release()

    async def stats(self) -> QueueStats:
        async with self._lock:
            return QueueStats(waiting=self._waiting, active=self._active)

