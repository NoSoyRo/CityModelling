"""In-process pub/sub for live job events.

Each job (e.g. `classify_image`) publishes `JobEvent` objects on a single
`EventBus`. WebSocket clients subscribe and receive the stream filtered by
job id (or no filter, if they want everything).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import AsyncIterator

from .models import JobEvent


class EventBus:
    """Async fan-out broker with per-subscriber bounded queues."""

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[JobEvent]] = []
        self._history: dict[str, list[JobEvent]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def publish(self, event: JobEvent) -> None:
        async with self._lock:
            self._history[event.job_id].append(event)
            # Cap history per job to avoid unbounded growth.
            if len(self._history[event.job_id]) > 5000:
                self._history[event.job_id] = self._history[event.job_id][-5000:]
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop on the floor for slow consumers.
                pass

    async def subscribe(self) -> tuple[asyncio.Queue[JobEvent], AsyncIterator[JobEvent]]:
        queue: asyncio.Queue[JobEvent] = asyncio.Queue(maxsize=2048)
        async with self._lock:
            self._subscribers.append(queue)

        async def iterator() -> AsyncIterator[JobEvent]:
            try:
                while True:
                    yield await queue.get()
            finally:
                async with self._lock:
                    if queue in self._subscribers:
                        self._subscribers.remove(queue)

        return queue, iterator()

    def history(self, job_id: str) -> list[JobEvent]:
        return list(self._history.get(job_id, []))


_BUS = EventBus()


def get_bus() -> EventBus:
    return _BUS
