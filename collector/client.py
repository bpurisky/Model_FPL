"""FPL API client: retry, backoff, User-Agent, global rate limit (§2.2)."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta

import httpx

logger = logging.getLogger("collector.client")


class FPLNotFoundError(Exception):
    """404 from the FPL API. Expected for `picks` before a deadline (§2.2)."""

    def __init__(self, url: str):
        super().__init__(f"404 Not Found: {url}")
        self.url = url


class FPLServerError(Exception):
    """429 or 5xx from the FPL API, after retries are exhausted."""

    def __init__(self, url: str, status_code: int):
        super().__init__(f"Server error {status_code}: {url}")
        self.url = url
        self.status_code = status_code


def in_deadline_blackout(now: datetime, deadline: datetime | None, blackout_minutes: int = 10) -> bool:
    """True in the final `blackout_minutes` before `deadline` — peak load, don't poll (§2.2)."""
    if deadline is None:
        return False
    delta = deadline - now
    return timedelta(0) <= delta <= timedelta(minutes=blackout_minutes)


class RateLimiter:
    """Serialises calls to at most one per `min_interval` seconds, globally."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last_request: float | None = None

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if self._last_request is not None:
                elapsed = now - self._last_request
                if elapsed < self.min_interval:
                    await asyncio.sleep(self.min_interval - elapsed)
            self._last_request = time.monotonic()


class FPLClient:
    """Thin async wrapper around the FPL API with retry/backoff and rate limiting."""

    def __init__(
        self,
        base_url: str,
        user_agent: str,
        rate_limit_per_second: float = 1.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_jitter: float = 0.5,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_jitter = backoff_jitter
        self._rate_limiter = RateLimiter(1.0 / rate_limit_per_second)
        self._client = httpx.AsyncClient(
            headers={"User-Agent": user_agent},
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> "FPLClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._client.aclose()

    async def get_json(self, path: str):
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            await self._rate_limiter.wait()
            try:
                response = await self._client.get(url)
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    await self._backoff_sleep(attempt)
                    continue
                raise
            if response.status_code == 404:
                raise FPLNotFoundError(url)
            if response.status_code == 429 or response.status_code >= 500:
                last_exc = FPLServerError(url, response.status_code)
                if attempt < self.max_retries:
                    await self._backoff_sleep(attempt)
                    continue
                raise last_exc
            response.raise_for_status()
            return response.json()
        assert last_exc is not None
        raise last_exc

    async def _backoff_sleep(self, attempt: int) -> None:
        delay = self.backoff_base * (2**attempt) + random.uniform(0, self.backoff_jitter)
        logger.warning("Retrying after %.2fs (attempt %d/%d)", delay, attempt + 1, self.max_retries)
        await asyncio.sleep(delay)
