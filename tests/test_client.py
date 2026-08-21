"""§2.2: global 1 req/sec rate limit; exponential backoff with jitter on
429/5xx, max 5 retries; deadline blackout window."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from collector.client import FPLClient, FPLNotFoundError, FPLServerError, in_deadline_blackout


def _client(transport: httpx.MockTransport, **kwargs) -> FPLClient:
    defaults = dict(
        base_url="https://example.invalid/api",
        user_agent="test-agent",
        rate_limit_per_second=kwargs.pop("rate_limit_per_second", 5.0),
        max_retries=kwargs.pop("max_retries", 5),
        backoff_base=kwargs.pop("backoff_base", 0.01),
        backoff_jitter=kwargs.pop("backoff_jitter", 0.01),
    )
    defaults.update(kwargs)
    return FPLClient(transport=transport, **defaults)


def test_rate_limit_serialises_requests_to_one_per_interval():
    async def scenario():
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        async with _client(transport, rate_limit_per_second=5.0) as client:
            start = time.monotonic()
            await asyncio.gather(*(client.get_json("/x/") for _ in range(4)))
            elapsed = time.monotonic() - start
        # 4 requests at 5/sec => at least 3 gaps of 0.2s = 0.6s minimum.
        assert elapsed >= 0.6 - 0.05

    asyncio.run(scenario())


def test_no_burst_exceeds_configured_rate():
    async def scenario():
        timestamps: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            timestamps.append(time.monotonic())
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        async with _client(transport, rate_limit_per_second=10.0) as client:
            await asyncio.gather(*(client.get_json("/x/") for _ in range(6)))

        timestamps.sort()
        gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
        min_interval = 1.0 / 10.0
        assert all(gap >= min_interval - 0.02 for gap in gaps)

    asyncio.run(scenario())


def test_retries_on_500_then_succeeds():
    async def scenario():
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(500)
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        async with _client(transport, max_retries=5) as client:
            result = await client.get_json("/x/")
        assert result == {"ok": True}
        assert calls["n"] == 3

    asyncio.run(scenario())


def test_retries_on_429_then_succeeds():
    async def scenario():
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 2:
                return httpx.Response(429)
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        async with _client(transport, max_retries=5) as client:
            result = await client.get_json("/x/")
        assert result == {"ok": True}
        assert calls["n"] == 2

    asyncio.run(scenario())


def test_gives_up_after_max_retries():
    async def scenario():
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503)

        transport = httpx.MockTransport(handler)
        async with _client(transport, max_retries=2) as client:
            with pytest.raises(FPLServerError):
                await client.get_json("/x/")
        assert calls["n"] == 3  # initial attempt + 2 retries

    asyncio.run(scenario())


def test_404_raises_not_found_without_retrying():
    async def scenario():
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        async with _client(transport, max_retries=5) as client:
            with pytest.raises(FPLNotFoundError):
                await client.get_json("/missing/")
        assert calls["n"] == 1

    asyncio.run(scenario())


def test_in_deadline_blackout():
    deadline = datetime(2026, 8, 21, 17, 30, tzinfo=timezone.utc)
    assert in_deadline_blackout(deadline - timedelta(minutes=5), deadline, blackout_minutes=10)
    assert not in_deadline_blackout(deadline - timedelta(minutes=15), deadline, blackout_minutes=10)
    assert not in_deadline_blackout(deadline + timedelta(minutes=1), deadline, blackout_minutes=10)
    assert not in_deadline_blackout(datetime.now(timezone.utc), None, blackout_minutes=10)
