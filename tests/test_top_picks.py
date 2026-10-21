"""`collector top-picks`: standings paging, picks rows, idempotence, fail-soft
policy, and that shard readers ignore the file. Fake client, no network."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import polars as pl
import pytest

from collector.client import FPLNotFoundError
from collector.top_picks import TopPicksFailure, current_gameweek, run_top_picks
from collector.snapshot import latest_distilled_state


class FakeClient:
    def __init__(self, n_entries: int, missing: set[int] = frozenset()):
        self.n = n_entries
        self.missing = missing
        self.calls: list[str] = []

    async def get_json(self, path: str):
        self.calls.append(path)
        if "/standings/" in path:
            page = int(path.rsplit("=", 1)[1])
            lo, hi = (page - 1) * 50 + 1, min(page * 50, self.n)
            return {
                "league": {"id": 314},
                "standings": {
                    "has_next": hi < self.n, "page": page,
                    "results": [{"entry": 1000 + r, "rank": r, "player_name": "x"} for r in range(lo, hi + 1)],
                },
            }
        entry = int(path.split("/")[2])
        if entry in self.missing:
            raise FPLNotFoundError(path)
        return {
            "active_chip": "3xc" if entry == 1001 else None,
            "entry_history": {},
            "picks": [
                {"element": 10 + i, "position": i + 1, "multiplier": 0 if i >= 11 else (2 if i == 0 else 1),
                 "is_captain": i == 0, "is_vice_captain": i == 1}
                for i in range(15)
            ],
        }


def _run(coro):
    return asyncio.run(coro)


def test_writes_one_row_per_manager_pick_across_pages(tmp_path):
    client = FakeClient(120)
    res = _run(run_top_picks(client, tmp_path, gw=7, top_n=120))
    df = pl.read_parquet(tmp_path / "gw7" / "top_picks.parquet")

    assert (res.n_fetched, res.n_failed, res.skipped) == (120, 0, False)
    assert df.height == 120 * 15
    assert df["entry_id"].n_unique() == 120 and set(df["gw"]) == {7}
    assert df.filter(pl.col("entry_id") == 1001)["active_chip"].unique().to_list() == ["3xc"]
    assert df.filter((pl.col("rank") == 2) & pl.col("is_captain"))["multiplier"].to_list() == [2]
    assert sum("standings" in c for c in client.calls) == 3
    assert set(df.columns) == {"rank", "entry_id", "gw", "element", "position", "multiplier",
                               "is_captain", "is_vice_captain", "active_chip", "fetched_at"}


def test_top_n_truncates(tmp_path):
    _run(run_top_picks(FakeClient(120), tmp_path, gw=7, top_n=60))
    assert pl.read_parquet(tmp_path / "gw7" / "top_picks.parquet")["entry_id"].n_unique() == 60


def test_idempotent_and_force(tmp_path):
    _run(run_top_picks(FakeClient(10), tmp_path, gw=3, top_n=10))
    again = FakeClient(10)
    res = _run(run_top_picks(again, tmp_path, gw=3, top_n=10))
    assert res.skipped and again.calls == []
    forced = FakeClient(10)
    assert not _run(run_top_picks(forced, tmp_path, gw=3, top_n=10, force=True)).skipped
    assert forced.calls


def test_failed_manager_is_skipped_under_the_threshold(tmp_path):
    res = _run(run_top_picks(FakeClient(50, missing={1005, 1006}), tmp_path, gw=2, top_n=50))
    df = pl.read_parquet(tmp_path / "gw2" / "top_picks.parquet")
    assert (res.n_fetched, res.n_failed) == (48, 2)
    assert df["entry_id"].n_unique() == 48


def test_too_many_failures_fails_and_writes_nothing(tmp_path):
    with pytest.raises(TopPicksFailure):
        _run(run_top_picks(FakeClient(50, missing={1000 + r for r in range(1, 10)}), tmp_path, gw=2, top_n=50))
    assert not (tmp_path / "gw2" / "top_picks.parquet").exists()


def test_current_gameweek_is_latest_passed_deadline(tmp_path):
    path = tmp_path / "events.parquet"
    pl.DataFrame({
        "id": [1, 2, 3],
        "deadline_time": [datetime(2026, 8, d, 17, 30, tzinfo=timezone.utc) for d in (14, 21, 28)],
    }).write_parquet(path)
    assert current_gameweek(path, datetime(2026, 8, 22, tzinfo=timezone.utc)) == 2
    assert current_gameweek(path, datetime(2026, 8, 1, tzinfo=timezone.utc)) is None
    assert current_gameweek(tmp_path / "nope.parquet") is None


def test_shard_readers_ignore_top_picks(tmp_path):
    _run(run_top_picks(FakeClient(5), tmp_path, gw=1, top_n=5))
    assert latest_distilled_state(tmp_path / "gw1") is None
