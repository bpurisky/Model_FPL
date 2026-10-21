"""Elite (top-N) ownership source: the picks of the overall league's top N.

Reads the classic overall league (id 314) standings, 50 entries a page, and
fetches each top manager's picks for the most recently deadlined gameweek
(`/entry/{id}/event/{gw}/picks/`). One row per manager x pick lands in
`data/distilled/gw{gw}/top_picks.parquet`; `web/export/ownership.py` turns
that into top-N ownership and effective ownership.

**Rate-limit choice.** Strictly sequential through the existing throttled
`FPLClient` (1 request/second in config/collector.yaml): N=1000 is 20
standings pages + 1000 picks calls, about 17 minutes, once per gameweek.
No parallelism, no extra burst. The job is idempotent (skips when the file
exists) so the 3-hourly cron costs nothing after the first success.

**Failure policy.** A manager whose picks fail (404, schema, server error)
is logged and skipped. If more than `MAX_FAIL_FRACTION` of the attempted
managers fail, nothing is written and the command fails, so a half-blind
file never stands in for the real thing.

**Rank caveat.** Standings are read when the job runs, so rank is "current
overall rank", which can differ slightly from the rank at the deadline if
the job runs after matches have been played. The population is "the top N
managers right now"; picks are as of the deadline.

**Shard readers.** The file sits beside the timestamped delta shards in
`gw{n}/` but is a different table; every shard reader skips it by name
(`collector.snapshot.TOP_PICKS_FILENAME`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from collector.client import FPLClient
from collector.schemas import parse_entry_picks, parse_league_standings
from collector.snapshot import TOP_PICKS_FILENAME, utc_now

logger = logging.getLogger("collector.top_picks")

OVERALL_LEAGUE_ID = 314
PAGE_SIZE = 50
DEFAULT_TOP_N = 1000
MAX_FAIL_FRACTION = 0.10

TOP_PICKS_SCHEMA = {
    "rank": pl.Int64,
    "entry_id": pl.Int64,
    "gw": pl.Int64,
    "element": pl.Int64,
    "position": pl.Int64,
    "multiplier": pl.Int64,
    "is_captain": pl.Boolean,
    "is_vice_captain": pl.Boolean,
    "active_chip": pl.Utf8,
    "fetched_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}


class TopPicksFailure(RuntimeError):
    """Too many managers failed; nothing was written."""


@dataclass
class TopPicksResult:
    gw: int
    path: Path
    n_requested: int
    n_fetched: int
    n_failed: int
    skipped: bool = False


def current_gameweek(events_path: Path, now: datetime | None = None) -> int | None:
    """Most recent gameweek whose deadline has passed (picks are public then)."""
    if not events_path.exists():
        return None
    now = now or utc_now()
    df = pl.read_parquet(events_path).filter(pl.col("deadline_time").is_not_null())
    passed = []
    for row in df.to_dicts():
        deadline = row["deadline_time"]
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        if deadline <= now:
            passed.append(row["id"])
    return max(passed) if passed else None


async def fetch_top_entries(client: FPLClient, top_n: int) -> list[tuple[int, int]]:
    """(rank, entry_id) for the top `top_n` of the overall league, in rank order."""
    entries: list[tuple[int, int]] = []
    page = 1
    while len(entries) < top_n:
        raw = await client.get_json(f"/leagues-classic/{OVERALL_LEAGUE_ID}/standings/?page_standings={page}")
        parsed = parse_league_standings(raw, logger).standings
        entries.extend((r.rank, r.entry) for r in parsed.results)
        if not parsed.has_next or not parsed.results:
            break
        page += 1
    return entries[:top_n]


async def run_top_picks(
    client: FPLClient,
    distilled_dir: Path,
    gw: int,
    top_n: int = DEFAULT_TOP_N,
    force: bool = False,
    ts: datetime | None = None,
) -> TopPicksResult:
    path = distilled_dir / f"gw{gw}" / TOP_PICKS_FILENAME
    if path.exists() and not force:
        logger.info("top picks for gw%d already collected (%s); skipping", gw, path)
        return TopPicksResult(gw, path, 0, 0, 0, skipped=True)

    ts = ts or utc_now()
    entries = await fetch_top_entries(client, top_n)
    rows: list[dict] = []
    failed = 0
    for rank, entry_id in entries:
        try:
            raw = await client.get_json(f"/entry/{entry_id}/event/{gw}/picks/")
            payload = parse_entry_picks(raw, logger)
        except Exception as exc:  # 404, 5xx after retries, transport, schema: all skip-this-manager
            failed += 1
            logger.warning("top picks: skipping entry %s (rank %s): %r", entry_id, rank, exc)
            continue
        for pick in payload.picks:
            rows.append({
                "rank": rank, "entry_id": entry_id, "gw": gw, "element": pick.element,
                "position": pick.position, "multiplier": pick.multiplier,
                "is_captain": pick.is_captain, "is_vice_captain": pick.is_vice_captain,
                "active_chip": payload.active_chip, "fetched_at": ts,
            })

    total = len(entries)
    if total == 0 or failed / total > MAX_FAIL_FRACTION:
        raise TopPicksFailure(
            f"top picks gw{gw}: {failed}/{total} managers failed (limit {MAX_FAIL_FRACTION:.0%}); nothing written"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=TOP_PICKS_SCHEMA).write_parquet(path)
    logger.info("top picks gw%d: %d managers (%d failed) -> %s", gw, total - failed, failed, path)
    return TopPicksResult(gw, path, total, total - failed, failed)
