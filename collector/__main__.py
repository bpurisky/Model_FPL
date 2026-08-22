"""CLI entrypoint: `uv run python -m collector <command>`.

Not part of the file list in §1.2 verbatim, but something has to be the
thing GitHub Actions invokes — this is that thing, kept thin, holding no
logic beyond wiring config to the functions in snapshot.py / entry.py.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from collector.client import FPLClient, in_deadline_blackout
from collector.config import CollectorConfig, load_config
from collector import entry, snapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("collector.main")


def _next_deadline(reference_dir: Path) -> datetime | None:
    events_path = reference_dir / "events.parquet"
    if not events_path.exists():
        return None
    df = pl.read_parquet(events_path)
    upcoming = df.filter(~pl.col("finished")).sort("deadline_time")
    if upcoming.height == 0:
        return None
    value = upcoming["deadline_time"][0]
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


async def cmd_bootstrap(cfg: CollectorConfig) -> None:
    async with FPLClient(**cfg.api.client_kwargs()) as client:
        result = await snapshot.run_bootstrap_snapshot(client, cfg.storage)
    logger.info(
        "bootstrap snapshot complete: gw=%s rows_changed=%d/%d shard=%s",
        result.event_id, result.rows_changed, result.rows_total, result.distilled_shard,
    )


async def cmd_element_summary(cfg: CollectorConfig) -> None:
    players_path = Path(cfg.storage.reference_dir) / "players.parquet"
    if not players_path.exists():
        logger.error("players.parquet missing; run the bootstrap command first")
        sys.exit(1)
    element_ids = pl.read_parquet(players_path)["id"].to_list()
    async with FPLClient(**cfg.api.client_kwargs()) as client:
        result = await snapshot.run_element_summary_snapshot(client, cfg.storage, element_ids)
    logger.info("element-summary snapshot complete: fetched=%d missing=%d", result.fetched, result.missing)


async def cmd_live(cfg: CollectorConfig) -> None:
    fixtures_path = Path(cfg.storage.reference_dir) / "fixtures.parquet"
    if not fixtures_path.exists():
        logger.warning("fixtures.parquet missing; skipping live poll")
        return
    fixtures_df = pl.read_parquet(fixtures_path)
    now = datetime.now(timezone.utc)
    # `finished` alone reads a completed match as still live for hours
    # (collector/schemas.py:fixture_is_played), so treat a provisionally
    # finished fixture as done too. Guarded on the column existing because
    # this job reads whatever fixtures.parquet the last hourly run
    # committed, which is the pre-2026-08-22 six-column shape until one
    # has run since — the three-hour kickoff window below is the real
    # bound either way, so a missing column degrades, it doesn't break.
    played = pl.col("finished")
    if "finished_provisional" in fixtures_df.columns:
        played = played | pl.col("finished_provisional").fill_null(False)
    live_window = fixtures_df.filter(
        (~played)
        & pl.col("kickoff_time").is_not_null()
        & (pl.col("kickoff_time") <= now)
        & (pl.col("kickoff_time") >= now - pl.duration(hours=3))
    )
    if live_window.height == 0:
        logger.info("no fixtures in the live window, skipping")
        return
    event_ids = sorted({e for e in live_window["event"].drop_nulls().to_list()})
    async with FPLClient(**cfg.api.client_kwargs()) as client:
        for event_id in event_ids:
            result = await snapshot.run_event_live_snapshot(client, cfg.storage, event_id)
            logger.info("event-live snapshot complete: gw=%s raw=%s", result.event_id, result.raw_path)


async def cmd_prune(cfg: CollectorConfig) -> None:
    removed = snapshot.prune_raw(Path(cfg.storage.raw_dir), cfg.storage.raw_retention_days)
    logger.info("pruned %d stale raw files", len(removed))


async def cmd_entry(cfg: CollectorConfig) -> None:
    if cfg.own_entry_id is None:
        logger.info("own_entry_id not configured, skipping entry snapshot")
        return
    async with FPLClient(**cfg.api.client_kwargs()) as client:
        await entry.run_entry_snapshot(client, cfg.storage, cfg.own_entry_id)
        await entry.run_entry_history_snapshot(client, cfg.storage, cfg.own_entry_id)
        await entry.run_entry_transfers_snapshot(client, cfg.storage, cfg.own_entry_id)
    logger.info("entry snapshot complete for entry_id=%s", cfg.own_entry_id)


COMMANDS = {
    "bootstrap": cmd_bootstrap,
    "element-summary": cmd_element_summary,
    "live": cmd_live,
    "prune": cmd_prune,
    "entry": cmd_entry,
}

# Blackout applies to polling the live API; pruning only touches local disk.
_BLACKOUT_EXEMPT = {"prune"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m collector")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--config", default="config/collector.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(Path(args.config))

    if args.command not in _BLACKOUT_EXEMPT:
        deadline = _next_deadline(Path(cfg.storage.reference_dir))
        if in_deadline_blackout(datetime.now(timezone.utc), deadline, cfg.api.deadline_blackout_minutes):
            logger.info("within deadline blackout window (deadline=%s); skipping run", deadline)
            return

    asyncio.run(COMMANDS[args.command](cfg))


if __name__ == "__main__":
    main()
