"""Own-team endpoints (§2.1). Own team only in Phase 0 — squad reconstruction
for arbitrary team IDs is Phase 3."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from collector.client import FPLClient, FPLNotFoundError
from collector.config import StorageConfig
from collector.schemas import (
    SchemaValidationError,
    parse_entry,
    parse_entry_history,
    parse_entry_picks,
    parse_entry_transfers,
)
from collector.snapshot import utc_now, write_raw

logger = logging.getLogger("collector.entry")


@dataclass
class EntrySnapshotResult:
    timestamp: datetime
    raw_path: Path


async def run_entry_snapshot(client: FPLClient, storage: StorageConfig, entry_id: int, ts: datetime | None = None) -> EntrySnapshotResult:
    ts = ts or utc_now()
    raw_payload = await client.get_json(f"/entry/{entry_id}/")
    raw_path = write_raw(Path(storage.raw_dir), f"entry-{entry_id}", raw_payload, ts)
    try:
        parse_entry(raw_payload, logger)
    except ValidationError as exc:
        raise SchemaValidationError(f"entry-{entry_id}", raw_path, exc) from exc
    return EntrySnapshotResult(timestamp=ts, raw_path=raw_path)


async def run_entry_history_snapshot(client: FPLClient, storage: StorageConfig, entry_id: int, ts: datetime | None = None) -> EntrySnapshotResult:
    ts = ts or utc_now()
    raw_payload = await client.get_json(f"/entry/{entry_id}/history/")
    raw_path = write_raw(Path(storage.raw_dir), f"entry-{entry_id}-history", raw_payload, ts)
    try:
        parse_entry_history(raw_payload, logger)
    except ValidationError as exc:
        raise SchemaValidationError(f"entry-{entry_id}-history", raw_path, exc) from exc
    return EntrySnapshotResult(timestamp=ts, raw_path=raw_path)


async def run_entry_transfers_snapshot(client: FPLClient, storage: StorageConfig, entry_id: int, ts: datetime | None = None) -> EntrySnapshotResult:
    ts = ts or utc_now()
    raw_payload: list[dict[str, Any]] = await client.get_json(f"/entry/{entry_id}/transfers/")
    raw_path = write_raw(Path(storage.raw_dir), f"entry-{entry_id}-transfers", raw_payload, ts)
    try:
        parse_entry_transfers(raw_payload, logger)
    except ValidationError as exc:
        raise SchemaValidationError(f"entry-{entry_id}-transfers", raw_path, exc) from exc
    return EntrySnapshotResult(timestamp=ts, raw_path=raw_path)


async def run_entry_picks_snapshot(
    client: FPLClient, storage: StorageConfig, entry_id: int, event_id: int, ts: datetime | None = None
) -> EntrySnapshotResult | None:
    """404 before the deadline is expected, not an error (§2.1, §2.2) — logged
    at debug level and the caller gets None back rather than an exception."""
    ts = ts or utc_now()
    try:
        raw_payload = await client.get_json(f"/entry/{entry_id}/event/{event_id}/picks/")
    except FPLNotFoundError:
        logger.debug("picks 404 for entry=%s event=%s (expected before deadline)", entry_id, event_id)
        return None
    raw_path = write_raw(Path(storage.raw_dir), f"entry-{entry_id}-picks-gw{event_id}", raw_payload, ts)
    try:
        parse_entry_picks(raw_payload, logger)
    except ValidationError as exc:
        raise SchemaValidationError(f"entry-{entry_id}-picks-gw{event_id}", raw_path, exc) from exc
    return EntrySnapshotResult(timestamp=ts, raw_path=raw_path)
