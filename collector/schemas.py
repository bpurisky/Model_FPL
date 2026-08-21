"""Pydantic models for every FPL payload the collector ingests.

Every model allows extra fields (schema drift is expected and should warn,
not fail — §0.5) but declares the fields the rest of the pipeline actually
depends on as required. A required field going missing is real schema drift
and must halt the run; an unrecognised extra field is merely noteworthy.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

__all__ = [
    "SchemaValidationError",
    "Team",
    "Event",
    "Element",
    "BootstrapStatic",
    "Fixture",
    "ElementSummaryHistory",
    "ElementSummary",
    "LiveElementStats",
    "LiveElement",
    "EventLive",
    "Entry",
    "EntryHistoryCurrent",
    "EntryHistoryPast",
    "EntryChip",
    "EntryHistory",
    "EntryTransfer",
    "EntryPick",
    "EntryPicksPayload",
    "resolve_current_event",
    "resolve_next_event",
    "parse_bootstrap_static",
    "parse_fixtures",
    "parse_element_summary",
    "parse_event_live",
    "parse_entry",
    "parse_entry_history",
    "parse_entry_transfers",
    "parse_entry_picks",
]


class SchemaValidationError(RuntimeError):
    """Raised when an ingested payload fails schema validation.

    Per §0.5 / §2.4: this halts the run. The raw payload has already been
    written to disk by the caller before validation runs, so it is preserved
    for offline inspection regardless of this exception.
    """

    def __init__(self, endpoint: str, raw_path: Path | None, original: Exception):
        message = f"Schema validation failed for '{endpoint}'"
        if raw_path is not None:
            message += f"; raw payload preserved at {raw_path}"
        message += f": {original}"
        super().__init__(message)
        self.endpoint = endpoint
        self.raw_path = raw_path
        self.original = original


class _LenientModel(BaseModel):
    model_config = ConfigDict(extra="allow")


def warn_on_extra_fields(instance: BaseModel, context: str, logger: logging.Logger) -> None:
    """Log (not raise) when a payload carries fields the schema doesn't know about."""
    extra = instance.model_extra
    if extra:
        logger.warning("Unexpected fields in %s: %s", context, sorted(extra.keys()))


def _warn_all(items: list[BaseModel], context: str, logger: logging.Logger) -> None:
    """One aggregated warning per collection, not one per item.

    Fields we deliberately didn't model (most of bootstrap-static's ~90
    element fields, say) show up as "extra" on every single item; warning
    per-item buries a genuine drift signal — a field that's actually new —
    under noise. The union of extra field names across the collection is
    the useful signal.
    """
    extra_keys: set[str] = set()
    for item in items:
        if item.model_extra:
            extra_keys.update(item.model_extra.keys())
    if extra_keys:
        logger.warning("Unexpected fields in %s (%d items): %s", context, len(items), sorted(extra_keys))


# --------------------------------------------------------------------------
# bootstrap-static
# --------------------------------------------------------------------------


class Team(_LenientModel):
    id: int
    name: str
    short_name: str
    # Null pre-season, before FPL computes strength ratings for the new
    # campaign (observed live 2026-08-21) — a legitimate value, not drift.
    strength: int | None = None


class Event(_LenientModel):
    """A gameweek.

    Cross-season drift (§2.4): older seasons expose the current/next
    gameweek only via top-level `current_event` / `next_event` integers on
    the bootstrap-static payload. 2025/26+ instead marks it per-event with
    `is_current` / `is_next` booleans here. Both are optional on this model;
    `resolve_current_event` / `resolve_next_event` handle the fallback.
    """

    id: int
    name: str
    deadline_time: datetime
    finished: bool
    is_current: bool | None = None
    is_next: bool | None = None
    is_previous: bool | None = None


class Element(_LenientModel):
    """A player. FPL serialises several numeric fields as strings."""

    id: int
    web_name: str
    team: int
    element_type: int
    now_cost: int
    selected_by_percent: str
    transfers_in_event: int
    transfers_out_event: int
    form: str
    status: str
    chance_of_playing_next_round: int | None = None
    news_added: datetime | None = None
    ep_next: str | None = None


class BootstrapStatic(_LenientModel):
    events: list[Event]
    teams: list[Team]
    elements: list[Element]


def resolve_current_event(bootstrap: BootstrapStatic, raw: dict[str, Any]) -> int | None:
    for event in bootstrap.events:
        if event.is_current:
            return event.id
    legacy = raw.get("current_event")
    return legacy if isinstance(legacy, int) else None


def resolve_next_event(bootstrap: BootstrapStatic, raw: dict[str, Any]) -> int | None:
    for event in bootstrap.events:
        if event.is_next:
            return event.id
    legacy = raw.get("next_event")
    return legacy if isinstance(legacy, int) else None


def parse_bootstrap_static(raw: dict[str, Any], logger: logging.Logger) -> BootstrapStatic:
    bootstrap = BootstrapStatic.model_validate(raw)
    warn_on_extra_fields(bootstrap, "bootstrap-static", logger)
    _warn_all(bootstrap.teams, "bootstrap-static.teams", logger)
    _warn_all(bootstrap.events, "bootstrap-static.events", logger)
    _warn_all(bootstrap.elements, "bootstrap-static.elements", logger)
    return bootstrap


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


class Fixture(_LenientModel):
    id: int
    event: int | None
    team_h: int
    team_a: int
    kickoff_time: datetime | None
    finished: bool


_FixturesAdapter = TypeAdapter(list[Fixture])


def parse_fixtures(raw: list[dict[str, Any]], logger: logging.Logger) -> list[Fixture]:
    fixtures = _FixturesAdapter.validate_python(raw)
    _warn_all(fixtures, "fixtures", logger)
    return fixtures


# --------------------------------------------------------------------------
# element-summary
# --------------------------------------------------------------------------


class ElementSummaryHistory(_LenientModel):
    element: int
    fixture: int
    round: int
    minutes: int
    total_points: int


class ElementSummary(_LenientModel):
    fixtures: list[dict[str, Any]] = Field(default_factory=list)
    history: list[ElementSummaryHistory] = Field(default_factory=list)
    history_past: list[dict[str, Any]] = Field(default_factory=list)


def parse_element_summary(raw: dict[str, Any], logger: logging.Logger, context: str = "element-summary") -> ElementSummary:
    summary = ElementSummary.model_validate(raw)
    warn_on_extra_fields(summary, context, logger)
    _warn_all(summary.history, f"{context}.history", logger)
    return summary


# --------------------------------------------------------------------------
# event live
# --------------------------------------------------------------------------


class LiveElementStats(_LenientModel):
    minutes: int
    total_points: int
    bps: int


class LiveElement(_LenientModel):
    id: int
    stats: LiveElementStats


class EventLive(_LenientModel):
    elements: list[LiveElement]


def parse_event_live(raw: dict[str, Any], logger: logging.Logger, context: str = "event-live") -> EventLive:
    live = EventLive.model_validate(raw)
    warn_on_extra_fields(live, context, logger)
    _warn_all(live.elements, f"{context}.elements", logger)
    for element in live.elements:
        warn_on_extra_fields(element.stats, f"{context}.elements.stats", logger)
    return live


# --------------------------------------------------------------------------
# entry (own team)
# --------------------------------------------------------------------------


class Entry(_LenientModel):
    id: int
    started_event: int
    summary_overall_points: int
    current_event: int | None = None


def parse_entry(raw: dict[str, Any], logger: logging.Logger) -> Entry:
    entry = Entry.model_validate(raw)
    warn_on_extra_fields(entry, "entry", logger)
    return entry


class EntryHistoryCurrent(_LenientModel):
    event: int
    points: int
    total_points: int
    bank: int
    value: int
    event_transfers: int
    event_transfers_cost: int


class EntryHistoryPast(_LenientModel):
    season_name: str
    total_points: int


class EntryChip(_LenientModel):
    name: str
    event: int


class EntryHistory(_LenientModel):
    current: list[EntryHistoryCurrent]
    past: list[EntryHistoryPast]
    chips: list[EntryChip]


def parse_entry_history(raw: dict[str, Any], logger: logging.Logger) -> EntryHistory:
    history = EntryHistory.model_validate(raw)
    warn_on_extra_fields(history, "entry-history", logger)
    _warn_all(history.current, "entry-history.current", logger)
    _warn_all(history.chips, "entry-history.chips", logger)
    return history


class EntryTransfer(_LenientModel):
    element_in: int
    element_in_cost: int
    element_out: int
    element_out_cost: int
    entry: int
    event: int
    time: datetime


_EntryTransfersAdapter = TypeAdapter(list[EntryTransfer])


def parse_entry_transfers(raw: list[dict[str, Any]], logger: logging.Logger) -> list[EntryTransfer]:
    transfers = _EntryTransfersAdapter.validate_python(raw)
    _warn_all(transfers, "entry-transfers", logger)
    return transfers


class EntryPick(_LenientModel):
    element: int
    position: int
    multiplier: int
    is_captain: bool
    is_vice_captain: bool
    # §5.3 says the picks endpoint provides selling_price directly. Verified
    # live against a real 2026/27 gw1 payload (2026-08-21) and it does not:
    # neither field is present pre-match, at least this season. Optional,
    # not required, so a real payload doesn't hard-fail schema validation
    # (§0.5) — squad/reconstruct.py's caller falls back to current now_cost
    # when these are absent, which is exact (not approximate) for a squad
    # that has never been transferred, since no price move is possible with
    # zero elapsed time. Revisit if a later gameweek's payload does include
    # them — that would mean the fields activate post-deadline-lock rather
    # than being gone for good.
    purchase_price: int | None = None
    selling_price: int | None = None


class EntryPicksPayload(_LenientModel):
    picks: list[EntryPick]
    entry_history: dict[str, Any]
    active_chip: str | None = None


def parse_entry_picks(raw: dict[str, Any], logger: logging.Logger) -> EntryPicksPayload:
    payload = EntryPicksPayload.model_validate(raw)
    warn_on_extra_fields(payload, "entry-picks", logger)
    _warn_all(payload.picks, "entry-picks.picks", logger)
    return payload


# Re-export for callers that want to catch the underlying pydantic error type
# without importing pydantic directly.
__all__.append("ValidationError")
