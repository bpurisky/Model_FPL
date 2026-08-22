"""Pydantic models for every FPL payload the collector ingests.

Every model allows extra fields (schema drift is expected and should warn,
not fail — §0.5) but declares the fields the rest of the pipeline actually
depends on as required. A required field going missing is real schema drift
and must halt the run; an unrecognised extra field is merely noteworthy.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
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
    "fixture_is_played",
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
    # Required alongside `finished` because the pipeline genuinely depends
    # on it (see fixture_is_played) — under this module's own rule, that
    # makes a missing one real drift and a hard error, not a warning.
    finished_provisional: bool
    # Recorded for the reference tier rather than depended on yet: scores
    # are what analytics/fdr.py's Elo will need to rate current-season
    # form once enough gameweeks exist, and until Phase 0 persisted them
    # they were simply thrown away every run. Optional, so a missing one
    # doesn't halt a collector run over data nothing reads yet.
    started: bool | None = None
    team_h_score: int | None = None
    team_a_score: int | None = None


_FixturesAdapter = TypeAdapter(list[Fixture])


def fixture_is_played(fixture: Mapping[str, Any]) -> bool:
    """Whether a fixture is over, and so whether its stats can be learned from.

    FPL exposes two flags here and they are not interchangeable.
    `finished_provisional` flips at full time, once provisional bonus has
    been applied. `finished` flips only once the gameweek's data has been
    confirmed, which lags by many hours — verified live on 2026-08-22,
    when gw1's Friday fixture (Arsenal 3-0, kicked off 2026-08-21T19:00Z)
    still reported `finished: false` alongside `finished_provisional: true`
    more than eighteen hours after full time.

    Gating "has this team played?" on `finished` alone therefore reads a
    completed match as unplayed for most of a gameweek. That is not a
    hypothetical: `squad/live.py:build_train_df` did exactly this, so every
    player fell through to the pooled prior and `papertrade/freezes/gw2.json`
    was frozen with one identical projection (0.8) for all 600 players.

    Deliberately NOT the right predicate for "is this gameweek's result
    final" — provisional bonus can still move. `papertrade/actuals.py`
    gates on event-level `finished` for that, and should keep doing so:
    recording actuals must wait for confirmation, learning trailing rates
    from minutes and goals need not.

    Takes a raw payload dict rather than a `Fixture`, because every caller
    (squad/live.py, via papertrade/freeze.py) works from the raw
    `/fixtures/` JSON that `parse_fixtures` only validates in passing.
    """
    return bool(fixture.get("finished") or fixture.get("finished_provisional"))


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
    """Every per-gameweek stat papertrade/actuals.py writes to the
    append-only store, declared required rather than left to `extra`.

    Same reasoning as `Fixture.finished_provisional`: the pipeline
    genuinely depends on these, so one going missing is real drift and
    belongs as a hard error here — with the endpoint and the offending
    payload attached — rather than as a `KeyError` raised later inside
    `_build_actuals_frame`, after the drift has already been read past.

    The four `expected_*` fields arrive as decimal strings; pydantic
    coerces them, and the frame builder parses the raw dict separately.
    """

    minutes: int
    total_points: int
    bps: int
    goals_scored: int
    assists: int
    clean_sheets: int
    goals_conceded: int
    saves: int
    bonus: int
    yellow_cards: int
    red_cards: int
    own_goals: int
    penalties_missed: int
    penalties_saved: int
    defensive_contribution: int
    starts: int
    clearances_blocks_interceptions: int
    recoveries: int
    tackles: int
    expected_goals: float
    expected_assists: float
    expected_goal_involvements: float
    expected_goals_conceded: float
    influence: float
    creativity: float
    threat: float
    ict_index: float


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
