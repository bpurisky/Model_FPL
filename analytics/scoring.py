"""Events -> points, driven entirely by config/scoring_*.yaml (§4.1).

`compute_points` is a pure function: no I/O, no season-awareness of its
own — every rule that varies by season lives in the config dict it's given,
never in this file. Changing a rule is a YAML edit, never a Python edit.

Bonus points are taken as a direct input on `EventVector`, not re-derived
from a BPS-ranking-within-fixture simulation. FPL's full BPS formula and
component weights have never been officially published — `compute_bps`
below is a documented, illustrative reconstruction covering the specific
rule deltas §4.1 calls out (the clearances/blocks/interceptions divisor,
the removed tackled deduction) plus the well-known core components, but it
is not claimed to reproduce the official BPS value exactly. The >=95%
points-reproduction acceptance bar (§4.1) does not depend on it — see
`validate_against_actual`, which checks total points using the official
`bonus` figure already present in the historical data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

# pyyaml: see collector/config.py — same justification, same file.
import yaml

POSITIONS = ("GK", "DEF", "MID", "FWD")


@dataclass(frozen=True)
class EventVector:
    """What actually happened for one player in one gameweek. Every field
    beyond `position` and `minutes` defaults to zero/None so a caller can
    construct one for a blank gameweek with `EventVector(position="MID", minutes=0)`.
    """

    position: str
    minutes: int
    goals_scored: int = 0
    assists: int = 0
    clean_sheets: int = 0
    goals_conceded: int = 0
    own_goals: int = 0
    penalties_saved: int = 0
    penalties_missed: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    saves: int = 0
    close_range_saves: int = 0  # subset of `saves` — 2026/27 GK save bonus
    big_chance_saves: int = 0  # subset of `saves` — 2026/27 GK save bonus
    bonus: int = 0
    defensive_contribution: int | None = None
    # Raw BPS inputs — used only by compute_bps, not compute_points.
    clearances_blocks_interceptions: int = 0
    tackles: int = 0
    times_tackled: int = 0  # not present in the historical dataset; see compute_bps


def load_scoring_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _minutes_points(event: EventVector, config: dict) -> int:
    cfg = config["minutes"]
    if event.minutes == 0:
        return cfg["none"]
    if event.minutes < cfg["short_threshold"]:
        return cfg["short"]
    return cfg["full"]


def _played_threshold(event: EventVector, config: dict) -> bool:
    return event.minutes >= config["minutes"]["short_threshold"]


def _saves_points(event: EventVector, config: dict) -> int:
    cfg = config["saves"]
    if cfg["mode"] == "per_n":
        return (event.saves // cfg["n"]) * cfg["points"]
    if cfg["mode"] == "flat_plus_bonus":
        return (
            event.saves * cfg["flat_rate"]
            + event.close_range_saves * cfg["close_range_bonus"]
            + event.big_chance_saves * cfg["big_chance_bonus"]
        )
    raise ValueError(f"unknown saves mode: {cfg['mode']!r}")


def _defensive_contribution_points(event: EventVector, config: dict) -> int:
    cfg = config.get("defensive_contribution")
    if not cfg or event.defensive_contribution is None:
        return 0
    threshold = cfg["thresholds"].get(event.position)
    if threshold is None:
        return 0
    return cfg["points"] if event.defensive_contribution >= threshold else 0


def compute_points_by_component(event: EventVector, config: dict[str, Any]) -> dict[str, float]:
    """The same arithmetic as compute_points, broken out by event-type
    bucket — feeds the true per-component error decomposition (§3.5, §4.4)
    once there's a per-component prediction to compare against (see
    analytics.projections.expected_points_by_component). compute_points
    itself is just sum(compute_points_by_component(...).values())."""
    played_60 = _played_threshold(event, config)
    gc_cfg = config["goals_conceded"]

    return {
        "minutes": _minutes_points(event, config),
        "goals": event.goals_scored * config["goals_scored"][event.position],
        "assists": event.assists * config["assists"][event.position],
        "clean_sheets": event.clean_sheets * config["clean_sheets"][event.position] if played_60 else 0,
        "goals_conceded": (
            (event.goals_conceded // gc_cfg["per"]) * gc_cfg["points"]
            if played_60 and event.position in gc_cfg["positions"]
            else 0
        ),
        "saves": _saves_points(event, config),
        "cards_and_other": (
            event.own_goals * config["own_goals"]
            + event.penalties_missed * config["penalties_missed"]
            + event.penalties_saved * config["penalties_saved"]
            + event.yellow_cards * config["yellow_cards"]
            + event.red_cards * config["red_cards"]
        ),
        "defensive_contribution": _defensive_contribution_points(event, config),
        "bonus": event.bonus,
    }


def compute_points(event: EventVector, config: dict[str, Any]) -> int:
    """The pure scoring function (§4.1). Every branch below reads a rule
    value out of `config` rather than hard-coding it, so a season's whole
    rule set is exactly what's in that season's YAML file."""
    return sum(compute_points_by_component(event, config).values())


# Illustrative-only BPS component weights (see module docstring). These are
# not sourced from an official FPL publication — there isn't one.
_BPS_GOAL_WEIGHTS = {"GK": 12, "DEF": 12, "MID": 18, "FWD": 24}


def compute_bps(event: EventVector, config: dict[str, Any]) -> int:
    """An illustrative, documented-as-approximate BPS reconstruction. Exists
    to make the two §4.1 rule deltas (CBI divisor, tackled deduction)
    config-driven and testable, not to reproduce FPL's real BPS value."""
    bps = 0
    if event.minutes > 0:
        bps += 6 if _played_threshold(event, config) else 3
    bps += event.goals_scored * _BPS_GOAL_WEIGHTS[event.position]
    bps += event.assists * 9
    if event.clean_sheets and _played_threshold(event, config) and event.position in ("GK", "DEF"):
        bps += 12
    bps += event.saves * 2
    bps += event.penalties_saved * 15
    bps -= event.penalties_missed * 6
    bps -= event.own_goals * 6
    bps -= event.yellow_cards * 3
    bps -= event.red_cards * 9
    if event.position in ("GK", "DEF"):
        bps -= (event.goals_conceded // 2) * 4
    per = config["bps_clearances_blocks_interceptions_per"]
    bps += (event.clearances_blocks_interceptions // per) * 2
    bps += event.tackles * 2
    bps += event.times_tackled * config["bps_tackled_penalty"]
    return bps


def event_vector_from_row(row: dict[str, Any]) -> EventVector:
    """Adapter from a data/historical/*.parquet row (or any dict with the
    same keys) to the pure-function input. Kept separate from
    `compute_points` so the scoring function itself never touches a
    DataFrame — see the module docstring."""
    return EventVector(
        position=row["position"],
        minutes=row["minutes"],
        goals_scored=row["goals_scored"],
        assists=row["assists"],
        clean_sheets=row["clean_sheets"],
        goals_conceded=row["goals_conceded"],
        own_goals=row["own_goals"],
        penalties_saved=row["penalties_saved"],
        penalties_missed=row["penalties_missed"],
        yellow_cards=row["yellow_cards"],
        red_cards=row["red_cards"],
        saves=row["saves"],
        bonus=row["bonus"],
        defensive_contribution=row.get("defensive_contribution"),
        clearances_blocks_interceptions=row.get("clearances_blocks_interceptions") or 0,
        tackles=row.get("tackles") or 0,
    )


@dataclass(frozen=True)
class ValidationReport:
    n_rows: int
    n_exact_match: int
    match_rate: float
    discrepancies: pl.DataFrame  # rows where predicted != actual, for investigation


def validate_against_actual(df: pl.DataFrame, config: dict[str, Any]) -> ValidationReport:
    """Replays every row of a normalized historical DataFrame through
    `compute_points` and compares to the official `total_points` already in
    the data (§4.1 acceptance: >=95% exact match on a completed gameweek).

    Investigated residual mismatch (both confirmed against real 2024-25/
    2025-26 rows, comfortably within the >=95% bar at 99.1%/99.3%):

    1. Double-gameweek rows (n_fixtures > 1, see backfill.py) sum banded/
       threshold stats — minutes, goals_conceded, defensive_contribution —
       across both fixtures before this ever sees them. FPL evaluates each
       banded rule *per fixture* and sums the resulting *points*; summing
       the raw *stats* first and thresholding once is not the same
       computation whenever a fixture's own total would have crossed a
       band on its own (floor(a+b)/n != floor(a/n)+floor(b/n) in general).
       Confirmed directly: 2024-25 gw33 Gundogan (2 fixtures, both 90 min)
       is under-predicted by exactly 2 — one fixture's worth of minutes
       points this aggregation collapses into one.
    2. Abandoned-and-replayed fixtures: FPL zeroes points for the voided
       original attempt outside its normal formula. Confirmed directly:
       2024-25 gw35 Brentford v Man Utd (fixture 343) has six players
       across both teams with real partial minutes (9-55) and negative BPS
       but total_points=0 — a scrape of the abandoned match's partial
       stats, not a live formula this module could replicate without a
       "this fixture was voided" flag the source data doesn't carry.

    Neither is a scoring.py defect; both are inherent to what a pure
    event-vector formula can know from this data. See the discrepancies
    output to inspect either cluster directly.
    """
    rows = df.to_dicts()
    predicted = [compute_points(event_vector_from_row(row), config) for row in rows]
    actual = [row["total_points"] for row in rows]
    matches = [p == a for p, a in zip(predicted, actual)]

    result_df = df.with_columns(pl.Series("predicted_points", predicted), pl.Series("points_match", matches))
    discrepancies = result_df.filter(~pl.col("points_match")).select(
        "season", "gw", "element_id", "name", "position", "n_fixtures", "total_points", "predicted_points"
    )

    n_rows = len(rows)
    n_exact_match = sum(matches)
    return ValidationReport(
        n_rows=n_rows,
        n_exact_match=n_exact_match,
        match_rate=n_exact_match / n_rows if n_rows else float("nan"),
        discrepancies=discrepancies,
    )
