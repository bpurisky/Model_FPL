"""Wires the event model (projections.py) into the same walk-forward
harness the three Phase 1 baselines already use (§4.4), for a direct,
apples-to-apples comparison: MAE, within-position Spearman, calibration,
and error decomposition, per season and pooled.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Callable

import polars as pl

from analytics.fdr import team_gameweek_difficulty
from analytics.projections import expected_points_by_component, project_event_vectors, project_points
from analytics.scoring import EventVector, compute_points_by_component, load_scoring_config
from backtest.backfill import NORMALIZED_DIR, RAW_CACHE_DIR, load_match_results, load_teams
from backtest.baselines import BASELINES
from backtest.harness import ROSTER_COLUMNS, walk_forward

_ACTUAL_EVENT_COLUMNS = [
    "element_id", "position", "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded",
    "own_goals", "penalties_saved", "penalties_missed", "yellow_cards", "red_cards", "saves", "bonus",
    "defensive_contribution",
]

# 2023-24 has no scoring rule difference from 2024-25 (defensive
# contribution and the BPS/save-formula changes are both later), and the
# repo layout (§1.2) only calls for scoring_2024_25/2025_26/2026_27.yaml —
# no separate 2023-24 file, reused deliberately rather than duplicated.
SEASON_SCORING_CONFIG = {
    "2023-24": "config/scoring_2024_25.yaml",
    "2024-25": "config/scoring_2024_25.yaml",
    "2025-26": "config/scoring_2025_26.yaml",
}


def build_difficulty_table(season: str) -> pl.DataFrame:
    matches = load_match_results(RAW_CACHE_DIR / season / "fixtures.csv")
    teams = load_teams(RAW_CACHE_DIR / season / "teams.csv")
    return team_gameweek_difficulty(matches, teams)


def build_season_baselines(season: str, on_projection: Callable[[int, pl.DataFrame], None] | None = None) -> dict:
    config = load_scoring_config(Path(SEASON_SCORING_CONFIG[season]))
    difficulty_table = build_difficulty_table(season)
    model_fn = functools.partial(
        project_points, config=config, difficulty_table=difficulty_table, on_projection=on_projection
    )
    return {**BASELINES, "event_model": model_fn}


def _decompose_gameweek(projected: pl.DataFrame, target_df: pl.DataFrame, config: dict, out: dict[str, list]) -> None:
    """Predicted and actual points per component for one gameweek, plus the
    minutes head's predicted distribution against actual minutes — the
    detail backtest.report's component_decomposition_mae /
    minutes_head_metrics need (§4.4)."""
    actual_by_id = {row["element_id"]: row for row in target_df.select(_ACTUAL_EVENT_COLUMNS).to_dicts()}
    for row in projected.to_dicts():
        actual_row = actual_by_id.get(row["element_id"])
        if actual_row is None:
            continue
        out["predicted_components"].append(expected_points_by_component(row, config))
        actual_event = EventVector(**{c: actual_row[c] for c in _ACTUAL_EVENT_COLUMNS if c != "element_id"})
        out["actual_components"].append(compute_points_by_component(actual_event, config))
        out["predicted_minutes_dist"].append({"p_blank": row["p_blank"], "p_short": row["p_short"], "p_full": row["p_full"]})
        out["actual_minutes"].append(actual_row["minutes"])


def run_evaluation(seasons: list[str] | None = None) -> tuple[pl.DataFrame, dict[str, list]]:
    """Walk-forward results for the three Phase 1 baselines plus the event
    model (in the shape backtest.report.build_report expects), and the
    event model's component decomposition, from a single walk-forward.

    The decomposition used to walk every season again just to recover the
    event vectors the event model had already projected. Now the event
    model hands them over through `project_points(on_projection=...)`, so
    each gameweek is projected once. The vectors are the same ones the
    points prediction was made from, so the two halves cannot disagree.
    """
    seasons = seasons or list(SEASON_SCORING_CONFIG)
    batches = []
    decomposition: dict[str, list] = {
        "predicted_components": [], "actual_components": [],
        "predicted_minutes_dist": [], "actual_minutes": [],
    }
    for season in seasons:
        season_df = pl.read_parquet(NORMALIZED_DIR / f"{season}.parquet")
        config = load_scoring_config(Path(SEASON_SCORING_CONFIG[season]))
        projections: dict[int, pl.DataFrame] = {}
        baselines = build_season_baselines(season, on_projection=projections.__setitem__)
        batches.append(walk_forward(season_df, season, baselines))
        for target_gw, projected in sorted(projections.items()):
            _decompose_gameweek(projected, season_df.filter(pl.col("gw") == target_gw), config, decomposition)

    non_empty = [b for b in batches if b.height > 0]
    return (pl.concat(non_empty) if non_empty else pl.DataFrame()), decomposition
