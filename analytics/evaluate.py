"""Wires the event model (projections.py) into the same walk-forward
harness the three Phase 1 baselines already use (§4.4), for a direct,
apples-to-apples comparison: MAE, within-position Spearman, calibration,
and error decomposition, per season and pooled.
"""

from __future__ import annotations

import functools
from pathlib import Path

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


def build_season_baselines(season: str) -> dict:
    config = load_scoring_config(Path(SEASON_SCORING_CONFIG[season]))
    difficulty_table = build_difficulty_table(season)
    model_fn = functools.partial(project_points, config=config, difficulty_table=difficulty_table)
    return {**BASELINES, "event_model": model_fn}


def run_comparison(seasons: list[str] | None = None) -> pl.DataFrame:
    """Walk-forward results for the three Phase 1 baselines plus the event
    model, across the given seasons (default: all three backtest seasons),
    in the same shape backtest.report.build_report expects."""
    seasons = seasons or list(SEASON_SCORING_CONFIG)
    batches = []
    for season in seasons:
        season_df = pl.read_parquet(NORMALIZED_DIR / f"{season}.parquet")
        baselines = build_season_baselines(season)
        batches.append(walk_forward(season_df, season, baselines))
    non_empty = [b for b in batches if b.height > 0]
    return pl.concat(non_empty) if non_empty else pl.DataFrame()


def run_component_decomposition(seasons: list[str] | None = None) -> dict[str, list]:
    """Walks forward like run_comparison, but for the event model only,
    capturing the full per-component breakdown (predicted and actual) and
    the minutes-head's own predicted-vs-actual — the detail backtest.report's
    component_decomposition_mae / minutes_head_metrics need (§4.4) that the
    generic walk_forward loop's [element_id, prediction] shape doesn't carry.
    """
    seasons = seasons or list(SEASON_SCORING_CONFIG)
    predicted_components: list[dict] = []
    actual_components: list[dict] = []
    predicted_minutes_dist: list[dict] = []
    actual_minutes: list[int] = []

    for season in seasons:
        season_df = pl.read_parquet(NORMALIZED_DIR / f"{season}.parquet")
        config = load_scoring_config(Path(SEASON_SCORING_CONFIG[season]))
        difficulty_table = build_difficulty_table(season)
        max_gw = season_df["gw"].max()
        if max_gw is None:
            continue

        for target_gw in range(2, max_gw + 1):
            train_df = season_df.filter(pl.col("gw") < target_gw)
            target_df = season_df.filter(pl.col("gw") == target_gw)
            if target_df.height == 0:
                continue

            gw_difficulty = difficulty_table.filter(pl.col("gw") == target_gw).select("team", "custom_difficulty")
            roster = target_df.select(ROSTER_COLUMNS).join(gw_difficulty, on="team", how="left").with_columns(
                pl.col("custom_difficulty").fill_null(3.0)
            )
            projected = project_event_vectors(train_df, roster, target_gw, config)

            actual_by_id = {row["element_id"]: row for row in target_df.select(_ACTUAL_EVENT_COLUMNS).to_dicts()}

            for row in projected.to_dicts():
                actual_row = actual_by_id.get(row["element_id"])
                if actual_row is None:
                    continue
                predicted_components.append(expected_points_by_component(row, config))
                actual_event = EventVector(
                    position=actual_row["position"],
                    minutes=actual_row["minutes"],
                    goals_scored=actual_row["goals_scored"],
                    assists=actual_row["assists"],
                    clean_sheets=actual_row["clean_sheets"],
                    goals_conceded=actual_row["goals_conceded"],
                    own_goals=actual_row["own_goals"],
                    penalties_saved=actual_row["penalties_saved"],
                    penalties_missed=actual_row["penalties_missed"],
                    yellow_cards=actual_row["yellow_cards"],
                    red_cards=actual_row["red_cards"],
                    saves=actual_row["saves"],
                    bonus=actual_row["bonus"],
                    defensive_contribution=actual_row["defensive_contribution"],
                )
                actual_components.append(compute_points_by_component(actual_event, config))
                predicted_minutes_dist.append({"p_blank": row["p_blank"], "p_short": row["p_short"], "p_full": row["p_full"]})
                actual_minutes.append(actual_row["minutes"])

    return {
        "predicted_components": predicted_components,
        "actual_components": actual_components,
        "predicted_minutes_dist": predicted_minutes_dist,
        "actual_minutes": actual_minutes,
    }
