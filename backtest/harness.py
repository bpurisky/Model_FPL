"""Walk-forward validation (§3.5): train on gameweeks 1..N, predict N+1,
advance, repeat. Never trains on data from after the prediction target.

Each season is walked independently — FPL element ids aren't stable across
seasons (they're reassigned each season), so there is no valid way to build
a cross-season trailing feature from vaastav's per-season ids without an
identity-resolution step this phase doesn't need. §3.5's "report per-season
and pooled" already implies per-season walks whose *metrics* get pooled,
not one continuous multi-season walk — that's what this does.

Gameweek deadline_time isn't published in the historical source data,
so it's approximated as the earliest kickoff_time among that gameweek's
fixtures. The true deadline is somewhat earlier, which only makes the
leakage assertion below *stricter* than necessary, never looser: every
feature here is built from strictly prior gameweeks with days of margin,
so the approximation doesn't change which runs pass or fail.
"""

from __future__ import annotations

from typing import Callable

import polars as pl

from backtest.baselines import BASELINES
from backtest.leakage import Feature, assert_no_leakage

ROSTER_COLUMNS = ["element_id", "position", "team", "is_promoted_club", "opponent_difficulty", "kickoff_time"]
ACTUAL_COLUMNS = ["element_id", "total_points", "minutes", "goals_scored", "assists", "clean_sheets", "bonus", "bps"]

BaselineFn = Callable[[pl.DataFrame, pl.DataFrame, int], pl.DataFrame]


def walk_forward(season_df: pl.DataFrame, season: str, baselines: dict[str, BaselineFn] | None = None) -> pl.DataFrame:
    baselines = baselines if baselines is not None else BASELINES
    max_gw = season_df["gw"].max()
    if max_gw is None:
        return pl.DataFrame()

    batches: list[pl.DataFrame] = []

    for target_gw in range(2, max_gw + 1):
        train_df = season_df.filter(pl.col("gw") < target_gw)
        target_df = season_df.filter(pl.col("gw") == target_gw)
        if target_df.height == 0:
            continue

        target_roster = target_df.select(ROSTER_COLUMNS)
        actuals = target_df.select(ACTUAL_COLUMNS)
        deadline_time = target_df["kickoff_time"].min()
        available_at = train_df["kickoff_time"].max()

        for name, fn in baselines.items():
            predictions = fn(train_df, target_roster, target_gw)

            features = [
                Feature(name=name, element_id=row["element_id"], value=row["prediction"], available_at=available_at)
                for row in predictions.iter_rows(named=True)
            ]
            assert_no_leakage(features, deadline_time, context=f"{season} gw{target_gw} {name}")

            merged = (
                predictions.join(target_roster.select(["element_id", "position"]), on="element_id", how="left")
                .join(actuals, on="element_id", how="inner")
                .with_columns(
                    pl.lit(season).alias("season"),
                    pl.lit(target_gw).alias("gw"),
                    pl.lit(name).alias("baseline"),
                    (pl.col("prediction") - pl.col("total_points")).alias("error"),
                )
            )
            batches.append(merged)

    return pl.concat(batches) if batches else pl.DataFrame()


def walk_forward_all_seasons(season_dfs: dict[str, pl.DataFrame], baselines: dict[str, BaselineFn] | None = None) -> pl.DataFrame:
    batches = [walk_forward(df, season, baselines) for season, df in season_dfs.items()]
    batches = [b for b in batches if b.height > 0]
    return pl.concat(batches) if batches else pl.DataFrame()
