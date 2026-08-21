"""§3.5: walk-forward never trains on data from after the prediction target,
and the leakage assertion (§3.3) is actually wired into the loop — not just
a standalone check nobody calls."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from backtest.baselines import BASELINES
from backtest.harness import walk_forward
from backtest.leakage import LeakageError

BASE_DAY = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _season_df(n_gws: int = 6, n_players: int = 4) -> pl.DataFrame:
    rows = []
    for gw in range(1, n_gws + 1):
        for element_id in range(1, n_players + 1):
            rows.append(
                {
                    "gw": gw,
                    "element_id": element_id,
                    "position": "MID" if element_id % 2 == 0 else "FWD",
                    "team": "Team A",
                    "is_promoted_club": False,
                    "opponent_difficulty": 3,
                    "kickoff_time": BASE_DAY + timedelta(days=7 * (gw - 1)),
                    "total_points": (element_id * gw) % 12,
                    "minutes": 90,
                    "goals_scored": 0,
                    "assists": 0,
                    "clean_sheets": 0,
                    "bonus": 0,
                    "bps": 20,
                }
            )
    return pl.DataFrame(rows)


def test_never_trains_on_future_gameweeks():
    season_df = _season_df(n_gws=6)
    seen_max_train_gw: dict[int, int] = {}

    def spy_baseline(train_df, target_roster, target_gw):
        seen_max_train_gw[target_gw] = train_df["gw"].max()
        return target_roster.select("element_id").with_columns(pl.lit(3.0).alias("prediction"))

    walk_forward(season_df, "2024-25", baselines={"spy": spy_baseline})

    assert seen_max_train_gw  # the loop actually ran
    for target_gw, max_train_gw in seen_max_train_gw.items():
        assert max_train_gw < target_gw


def test_walk_forward_runs_clean_on_well_formed_data():
    season_df = _season_df(n_gws=6)
    results = walk_forward(season_df, "2024-25", baselines=BASELINES)
    assert results.height > 0
    assert set(BASELINES) <= set(results["baseline"].unique().to_list())
    assert results["gw"].min() >= 2


def test_walk_forward_raises_on_mislabeled_future_data():
    """A row scraped with the wrong round number (a real data-quality risk,
    not a contrived one) lands in an earlier gameweek's training window while
    actually belonging to match day ~100. The leakage assertion must catch
    this even though nothing in the *code path* is misbehaving — the bad
    input alone is enough, which is the point of asserting in code (§0.3)."""
    season_df = _season_df(n_gws=6)
    mislabeled = season_df.filter((pl.col("gw") == 6) & (pl.col("element_id") == 1)).with_columns(
        pl.lit(2).alias("gw"),  # relabel a far-future row as gw2
    )
    corrupted = pl.concat([season_df, mislabeled], how="vertical_relaxed")

    with pytest.raises(LeakageError):
        walk_forward(corrupted, "2024-25", baselines={"trailing_mean": BASELINES["trailing_mean"]})
