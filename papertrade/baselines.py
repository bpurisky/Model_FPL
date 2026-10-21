"""§6.5 criteria 1-2's missing half: the three Phase 1 baselines
(`backtest/baselines.py`), run against this season's live actuals so the
frozen model has something live to be compared to.

**Retroactive, and still point-in-time.** Nothing here is frozen before a
deadline the way the model's projections are, and it doesn't need to be:
every baseline for target gameweek N reads only (a) actuals for gameweeks
< N, which `papertrade/actuals.py` stores append-only and never revises,
and (b) gw N's FPL fixture difficulty, published weeks before its deadline.
Recomputing gw6's baseline in gw13 gives exactly the number it would have
given on gw6's deadline day. `assert_no_leakage` checks that claim against
the real deadline rather than taking it on trust.

The one known caveat: `data/reference/fixtures.parquet` holds *today's*
schedule. A fixture rearranged after a deadline carries its final `event`,
not the one it had when the deadline passed — so a postponed match's
difficulty lands in the gameweek it was actually played in. That changes
which gameweek a difficulty rating is attached to, never which results the
trailing mean can see.

`backtest/baselines.py` runs unmodified: `live_panel` rebuilds the columns
the historical panel has and the actuals store deliberately doesn't
(`opponent_difficulty`, `kickoff_time` — see `papertrade/actuals.py`'s
module docstring), using the same double-gameweek aggregation as
`backtest/backfill.py` (earliest kickoff, mean difficulty).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from backtest.baselines import BASELINES
from backtest.harness import ROSTER_COLUMNS
from backtest.leakage import Feature, assert_no_leakage

REFERENCE_DIR = Path("data/reference")


@dataclass(frozen=True)
class LiveReference:
    fixtures: pl.DataFrame  # id, event, team_h, team_a, kickoff_time, team_h_difficulty, team_a_difficulty
    teams: pl.DataFrame  # id, name
    events: pl.DataFrame  # id, deadline_time


def load_reference(reference_dir: Path = REFERENCE_DIR) -> LiveReference:
    return LiveReference(
        fixtures=pl.read_parquet(reference_dir / "fixtures.parquet"),
        teams=pl.read_parquet(reference_dir / "teams.parquet"),
        events=pl.read_parquet(reference_dir / "events.parquet"),
    )


def _team_gw_context(reference: LiveReference) -> pl.DataFrame:
    """team name, gw -> earliest kickoff and mean opponent difficulty over
    every fixture that team has in that gameweek. A team with no fixture
    (blank gameweek) has no row, so the left join below leaves it null —
    which the fixture-adjusted baseline already reads as difficulty 3."""
    fixtures = reference.fixtures.filter(pl.col("event").is_not_null())
    sides = pl.concat([
        fixtures.select(pl.col("team_h").alias("team_id"), pl.col("event").alias("gw"), "kickoff_time",
                        pl.col("team_h_difficulty").alias("opponent_difficulty")),
        fixtures.select(pl.col("team_a").alias("team_id"), pl.col("event").alias("gw"), "kickoff_time",
                        pl.col("team_a_difficulty").alias("opponent_difficulty")),
    ])
    names = reference.teams.select(pl.col("id").alias("team_id"), pl.col("name").alias("team"))
    return (
        sides.join(names, on="team_id", how="inner")
        .group_by(["team", "gw"])
        .agg(
            pl.col("kickoff_time").min().alias("kickoff_time"),
            pl.col("opponent_difficulty").cast(pl.Float64).mean().alias("opponent_difficulty"),
        )
    )


def live_panel(actuals: pl.DataFrame, reference: LiveReference) -> pl.DataFrame:
    """The actuals store plus the fixture context `backtest/baselines.py`
    and `backtest/harness.py:ROSTER_COLUMNS` expect."""
    return actuals.join(_team_gw_context(reference), on=["team", "gw"], how="left")


def _deadline(reference: LiveReference, gw: int):
    match = reference.events.filter(pl.col("id") == gw)["deadline_time"]
    if match.len() == 0 or match[0] is None:
        raise ValueError(f"no deadline_time for gw{gw} in data/reference/events.parquet")
    return match[0]


def baseline_predictions(panel: pl.DataFrame, target_gw: int, reference: LiveReference) -> dict[str, pl.DataFrame] | None:
    """name -> (element_id, prediction) for every `BASELINES` entry, or
    None when there is no earlier gameweek to build a trailing mean from
    (every baseline would be the pooled prior of nothing — not a baseline).

    Raises `LeakageError` (it does not warn — §0.3) if any training row's
    latest kickoff isn't strictly before the target's real deadline.
    """
    train_df = panel.filter(pl.col("gw") < target_gw)
    target_df = panel.filter(pl.col("gw") == target_gw)
    if train_df.height == 0 or target_df.height == 0:
        return None

    target_roster = target_df.select(ROSTER_COLUMNS)
    deadline = _deadline(reference, target_gw)
    available_at = train_df["kickoff_time"].max()

    predictions: dict[str, pl.DataFrame] = {}
    for name, fn in BASELINES.items():
        prediction = fn(train_df, target_roster, target_gw)
        if available_at is not None:
            features = [
                Feature(name=name, element_id=row["element_id"], value=row["prediction"], available_at=available_at)
                for row in prediction.iter_rows(named=True)
            ]
            assert_no_leakage(features, deadline, context=f"live gw{target_gw} {name}")
        predictions[name] = prediction
    return predictions
