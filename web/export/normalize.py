"""Within-position normalization (§5.7), computed here and never in the
browser (§5.6.2).

The doctrine in one line: a defender will not post a forward's xG, and
that says nothing about whether he is a good defender. Every cross-player
comparison is therefore against the player's own position group, and each
normalized value ships with the population it was computed against.

Three companion columns per normalizable metric, per §5.7.2:

    {key}_z_pos     z-score against the position group
    {key}_pct_pos   percentile within the position group
    {key}_n_pos     the size of that group

`_n_pos` is not optional. It is what lets §5.7.4's tooltip say "1.34 sigma
above DEF mean (n=147, >=450 min, GW1-8)" rather than presenting a bare
number, and a z-score without its basis is an unfalsifiable number.

Eligibility (§5.15 Q5) is the decision this module exists to encode; see
`eligible_mask` for why the floor scales rather than sitting at 450.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import yaml

from web.export.columns import companion_keys

FRONTEND_CONFIG = Path("config/frontend.yaml")


def load_frontend_config(path: Path = FRONTEND_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def minutes_floor(n_fixtures_played: pl.Expr | int, per_fixture: int) -> pl.Expr | int:
    """The eligibility bar, as a function of how much football has been
    played rather than as a constant.

    Measured against 2024-25, the proposed flat 450 admits **zero players
    in every position until gameweek 5** -- the normalized surfaces would
    render empty for the opening month of a season, which is when an FPL
    tool is used most. Scaling the bar with fixtures played holds the
    eligible population between 212 and 224 from GW1 to GW38, and that
    stability is the real prize: it means a z-score in GW3 is computed
    against the same *kind* of reference group as one in GW30, so the
    scale does not quietly change meaning as the season runs.

    Counted per team fixture rather than per gameweek elapsed so that
    double gameweeks and blanks are fair rather than punitive.
    """
    return n_fixtures_played * per_fixture


def eligible_mask(df: pl.DataFrame, per_fixture: int) -> pl.Series:
    """Which rows enter the reference population.

    A player below the floor is excluded from the population *and* gets
    null normalized values -- not zero (§5.3.3). A 30-minute cameo should
    not move the positional mean, and a z-score of 0.0 would claim the
    player is exactly average, which is a finding rather than a gap.
    """
    return df["cum_minutes"] >= minutes_floor(df["cum_fixtures"], per_fixture)


def with_season_to_date(df: pl.DataFrame) -> pl.DataFrame:
    """Cumulative minutes and fixtures per player, up to and including
    each row's gameweek.

    Season-to-date is the basis named in the header
    (`within_position_season_to_date`), so it is computed from the panel
    rather than passed in: a basis the file claims and the numbers do not
    share is worse than no basis at all.
    """
    return df.sort(["season", "element_id", "gw"]).with_columns(
        pl.col("minutes").cum_sum().over(["season", "element_id"]).alias("cum_minutes"),
        pl.col("n_fixtures").cum_sum().over(["season", "element_id"]).alias("cum_fixtures"),
    )


def normalize_column(df: pl.DataFrame, key: str, per_fixture: int) -> pl.DataFrame:
    """Add `{key}_z_pos`, `{key}_pct_pos` and `{key}_n_pos`.

    Grouped by (season, gw, position): the population is "players at this
    position, in this gameweek, who have played enough this season", which
    is what a reader comparing two players in a given week actually means.

    Both the ineligible and those whose metric is itself null are excluded
    from the moments. The second case is not an edge: four of the sixteen
    metrics do not exist before 2025-26, and letting a null-filled column
    contribute a zero to its own mean would corrupt every other player's
    z-score in the group.
    """
    z_key, pct_key, n_key = companion_keys(key)

    eligible = pl.col("_eligible") & pl.col(key).is_not_null()
    grp = ["season", "gw", "position"]

    # Moments over the eligible subset only. `pl.when(...).then(col)` leaves
    # null for excluded rows, and polars' mean/std/count all skip nulls.
    contributing = pl.when(eligible).then(pl.col(key)).otherwise(None)

    df = df.with_columns(
        contributing.mean().over(grp).alias("_mu"),
        contributing.std().over(grp).alias("_sigma"),
        contributing.count().over(grp).alias(n_key),
        contributing.rank(method="average").over(grp).alias("_rank"),
    )

    # A zero or absent sigma yields null, not a division blow-up or a
    # spurious 0.0: if every eligible player at a position posted the same
    # value, no one is above or below the mean and saying so is honest.
    z = (
        pl.when(eligible & pl.col("_sigma").is_not_null() & (pl.col("_sigma") > 0))
        .then((pl.col(key) - pl.col("_mu")) / pl.col("_sigma"))
        .otherwise(None)
    )
    # Percentile from the average-rank convention, so ties share a value
    # rather than being ordered arbitrarily. Guarded at n=1, where a
    # percentile is not defined.
    pct = (
        pl.when(eligible & (pl.col(n_key) > 1))
        .then((pl.col("_rank") - 1) / (pl.col(n_key) - 1))
        .otherwise(None)
    )

    return df.with_columns(z.alias(z_key), pct.alias(pct_key)).drop("_mu", "_sigma", "_rank")


def normalize(df: pl.DataFrame, keys: list[str], per_fixture: int | None = None) -> pl.DataFrame:
    """Add the three companion columns for every key in `keys`.

    `df` must already carry the metric columns themselves plus `season`,
    `gw`, `position`, `minutes` and `n_fixtures`.
    """
    if per_fixture is None:
        per_fixture = load_frontend_config()["normalization"]["minutes_per_fixture_floor"]

    out = with_season_to_date(df)
    out = out.with_columns(eligible_mask(out, per_fixture).alias("_eligible"))
    for key in keys:
        out = normalize_column(out, key, per_fixture)
    return out.drop("_eligible")


def normalization_basis(per_fixture: int | None = None) -> str:
    """The string the header carries and §5.7.4's tooltip renders."""
    if per_fixture is None:
        per_fixture = load_frontend_config()["normalization"]["minutes_per_fixture_floor"]
    return f"within_position_season_to_date_min{per_fixture}_per_fixture"
