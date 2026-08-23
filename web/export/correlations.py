"""`correlations.json` — precomputed rank-correlation matrices (§5.3.2).

Feeds the Correlation Lab (§5.4.1), the hero surface. The position filter
must be instant (§5.9: matrix re-render <= 100 ms), which is the whole
reason these are precomputed here rather than recomputed in the browser --
and §5.6 forbids the browser inferring anyway.

Three decisions are baked in and each one changes every number in the file.

**Grain: one row per player per season, not per player-gameweek.** A
gameweek-grain correlation would run over ~85,000 rows and be worthless
twice over. Every metric here is a per-90 rate, so they all share the same
1/minutes denominator; correlating X/m against Y/m across rows where m
varies from 3 to 90 manufactures correlation out of the denominator alone.
And a 10-minute substitute who scores once posts xg_per90 far above any
season rate, which is noise wearing a metric's clothes. Season rates are
the quantity a reader means when they ask whether two metrics travel
together.

**Season rates are minutes-weighted, not averaged.** The mean of a
player's weekly per-90 rates is not his season per-90 rate -- it weights a
7-minute cameo equally with a full match. Weighting each gameweek rate by
its own minutes recovers the exact season figure, since
`sum(rate_i * m_i) / sum(m_i)` telescopes back to `90 * sum(stat_i) /
sum(m_i)`. That identity is asserted in the tests rather than trusted.

**Seasons pool.** All three archive seasons stack into one population per
position group. The consequence to keep in view: metric availability
differs by season (tackles, recoveries, CBI and defensive contribution
exist only from 2025-26), so a cell pairing one of those with xG draws its
n from one season while the cell beside it draws from three. That is not
hidden -- every cell carries its own n, and `columns.json` already records
each metric's `available_from_season`, so the UI can state the coverage
without this file duplicating it per cell.

The `all` group pools positions and therefore carries the §5.7.5
distortion in correlation form: forwards and goalkeepers differ so
systematically on most of these metrics that position alone drives part of
any pooled rho. It ships because §5.4.1 offers an "all" filter, it is not
the default (§5.7.3 defaults the Lab to position-filtered), and it is
flagged `mixed_position` so that caution copy can attach to it. Silently
dropping it would be the worse choice: the user would get no answer rather
than a labelled one.
"""

from __future__ import annotations

import logging
import math
from itertools import combinations
from pathlib import Path

import polars as pl

from backtest.report import spearman_with_significance
from web.export.columns import MATRIX_METRICS
from web.export.contract import (
    CorrelationCell,
    CorrelationsFile,
    GroupSummary,
    build_header,
)
from web.export.normalize import load_frontend_config, minutes_floor

logger = logging.getLogger("web.export.correlations")

PANEL_PATH = Path("data/web/v1/panel.parquet")

# "all" first: it is the union, and the UI renders the filter in this order.
POSITIONS = ["GK", "DEF", "MID", "FWD"]
GROUPS = ["all", *POSITIONS]


def _finite(value: float | None) -> float | None:
    """NaN and infinity are not JSON, and `float('nan')` serializes to a
    bare `NaN` token that `JSON.parse` rejects outright.

    They are also not the same claim as a number. `spearman` returns NaN
    for a degenerate input -- fewer than two pairs, or a metric with no
    spread -- and that means "no correlation is defined here", which is
    §5.3.3's null, not a zero.
    """
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def complete_pairs(x: pl.Series, y: pl.Series) -> tuple[pl.Series, pl.Series]:
    """Rows where both metrics are present.

    Not optional, and not a tidiness measure. `report.spearman` ranks each
    series independently -- polars' `rank()` leaves nulls null and ranks
    only the rest -- while `spearman_with_significance` reports `n` as
    `x.len()`, which counts the nulls it just ignored. Feed it a column
    that is half null and a *perfect* rank correlation comes back as
    rho=0.23 over n=10 with p=0.52: no relationship, with a large sample
    apparently backing it. Verified 2026-08-23 against a constructed pair.

    Four of the fourteen metrics are null for two of the three pooled
    seasons, so this is the normal case here, not an edge.
    """
    mask = x.is_not_null() & y.is_not_null()
    return x.filter(mask), y.filter(mask)


def season_rate(metric: str) -> tuple[pl.Expr, pl.Expr]:
    """Numerator and denominator of the minutes-weighted season rate.

    Kept as two aggregates rather than one ratio because the division has
    to be guarded after aggregation: polars sums an all-null column to 0
    rather than to null, so a metric that does not exist in a season would
    otherwise divide 0 by 0 and reach the export as NaN.
    """
    weighted = (pl.col(metric) * pl.col("minutes")).sum().alias(f"_num_{metric}")
    covered = (
        pl.when(pl.col(metric).is_not_null())
        .then(pl.col("minutes"))
        .otherwise(None)
        .sum()
        .alias(f"_den_{metric}")
    )
    return weighted, covered


def player_season_frame(
    panel: pl.DataFrame, metrics: list[str], per_fixture: int
) -> pl.DataFrame:
    """One row per (season, element_id), eligible players only.

    Eligibility is §5.15 Q5's rolling floor, evaluated at the end of the
    player's season: `cum_minutes` and `cum_fixtures` are already
    season-to-date cumulatives in the panel, so their maxima are the
    season totals. Reusing `minutes_floor` rather than restating 45 keeps
    the correlation population and the normalization population defined by
    the same rule -- if that floor moves, both move together.
    """
    aggs: list[pl.Expr] = [
        pl.col("position").drop_nulls().last().alias("position"),
        pl.col("cum_minutes").max().alias("season_minutes"),
        pl.col("cum_fixtures").max().alias("season_fixtures"),
    ]
    for metric in metrics:
        aggs.extend(season_rate(metric))

    grouped = panel.group_by(["season", "element_id"]).agg(aggs)

    rates = [
        pl.when(pl.col(f"_den_{m}") > 0)
        .then(pl.col(f"_num_{m}") / pl.col(f"_den_{m}"))
        .otherwise(None)
        .alias(m)
        for m in metrics
    ]
    grouped = grouped.with_columns(rates).drop(
        [f"_num_{m}" for m in metrics] + [f"_den_{m}" for m in metrics]
    )

    eligible = pl.col("season_minutes") >= minutes_floor(
        pl.col("season_fixtures"), per_fixture
    )
    return grouped.filter(eligible).sort(["season", "element_id"])


def correlate_group(df: pl.DataFrame, metrics: list[str], group: str) -> list[dict]:
    """Every unordered metric pair for one position group.

    The diagonal is omitted deliberately: a metric's correlation with
    itself is 1 by construction, carries no n worth reading, and the UI can
    fill it without being told. The lower triangle is omitted for the same
    reason -- Spearman is symmetric, and shipping both halves would double
    the file to say each thing twice.
    """
    subset = df if group == "all" else df.filter(pl.col("position") == group)
    cells: list[dict] = []
    for a, b in combinations(metrics, 2):
        x, y = complete_pairs(subset[a], subset[b])
        stats = spearman_with_significance(x, y)
        cells.append(
            {
                "group": group,
                "a": a,
                "b": b,
                "rho": _finite(stats["rho"]),
                "n": int(stats["n"]),
                "p_value": _finite(stats["p_value"]),
            }
        )
    return cells


def correlation_basis(per_fixture: int) -> str:
    """The population string this file claims, in the same register as
    `normalize.normalization_basis` -- and deliberately different from it,
    because it is a different population: player-seasons pooled across
    every archive season, not player-gameweeks within one."""
    return f"player_season_pooled_min{per_fixture}_per_fixture"


def build_correlations(
    panel: pl.DataFrame | None = None,
    panel_path: Path = PANEL_PATH,
    config: dict | None = None,
) -> CorrelationsFile:
    """The whole file: population, every group, every pair.

    `panel` is injectable so the tests can drive this over a constructed
    frame; by default it reads the build artifact, which is the same
    `panel.parquet` §5.3.4 declines to commit.
    """
    if panel is None:
        if not panel_path.exists():
            raise FileNotFoundError(
                f"{panel_path} not found — run `python -m web.export panel` first. "
                "It is a build artifact and §5.3.4 does not commit it."
            )
        panel = pl.read_parquet(panel_path)

    config = config or load_frontend_config()
    per_fixture = config["normalization"]["minutes_per_fixture_floor"]
    min_n_cell = config["correlations"]["min_n_cell"]

    # Only the metrics the panel actually carries. `clean_sheet_prob` and
    # `minutes_reliability` are registered as `source: "model"` and are not
    # computed yet, and a matrix cell for an unwritten model head would be
    # a build gap dressed as a measurement gap -- a distinction §5.3.3
    # exists to keep. The file names the metrics it covers, so the UI reads
    # the matrix it was given rather than the one it assumed.
    metrics = [m for m in MATRIX_METRICS if m in panel.columns]
    absent = [m for m in MATRIX_METRICS if m not in panel.columns]
    if absent:
        logger.info("omitting %d registered metric(s) absent from the panel: %s", len(absent), absent)

    people = player_season_frame(panel, metrics, per_fixture)

    cells: list[dict] = []
    groups: list[GroupSummary] = []
    for group in GROUPS:
        subset = people if group == "all" else people.filter(pl.col("position") == group)
        groups.append(
            GroupSummary(
                key=group,
                n_player_seasons=subset.height,
                mixed_position=(group == "all"),
            )
        )
        cells.extend(correlate_group(people, metrics, group))

    seasons = sorted(panel["season"].unique().to_list())
    latest = panel.filter(pl.col("season") == seasons[-1])

    return CorrelationsFile(
        header=build_header(
            rows=len(cells),
            source_gameweek=int(latest["gw"].max()),
            normalization_basis=correlation_basis(per_fixture),
        ),
        basis=correlation_basis(per_fixture),
        min_n_cell=min_n_cell,
        seasons=seasons,
        metrics=metrics,
        groups=groups,
        cells=[CorrelationCell(**c) for c in cells],
    )
