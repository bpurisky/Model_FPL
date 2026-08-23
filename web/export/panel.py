"""`panel.parquet` — the tidy long table (§5.3.2).

One row per player per completed gameweek: raw stats, fantasy points,
fixture context, the sixteen per-90 metrics, and their position-normalized
companions. It is the source for Graph Builder, Form Matrix and Trend
Explorer, and it is deliberately *not* committed (§5.3.4) — size and
churn — so those routes show an explanatory empty state when it is absent
rather than a spinner or a silent blank chart.

The one thing this module does that is easy to get wrong and impossible
to notice afterwards is `apply_availability`. Read its docstring before
changing anything here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from web.export.columns import (
    MATRIX_METRICS,
    PER90_SOURCES,
    by_key,
    companion_keys,
    per90_expr,
)
from web.export.current import load_current_season
from web.export.normalize import normalize

logger = logging.getLogger("web.export.panel")

HISTORICAL_DIR = Path("data/historical")

# Carried through to the export so the UI can render fixture context and
# so §5.7.4's tooltip has a window to name.
CONTEXT_COLUMNS = [
    "season", "gw", "element_id", "name", "team", "position",
    "opponent_team", "was_home", "kickoff_time", "n_fixtures",
    "minutes", "total_points", "value", "selected",
]


def available_seasons(historical_dir: Path = HISTORICAL_DIR) -> list[str]:
    return sorted(p.stem for p in historical_dir.glob("*.parquet") if p.is_file())


def derive_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """Add every per-90 rate the registry knows how to derive.

    Only the metrics whose source column is present: the current-season
    frame will not carry everything the historical archive does, and a
    missing source is a smaller column set rather than a crash.
    """
    present = {k: src for k, src in PER90_SOURCES.items() if src in df.columns}
    missing = set(PER90_SOURCES) - set(present)
    if missing:
        logger.info("skipping %d metric(s) with no source column: %s", len(missing), sorted(missing))
    return df.with_columns([per90_expr(src, key) for key, src in present.items()])


def apply_availability(df: pl.DataFrame) -> pl.DataFrame:
    """Null out any metric in a season where it carries no measurement.

    This is §5.3.3 enforced rather than assumed, and it is the reason
    `available_to_season` exists. Two distinct failure modes:

    A column that starts late (tackles, recoveries, CBI and defensive
    contribution, none of which exist before 2025-26) already arrives
    null from the archive, so this is a no-op for them — but only because
    backtest/backfill.py restores the null after its group_by. If that
    ever regresses to zeros, this is the layer that stops them reaching a
    chart.

    A column that *stops* is the dangerous one. FPL publishes influence,
    creativity, threat and ict_index as literal 0.0 for all 604 elements
    from 2026-27, so nothing upstream is null and nothing raises: a
    perfectly well-formed column of manufactured zeros would flow into
    the panel, into the positional means, and onto a heat map as though
    it were measurement. Zero is a claim; absent is not the same claim.
    """
    registry = by_key()
    exprs = []
    for key in PER90_SOURCES:
        if key not in df.columns:
            continue
        column = registry[key]
        if column.available_from_season is None and column.available_to_season is None:
            continue
        applies = pl.lit(True)
        if column.available_from_season:
            applies = applies & (pl.col("season") >= pl.lit(column.available_from_season))
        if column.available_to_season:
            applies = applies & (pl.col("season") <= pl.lit(column.available_to_season))
        exprs.append(pl.when(applies).then(pl.col(key)).otherwise(None).alias(key))
    return df.with_columns(exprs) if exprs else df


def build_panel(
    seasons: list[str] | None = None,
    historical_dir: Path = HISTORICAL_DIR,
    current: pl.DataFrame | None = None,
    include_current: bool = True,
) -> pl.DataFrame:
    """The full tidy panel across `seasons`, plus the current season.

    `current` is still injectable for tests. When it is not supplied and
    `include_current` is set, the store is loaded and enriched by
    `web/export/current.py` — automatically, because the alternative is a
    manual step nobody remembers on the morning the first gameweek is
    recorded, and a panel silently missing the only season anyone is
    playing looks exactly like a complete one.
    """
    seasons = seasons if seasons is not None else available_seasons(historical_dir)
    frames = [pl.read_parquet(historical_dir / f"{s}.parquet") for s in seasons]
    if current is None and include_current:
        current = load_current_season()
    if current is not None and current.height:
        logger.info("appending %d current-season row(s) to the panel", current.height)
        frames.append(current)
    if not frames:
        raise ValueError("no seasons to build a panel from")

    # `diagonal`, not `vertical`: the current-season store is a strict
    # subset of the historical panel (papertrade/actuals.py's schema omits
    # the fixture context that /event/{gw}/live/ does not serve), so a
    # vertical concat raises the moment a real current season arrives.
    # Diagonal unions the columns and fills the gaps with null, which is
    # also the honest value -- those columns are unknown for those rows,
    # not zero (§5.3.3).
    df = pl.concat(frames, how="diagonal_relaxed")
    df = derive_metrics(df)
    df = apply_availability(df)

    metrics = [k for k in MATRIX_METRICS if k in df.columns]
    df = normalize(df, metrics)

    keep = [c for c in CONTEXT_COLUMNS if c in df.columns]
    keep += ["cum_minutes", "cum_fixtures"]
    for key in metrics:
        keep.append(key)
        keep.extend(companion_keys(key))
    # Registered metrics outside the matrix set (ICT) ride along raw: they
    # are real for three seasons and belong in Graph Builder, they are just
    # not part of the hero matrix.
    keep += [k for k in PER90_SOURCES if k in df.columns and k not in metrics]

    return df.select(keep).sort(["season", "gw", "position", "element_id"])


def write_panel(df: pl.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "panel.parquet"
    df.write_parquet(path, compression="zstd")
    return path
