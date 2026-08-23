"""The current season, shaped to fit the historical panel (§5.3.2).

`papertrade/actuals.py` writes an append-only per-gameweek store from
`/event/{gw}/live/`. That store is deliberately narrow — it is
`squad/live.py:build_train_df`'s schema, built to feed the model, and the
live endpoint does not serve fixture context — so it carries all seventeen
per-90 source columns and none of the eight context columns the panel
expects.

Concatenating it into the panel unenriched would not raise. `build_panel`
uses `diagonal_relaxed`, which fills the missing columns with null, and
that is the honest value for most of them. **It is not honest for
`n_fixtures`**, and that one is load-bearing in a way nothing announces:
`normalize.with_season_to_date` takes `cum_sum()` of it, `eligible_mask`
compares `cum_minutes >= cum_fixtures * 45`, and a null propagates through
both. Every 2026-27 player would come out ineligible, so the entire
current season would carry null z-scores, null percentiles and null `n`
— a panel that looks complete, builds green, and silently has no
normalized values for the only season anyone is playing.

So this module derives what the store does not carry, from data that is
committed:

`n_fixtures`, `opponent_team`, `was_home`, `kickoff_time` come from
`data/reference/fixtures.parquet`. A double gameweek takes the *first*
fixture's opponent and kickoff, matching what `backtest/backfill.py`
already does for the archive — verified against 2023-24 gw7, where
Burnley's doubled rows carry one opponent and the earlier kickoff.

`value` is the price at that gameweek's deadline, recovered from the
distilled shards. Those are delta-only (§2.3), so the correct price is the
last snapshot at or before the deadline rather than any single row.

`selected` stays **null**, and that is a real gap rather than a choice.
The panel's `selected` is a squad *count* (0–9.5M across the archive);
distilled records `selected_by_percent`. Converting needs
bootstrap-static's `total_players`, which nothing collects yet. Writing a
percentage into a column the registry declares as a count would be a unit
error that every downstream chart would render confidently. Fixing it
properly is a one-field collector change.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger("web.export.current")

CURRENT_SEASON_DIR = Path("data/current_season")
REFERENCE_DIR = Path("data/reference")
DISTILLED_DIR = Path("data/distilled")

# The eight columns `papertrade/actuals.py` cannot serve, in panel order.
DERIVED_CONTEXT = [
    "season", "name", "opponent_team", "was_home", "kickoff_time",
    "n_fixtures", "value", "selected", "opponent_difficulty",
]


def load_current_actuals(current_dir: Path = CURRENT_SEASON_DIR) -> pl.DataFrame | None:
    """The append-only actuals store, or None before it exists.

    None rather than an empty frame: "no gameweek has been recorded yet"
    and "a gameweek was recorded and had no rows" are different facts, and
    only the first is normal.
    """
    if not current_dir.exists():
        return None
    paths = sorted(current_dir.glob("*.parquet"))
    if not paths:
        return None

    frames = []
    for path in paths:
        df = pl.read_parquet(path)
        if not df.height:
            continue
        if "season" not in df.columns:
            # The store is one file per season and names itself; the rows
            # inside do not repeat it.
            df = df.with_columns(pl.lit(path.stem).alias("season"))
        frames.append(df)
    if not frames:
        return None
    return pl.concat(frames, how="diagonal_relaxed")


def fixture_context(reference_dir: Path = REFERENCE_DIR) -> pl.DataFrame:
    """One row per (team name, gw): `n_fixtures` and the first fixture's
    opponent, venue and kickoff.

    Both sides of every fixture are emitted, so a team appears once per
    match it plays — which is what makes the count correct for a double
    and zero-by-absence for a blank.
    """
    fixtures = pl.read_parquet(reference_dir / "fixtures.parquet")
    teams = pl.read_parquet(reference_dir / "teams.parquet").select("id", "name")
    names = dict(zip(teams["id"].to_list(), teams["name"].to_list()))

    sides = []
    for own, other, home in (("team_h", "team_a", True), ("team_a", "team_h", False)):
        sides.append(
            fixtures.select(
                pl.col(own).replace_strict(names, default=None).alias("team"),
                pl.col("event").alias("gw"),
                pl.col(other).replace_strict(names, default=None).alias("opponent_team"),
                pl.lit(home).alias("was_home"),
                pl.col("kickoff_time"),
                # FPL publishes difficulty per side; each team's own is the
                # one for the fixture *it* faces.
                pl.col("team_h_difficulty" if home else "team_a_difficulty")
                .cast(pl.Float64)
                .alias("opponent_difficulty"),
            )
        )

    both = pl.concat(sides).filter(pl.col("gw").is_not_null()).sort(["team", "gw", "kickoff_time"])
    return both.group_by(["team", "gw"], maintain_order=True).agg(
        pl.len().cast(pl.UInt32).alias("n_fixtures"),
        pl.col("opponent_team").first(),
        pl.col("was_home").first(),
        pl.col("kickoff_time").first(),
        # Mean across a double, matching backtest/backfill.py's own
        # aggregation so current-season rows join the archive's scale.
        pl.col("opponent_difficulty").mean(),
    )


def price_at_deadline(
    gws: list[int],
    reference_dir: Path = REFERENCE_DIR,
    distilled_dir: Path = DISTILLED_DIR,
) -> pl.DataFrame:
    """One row per (element_id, gw): `now_cost` as it stood at the
    gameweek's deadline.

    The distilled tier is delta-only by design (§2.3) — a row exists only
    where a value changed — so the price in force at a deadline is the
    most recent snapshot at or before it, not a row with that timestamp.
    An element with no snapshot that early gets null rather than the
    earliest price known, which would be a guess about the past.
    """
    empty = pl.DataFrame(schema={"element_id": pl.Int64, "gw": pl.Int64, "value": pl.Int64})
    shards = sorted(distilled_dir.glob("**/*.parquet")) if distilled_dir.exists() else []
    events_path = reference_dir / "events.parquet"
    if not shards or not events_path.exists():
        return empty

    snapshots = pl.concat(
        [pl.read_parquet(p, columns=["snapshot_ts", "element_id", "now_cost"]) for p in shards],
        how="diagonal_relaxed",
    ).sort("snapshot_ts")
    deadlines = dict(
        zip(
            pl.read_parquet(events_path)["id"].to_list(),
            pl.read_parquet(events_path)["deadline_time"].to_list(),
        )
    )

    frames = []
    for gw in gws:
        deadline = deadlines.get(gw)
        if deadline is None:
            continue
        upto = snapshots.filter(pl.col("snapshot_ts") <= deadline)
        if not upto.height:
            continue
        frames.append(
            upto.group_by("element_id")
            .agg(pl.col("now_cost").last().alias("value"))
            .with_columns(pl.lit(gw, dtype=pl.Int64).alias("gw"))
            .select("element_id", "gw", "value")
        )
    return pl.concat(frames) if frames else empty


def enrich(
    actuals: pl.DataFrame,
    reference_dir: Path = REFERENCE_DIR,
    distilled_dir: Path = DISTILLED_DIR,
) -> pl.DataFrame:
    """Add every panel context column the actuals store cannot carry."""
    players_path = reference_dir / "players.parquet"
    if players_path.exists():
        players = pl.read_parquet(players_path).select(
            pl.col("id").alias("element_id"), pl.col("web_name").alias("name")
        )
        actuals = actuals.join(players, on="element_id", how="left")
    else:  # pragma: no cover - reference is committed
        actuals = actuals.with_columns(pl.lit(None, dtype=pl.Utf8).alias("name"))

    context = fixture_context(reference_dir)
    actuals = actuals.join(context, on=["team", "gw"], how="left")

    prices = price_at_deadline(
        sorted(actuals["gw"].unique().to_list()), reference_dir, distilled_dir
    )
    actuals = actuals.join(prices, on=["element_id", "gw"], how="left")

    # A team with no fixture row for this gameweek did not play one. The
    # store only holds rows for gameweeks that happened, so this is a
    # reference-data gap rather than a blank, and 1 is the safe reading:
    # it keeps the minutes floor meaningful instead of making the player
    # unconditionally eligible, which a 0 would.
    missing = actuals["n_fixtures"].null_count()
    if missing:
        logger.warning("no fixture context for %d row(s); defaulting n_fixtures to 1", missing)
    actuals = actuals.with_columns(pl.col("n_fixtures").fill_null(1).cast(pl.UInt32))

    if "selected" not in actuals.columns:
        # See the module docstring: derivable only with `total_players`,
        # which nothing collects. Null, never a percentage in a count column.
        actuals = actuals.with_columns(pl.lit(None, dtype=pl.Int64).alias("selected"))
    return actuals


def load_current_season(
    current_dir: Path = CURRENT_SEASON_DIR,
    reference_dir: Path = REFERENCE_DIR,
    distilled_dir: Path = DISTILLED_DIR,
) -> pl.DataFrame | None:
    """The store, enriched and ready to concatenate into the panel."""
    actuals = load_current_actuals(current_dir)
    if actuals is None:
        return None
    enriched = enrich(actuals, reference_dir, distilled_dir)
    logger.info(
        "current season: %d row(s) across gw%s",
        enriched.height, sorted(enriched["gw"].unique().to_list()),
    )
    return enriched
