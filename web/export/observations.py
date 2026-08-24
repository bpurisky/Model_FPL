"""`observations.json` — the player-seasons the matrix is computed over.

**Deviation from §5.3.2's nine files, recorded per §5.16.** This is a
tenth export, and it exists because §5.6.1 does.

That section permits one client-side statistic: "Arbitrary user-defined
filters (e.g. 'defenders over £6.0m with 400+ minutes') cannot be
precomputed. Where such a filter must produce a fresh correlation, a
client-side implementation is permitted." Choosing which seasons to
correlate across is exactly such a filter, and it cannot be served by
precomputation because **Spearman does not compose**: rho over 2023-24 and
rho over 2024-25 cannot be combined into rho over both. Answering the
question from precomputed matrices means shipping one per subset, which is
2^n — seven matrices for three seasons, thirty-one for five, and a
thousand for ten. The observations are 649 rows and answer every subset.

So the browser gets the values and `src/data/spearman.ts` computes the
correlation, under all three of §5.6.1's conditions: it is a deliberate
port of `backtest/report.py`'s method, `golden_spearman.json` exists to
check it against, and CI fails on any disagreement beyond 1e-9.

**A separate file rather than a block inside `correlations.json`.** That
file is the hero's first paint and §5.9 budgets time-to-interactive at
2.5s on 4G; these rows would roughly quadruple it to serve a feature most
loads never touch. Loaded lazily, on the first season change.

The population is the same one `correlations.py` correlates: eligible
player-seasons, minutes-weighted season rates, the §5.15 Q5 floor. It has
to be, or the client-side matrix for "all seasons" would disagree with
the precomputed one sitting beside it.

**The current season appears here as soon as it has a gameweek**, because
`build_panel` loads `data/current_season/` and this reads the panel. It
arrives with two gameweeks behind it while the archive seasons have
thirty-eight, and a rate over two matches is mostly noise — so every
season carries its own `gameweeks` and `partial`, and the selector says
so rather than offering four seasons that look alike and are not.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from web.export.columns import MATRIX_METRICS
from web.export.contract import (
    ObservationRow,
    ObservationsFile,
    SeasonSummary,
    build_header,
    json_safe,
)
from web.export.correlations import PANEL_PATH, correlation_basis, player_season_frame
from web.export.normalize import load_frontend_config

logger = logging.getLogger("web.export.observations")

# A Premier League season. Anything short of it is still being played, and
# the selector marks it rather than letting it pass as a peer of a
# completed one.
FULL_SEASON_GAMEWEEKS = 38


def identity(panel: pl.DataFrame) -> pl.DataFrame:
    """Name and club per (season, element_id), taken from the player's last
    gameweek of that season.

    Last rather than first because a player transferred mid-season should
    read as the club he ended it at, which is the one his season rates
    were mostly earned for.
    """
    return (
        panel.sort(["season", "element_id", "gw"])
        .group_by(["season", "element_id"])
        .agg(pl.col("name").last(), pl.col("team").last())
    )


def build_observations(
    panel: pl.DataFrame | None = None,
    panel_path: Path = PANEL_PATH,
    config: dict | None = None,
) -> ObservationsFile:
    """One row per eligible player-season, with every matrix metric."""
    if panel is None:
        if not panel_path.exists():
            raise FileNotFoundError(
                f"{panel_path} not found — run `python -m web.export panel` first. "
                "It is a build artifact and §5.3.4 does not commit it."
            )
        panel = pl.read_parquet(panel_path)

    config = config or load_frontend_config()
    per_fixture = config["normalization"]["minutes_per_fixture_floor"]
    metrics = [metric for metric in MATRIX_METRICS if metric in panel.columns]

    people = player_season_frame(panel, metrics, per_fixture).join(
        identity(panel), on=["season", "element_id"], how="left"
    )

    rows = [
        ObservationRow(
            season=row["season"],
            element_id=int(row["element_id"]),
            name=row["name"],
            team=row["team"],
            position=row["position"],
            # Positional, aligned to `metrics` — the same convention
            # `GoldenSample.rows` uses, so a reader meets it once.
            values=[json_safe(row[metric]) for metric in metrics],
        )
        for row in people.sort(["season", "element_id"]).iter_rows(named=True)
    ]

    # Coverage per season, from the panel rather than from the rows above:
    # a season's gameweek count is a fact about the season, not about who
    # happened to clear the minutes floor in it.
    covered = (
        panel.group_by("season")
        .agg(pl.col("gw").max().alias("gameweeks"))
        .sort("season")
    )
    counts = {row.season: 0 for row in rows}
    for row in rows:
        counts[row.season] += 1

    seasons = [
        SeasonSummary(
            season=entry["season"],
            gameweeks=int(entry["gameweeks"]),
            players=counts.get(entry["season"], 0),
            # A season still being played is not comparable with a
            # finished one, and the difference is invisible in a rho.
            partial=int(entry["gameweeks"]) < FULL_SEASON_GAMEWEEKS,
        )
        for entry in covered.iter_rows(named=True)
        if counts.get(entry["season"], 0) > 0
    ]

    logger.info(
        "observations: %d player-seasons over %d metrics (%s)",
        len(rows),
        len(metrics),
        ", ".join(
            f"{s.season} gw{s.gameweeks} n={s.players}{' partial' if s.partial else ''}"
            for s in seasons
        ),
    )

    return ObservationsFile(
        header=build_header(
            rows=len(rows),
            source_gameweek=None,  # a season-grain population, not a gameweek
            normalization_basis=correlation_basis(per_fixture),
        ),
        basis=correlation_basis(per_fixture),
        seasons=seasons,
        metrics=metrics,
        rows=rows,
    )
