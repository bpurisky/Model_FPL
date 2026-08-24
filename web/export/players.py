"""`players.json` — one row per element (§5.3.2).

The per-player current state: identity, price, ownership, season-to-date
actuals, every rate metric with its within-position z-score and
percentile, the minutes head's own distribution, and the event model's
per-component projection for the next gameweek.

**It describes the latest gameweek present in the panel**, and follows the
panel forward. Right now that is the most recent completed season; the
moment `data/current_season/` holds a gameweek the panel carries 2026-27
and this switches to it with nobody remembering to. `board.json` resolves
the same way and for the same reason — a surface that needed a manual
re-point on the morning the season starts is a surface that would be
wrong on the morning the season starts.

**Everything here comes from committed data.** The projection is the same
`analytics/projections.py` the scorecard measures, run through the same
walk-forward discipline: trained on gameweeks strictly before the one it
projects, so the numbers on this surface are the numbers whose accuracy
`scorecard.json` publishes. No live API call, and no separate model that
could quietly disagree with the validated one.

Two consequences of that worth stating.

**The projected gameweek is the one after the latest, whether or not it
exists.** For a completed season there is no gameweek 39, so the fixture
difficulty falls back to neutral — `project_points` already does this —
and the projection becomes a fixture-blind statement about the player
rather than a forecast of a specific match. `projection_basis` says which
of the two it is, because reading a fixture-blind number as a fixture
forecast would be reading it wrong.

**The total is the sum of the components, by construction.**
`expected_points_from_projection` is defined as that sum, so this file
computes the components and adds them rather than calling both and hoping
they agree. A total that disagreed with its own parts is exactly the kind
of thing a decomposition panel exists to expose.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from analytics.evaluate import SEASON_SCORING_CONFIG
from analytics.projections import (
    DEFAULT_MINUTES_WINDOW,
    DEFAULT_WINDOW,
    expected_points_by_component,
    project_event_vectors,
)
from analytics.scoring import load_scoring_config
from web.export.columns import MATRIX_METRICS, companion_keys
from web.export.contract import (
    PlayerMetric,
    PlayerProjection,
    PlayerRow,
    PlayersFile,
    build_header,
    json_safe,
)
from web.export.correlations import PANEL_PATH, player_season_frame
from web.export.current import load_current_season
from web.export.normalize import load_frontend_config, normalize
from web.export.panel import HISTORICAL_DIR, available_seasons

logger = logging.getLogger("web.export.players")

# The archive's own mapping, plus the season the collector is now filling.
SCORING_CONFIG = {**SEASON_SCORING_CONFIG, "2026-27": "config/scoring_2026_27.yaml"}

# Season-to-date actuals carried alongside the projection, so a reader can
# see what the player has actually done next to what the model expects.
ACTUAL_TOTALS = [
    "minutes", "total_points", "goals_scored", "assists", "clean_sheets", "bonus",
]


def model_frame(historical_dir: Path = HISTORICAL_DIR) -> pl.DataFrame:
    """The raw rows the model consumes — the same inputs `build_panel`
    starts from, before per-90 derivation.

    Read here rather than taken from the panel because the panel keeps
    rates and drops the raw counts, and `project_event_vectors` trails on
    the counts.
    """
    frames = [
        pl.read_parquet(historical_dir / f"{season}.parquet").with_columns(
            pl.lit(season).alias("season")
        )
        if "season" not in pl.read_parquet_schema(historical_dir / f"{season}.parquet")
        else pl.read_parquet(historical_dir / f"{season}.parquet")
        for season in available_seasons(historical_dir)
    ]
    current = load_current_season()
    if current is not None and current.height:
        frames.append(current)
    return pl.concat(frames, how="diagonal_relaxed")


def difficulty_table(frame: pl.DataFrame) -> pl.DataFrame:
    """FPL's published difficulty, per (team, gw).

    The same source `web/export/panel.py:add_model_columns` uses, and for
    the same two reasons: it measured better than our Elo in the clean
    sheet model, and it is committed, where the Elo table derives from the
    gitignored raw cache. A build here and a build in CI must agree.
    """
    if "opponent_difficulty" not in frame.columns:
        return pl.DataFrame(schema={"team": pl.Utf8, "gw": pl.Int64, "custom_difficulty": pl.Float64})
    return (
        frame.group_by(["team", "gw"])
        .agg(pl.col("opponent_difficulty").mean().cast(pl.Float64).alias("custom_difficulty"))
        .drop_nulls("custom_difficulty")
    )


def latest_gameweek(frame: pl.DataFrame) -> tuple[str, int]:
    season = sorted(frame["season"].unique().to_list())[-1]
    return season, int(frame.filter(pl.col("season") == season)["gw"].max())


def target_roster(frame: pl.DataFrame, season: str, gameweek: int) -> pl.DataFrame:
    """Who to project: everyone who appeared in the latest gameweek.

    `is_promoted_club` rides along because `analytics/features.py`'s pooled
    prior needs it — a newly promoted club's players have no top-flight
    history to trail on, and without the flag they would silently fall
    back to a prior meant for someone else.
    """
    columns = ["element_id", "position", "team"]
    if "is_promoted_club" in frame.columns:
        columns.append("is_promoted_club")
    roster = (
        frame.filter((pl.col("season") == season) & (pl.col("gw") == gameweek))
        .select(columns)
        .unique(subset=["element_id"], keep="first")
    )
    if "is_promoted_club" not in roster.columns:
        roster = roster.with_columns(pl.lit(False).alias("is_promoted_club"))
    return roster


def project(
    frame: pl.DataFrame, season: str, gameweek: int, target_gw: int
) -> tuple[dict[int, PlayerProjection], bool]:
    """Per-component projections and the minutes distribution, for every
    player in the latest gameweek's roster.

    Trained on gameweeks strictly before `target_gw`, which is what makes
    these the same numbers `scorecard.json` publishes accuracy for.
    """
    config = load_scoring_config(Path(SCORING_CONFIG[season]))
    season_rows = frame.filter(pl.col("season") == season)
    train = season_rows.filter(pl.col("gw") < target_gw)
    roster = target_roster(frame, season, gameweek)

    difficulty = difficulty_table(season_rows)
    for_gw = difficulty.filter(pl.col("gw") == target_gw).select("team", "custom_difficulty")
    fixture_known = for_gw.height > 0
    roster = roster.join(for_gw, on="team", how="left").with_columns(
        pl.col("custom_difficulty").fill_null(3.0)
    )

    projected = project_event_vectors(
        train, roster, target_gw, config, DEFAULT_WINDOW, DEFAULT_MINUTES_WINDOW
    )

    out: dict[int, PlayerProjection] = {}
    for row in projected.to_dicts():
        components = expected_points_by_component(row, config)
        out[int(row["element_id"])] = PlayerProjection(
            components={k: json_safe(v) for k, v in components.items()},
            # Defined as the sum of the components rather than computed a
            # second way, so the total cannot disagree with its own parts.
            total=json_safe(sum(components.values())),
            p_blank=json_safe(row.get("p_blank")),
            p_short=json_safe(row.get("p_short")),
            p_full=json_safe(row.get("p_full")),
        )
    return out, fixture_known


def season_to_date(frame: pl.DataFrame, season: str, gameweek: int) -> pl.DataFrame:
    """Actual totals up to and including the latest gameweek.

    Every column is prefixed. The panel carries `minutes` and
    `total_points` of its own — for the single gameweek the row describes
    — so joining unprefixed totals silently keeps the panel's values for
    exactly those two and the season's for everything else. That produced
    a Haaland row reading 0 minutes and 0 points beside 27 goals.
    """
    totals = [c for c in ACTUAL_TOTALS if c in frame.columns]
    return (
        frame.filter((pl.col("season") == season) & (pl.col("gw") <= gameweek))
        .group_by("element_id")
        .agg(
            *[pl.col(c).sum().alias(f"std_{c}") for c in totals],
            pl.len().alias("std_gameweeks"),
        )
    )


def season_rates(panel: pl.DataFrame, season: str, gameweek: int, metrics: list[str]) -> pl.DataFrame:
    """Season-to-date rates for every element, normalized within position.

    The panel's own z-scores are per *gameweek*, which is the wrong grain
    for a player-level file: a player who did not feature in the latest
    gameweek has a null rate for it, so every metric on his card would be
    null even though he has a full season behind him. Haaland at gw38 was
    exactly that.

    The rates are minutes-weighted season figures — the same construction
    `correlations.py` uses, reused rather than restated — and are then
    normalized against the position group with a synthetic gameweek, so
    the peer group is "players at this position, this season", which is
    what a card comparing two players means.
    """
    per_fixture = load_frontend_config()["normalization"]["minutes_per_fixture_floor"]
    people = player_season_frame(panel, metrics, per_fixture, eligible_only=False)
    people = people.filter(pl.col("season") == season).with_columns(
        pl.lit(gameweek, dtype=pl.Int64).alias("gw"),
        pl.col("season_minutes").alias("minutes"),
        pl.col("season_fixtures").alias("n_fixtures"),
    )
    return normalize(people, metrics, per_fixture)


def build_players(
    panel: pl.DataFrame | None = None,
    panel_path: Path = PANEL_PATH,
    historical_dir: Path = HISTORICAL_DIR,
) -> PlayersFile:
    """One row per element in the latest gameweek."""
    if panel is None:
        if not panel_path.exists():
            raise FileNotFoundError(
                f"{panel_path} not found — run `python -m web.export panel` first. "
                "It is a build artifact and §5.3.4 does not commit it."
            )
        panel = pl.read_parquet(panel_path)

    frame = model_frame(historical_dir)
    season, gameweek = latest_gameweek(panel)
    target_gw = gameweek + 1

    projections, fixture_known = project(frame, season, gameweek, target_gw)

    # `model_frame` reads `data/current_season/` itself rather than taking
    # the injected panel, so the two can disagree about which season is
    # latest -- and when they do, `project` filters the model frame to a
    # season it does not carry and returns nothing at all. Every card then
    # renders em dashes for its whole decomposition, which looks exactly
    # like a model that had not spoken yet rather than like a build that
    # read the wrong file.
    #
    # It costs nothing to notice, and §5.3.3's whole argument is that a
    # silent nothing is the expensive kind of wrong.
    if not projections:
        raise ValueError(
            f"no projection for any player at {season} gw{gameweek}. "
            f"The panel's latest season is {season}, and the model frame carries "
            f"{sorted(frame['season'].unique().to_list())}. If those disagree, "
            "`data/current_season/` and `panel.parquet` were built from different data — "
            "rebuild the panel."
        )
    metrics = [m for m in MATRIX_METRICS if m in panel.columns]
    population: dict[str, dict[str, int]] = {}

    identity = panel.filter((pl.col("season") == season) & (pl.col("gw") == gameweek)).select(
        "element_id", "name", "team", "position", "value", "selected"
    )
    latest = (
        identity.join(season_to_date(frame, season, gameweek), on="element_id", how="left")
        .join(season_rates(panel, season, gameweek, metrics), on="element_id", how="left")
    )
    rows: list[PlayerRow] = []
    for row in latest.sort("element_id").iter_rows(named=True):
        element_id = int(row["element_id"])
        carried = {}
        for metric in metrics:
            z_key, pct_key, n_key = companion_keys(metric)
            carried[metric] = PlayerMetric(
                value=json_safe(row.get(metric)),
                z=json_safe(row.get(z_key)),
                percentile=json_safe(row.get(pct_key)),
            )
            if row.get(n_key) is not None:
                population.setdefault(row["position"], {})[metric] = int(row[n_key])
        rows.append(
            PlayerRow(
                element_id=element_id,
                name=row["name"],
                team=row["team"],
                position=row["position"],
                price=int(row["value"]) if row.get("value") is not None else None,
                selected=int(row["selected"]) if row.get("selected") is not None else None,
                gameweeks=int(row.get("std_gameweeks") or 0),
                actuals={
                    c: row.get(f"std_{c}")
                    for c in ACTUAL_TOTALS
                    if row.get(f"std_{c}") is not None
                },
                metrics=carried,
                projection=projections.get(element_id),
            )
        )

    basis = "next_fixture" if fixture_known else "fixture_neutral"
    logger.info(
        "players: %d at %s gw%d, projecting gw%d (%s)",
        len(rows), season, gameweek, target_gw, basis,
    )

    return PlayersFile(
        header=build_header(
            rows=len(rows),
            source_gameweek=gameweek,
            normalization_basis=load_frontend_config()["normalization"]["basis"],
        ),
        season=season,
        gameweek=gameweek,
        projected_gameweek=target_gw,
        projection_basis=basis,
        population=population,
        players=rows,
    )
