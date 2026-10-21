"""`projections.json` — a player by gameweek table of expected points.

`players.json` projects one gameweek, the next, and breaks it into
components. Planning a transfer or a hold means looking further than that,
and every other FPL tool shows a run of gameweeks, so this file is the
same model asked the same question for each of the next six (fewer when
the season ends first): one `xp` per player per gameweek, with the
fixtures behind it.

**It is the model `players.json`, the scorecard and the optimizer already
use, not a second one.** `analytics.projections.project_event_vectors`
produces the heads, `expected_points_by_component` turns them into points,
and the scoreline model (`analytics.scoreline`) reads each gameweek's
fixtures. The horizon loop is the one `squad.live.build_projections` runs
for the optimizer. Nothing here is fitted or adjusted on top.

**Blanks and doubles come from the schedule, not from a flag.**
`scoreline_table` counts a club's fixtures in the gameweek, `with_scoreline`
fills a club with none as `n_fixtures` 0, and every component is then
scaled by that count or built from the fixture's own Poisson expectation. A
blank gameweek therefore projects exactly 0.0 and a double sums both
fixtures, which is how the model has always treated them. The per-gameweek
fixture list shipped beside the numbers comes from the reference table the
scoreline model reads, so the two cannot disagree about which gameweek is a
double.

**Leakage (§4.5).** Every gameweek is projected from the same training set,
the gameweeks already played; nothing past `gameweek` is read. Team
strengths are taken as they stood before the first projected gameweek and
held fixed across the horizon, because the results that would update them
have not happened. A later gameweek's number is a forecast made today, not
the one the model would make once the intervening gameweeks are played.

**Minutes.** The minutes head's three-way distribution (blank / short /
60+) is shipped for the first projected gameweek only, and FPL's
availability flag is applied to that gameweek only, since it is FPL's
statement about the next round. The model has no P(start) head and this
file does not invent one: `p_full` is P(60+ minutes), which is what every
points component is gated on.

Committed data only, no live call: the current-season parquet,
`data/reference/`, and the injury flags the collector last snapshotted.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from analytics.carryover import previous_season, prior_history
from analytics.projections import (
    DEFAULT_MINUTES_WINDOW,
    DEFAULT_WINDOW,
    expected_points_by_component,
    project_event_vectors,
    with_scoreline,
)
from analytics.scoreline import scoreline_table, strengths_before
from analytics.scoring import load_scoring_config
from web.export.contract import (
    ProjectionFixture,
    ProjectionPlayer,
    ProjectionsFile,
    ProjectionTeam,
    ProjectionTeamGameweek,
    build_header,
    json_safe,
)
from web.export.correlations import PANEL_PATH
from web.export.normalize import load_frontend_config
from web.export.panel import HISTORICAL_DIR
from web.export.players import (
    REFERENCE_FIXTURES,
    REFERENCE_TEAMS,
    SCORING_CONFIG,
    _availability,
    _season_fixtures,
    latest_gameweek,
    model_frame,
    target_roster,
)

logger = logging.getLogger("web.export.projections")

# How far ahead. Six is what the comparable tools show, and the model has no
# edge to claim further out: the trailing rates barely move and the fixture
# list is the only thing that varies.
HORIZON = 6


def horizon_gameweeks(gameweek: int, last_gameweek: int | None, horizon: int = HORIZON) -> list[int]:
    """The gameweeks after `gameweek`, up to `horizon` of them and never
    past the season's last. Empty when the schedule is unknown."""
    if last_gameweek is None:
        return []
    return list(range(gameweek + 1, min(gameweek + horizon, last_gameweek) + 1))


def project_horizon(
    train: pl.DataFrame,
    roster: pl.DataFrame,
    config: dict,
    fixtures: pl.DataFrame,
    strengths: pl.DataFrame,
    gameweeks: list[int],
    prior: pl.DataFrame | None = None,
    availability: pl.DataFrame | None = None,
) -> dict[int, pl.DataFrame]:
    """gw -> the roster's event vectors with `xp` and the scoreline's
    `n_fixtures`, for each projected gameweek.

    `fixtures` is gw, team_h, team_a (names); `strengths` is
    `analytics.scoreline.team_strengths`' output. `availability` applies to
    the first gameweek only, see the module docstring.
    """
    out: dict[int, pl.DataFrame] = {}
    for i, gw in enumerate(gameweeks):
        with_fixture = with_scoreline(roster, scoreline_table(fixtures, strengths, gw))
        projected = project_event_vectors(
            train, with_fixture, gw, config, DEFAULT_WINDOW, DEFAULT_MINUTES_WINDOW,
            prior_history=prior, availability=availability if i == 0 else None,
        )
        xp = [sum(expected_points_by_component(row, config).values()) for row in projected.to_dicts()]
        # A player the model gives no chance of playing sums components that
        # cancel to +-1e-17, and a column of `-5.55112e-17` reads as a
        # negative projection. Snap float noise to the zero it is.
        xp = [0.0 if abs(x) < 1e-9 else x for x in xp]
        out[gw] = projected.with_columns(pl.Series("xp", xp, dtype=pl.Float64))
    return out


def team_fixtures(
    fixtures: pl.DataFrame, teams: pl.DataFrame, gameweeks: list[int]
) -> list[ProjectionTeam]:
    """Every club's fixtures in each projected gameweek, blanks as empty
    lists and doubles as two entries.

    `fixtures` is the reference table (integer team ids, FPL's own
    difficulty), `teams` the id -> name / short_name table.
    """
    names = {r["id"]: r for r in teams.to_dicts()}
    wanted = fixtures.filter(pl.col("event").is_in(gameweeks))
    by_team: dict[tuple[int, int], list[ProjectionFixture]] = {}
    for f in wanted.sort("event", "kickoff_time").to_dicts():
        for team, opponent, home, difficulty in (
            (f["team_h"], f["team_a"], True, f["team_h_difficulty"]),
            (f["team_a"], f["team_h"], False, f["team_a_difficulty"]),
        ):
            by_team.setdefault((team, int(f["event"])), []).append(
                ProjectionFixture(
                    opponent=names[opponent]["short_name"],
                    home=home,
                    difficulty=int(difficulty) if difficulty is not None else None,
                )
            )
    return [
        ProjectionTeam(
            team=row["name"],
            short_name=row["short_name"],
            gameweeks=[
                ProjectionTeamGameweek(gw=gw, fixtures=by_team.get((team_id, gw), []))
                for gw in gameweeks
            ],
        )
        for team_id, row in sorted(names.items(), key=lambda kv: kv[1]["name"])
    ]


def build_projections(
    panel: pl.DataFrame | None = None,
    panel_path: Path = PANEL_PATH,
    historical_dir: Path = HISTORICAL_DIR,
    as_of: datetime | None = None,
) -> ProjectionsFile:
    """One row per element in the latest gameweek's roster, `xp` aligned
    with `gameweeks`. `as_of` is when the injury flags are read, and the
    header's `generated_at`."""
    as_of = as_of or datetime.now(timezone.utc)
    if panel is None:
        if not panel_path.exists():
            raise FileNotFoundError(
                f"{panel_path} not found — run `python -m web.export panel` first. "
                "It is a build artifact and §5.3.4 does not commit it."
            )
        panel = pl.read_parquet(panel_path)

    frame = model_frame(historical_dir)
    season, gameweek = latest_gameweek(panel)
    if season not in set(frame["season"].unique().to_list()):
        raise ValueError(
            f"the panel's latest season {season} is not in the model frame "
            f"{sorted(frame['season'].unique().to_list())}; rebuild the panel."
        )

    season_rows = frame.filter(pl.col("season") == season)
    # Only what has happened. The roster is likewise the latest played gameweek's.
    train = season_rows.filter(pl.col("gw") <= gameweek)
    roster = target_roster(frame, season, gameweek)

    fixtures = _season_fixtures(season)
    gameweeks = horizon_gameweeks(
        gameweek, int(fixtures["gw"].max()) if fixtures is not None and fixtures.height else None
    )

    projected: dict[int, pl.DataFrame] = {}
    teams: list[ProjectionTeam] = []
    if gameweeks:
        prev = HISTORICAL_DIR / f"{previous_season(season)}.parquet"
        strengths = strengths_before(
            train, fixtures, pl.read_parquet(prev) if prev.exists() else None, gameweek + 1
        )
        projected = project_horizon(
            train, roster, load_scoring_config(Path(SCORING_CONFIG[season])), fixtures, strengths,
            gameweeks, prior=prior_history(season, roster), availability=_availability(season, as_of),
        )
        teams = team_fixtures(pl.read_parquet(REFERENCE_FIXTURES), pl.read_parquet(REFERENCE_TEAMS), gameweeks)
        # A roster club the teams table lacks would render as a silent
        # blank in every gameweek, so that is the direction worth failing on.
        missing = set(roster["team"].to_list()) - {t.team for t in teams}
        if missing:
            raise ValueError(f"clubs in the roster but not in the reference teams table: {sorted(missing)}")

    identity = panel.filter((pl.col("season") == season) & (pl.col("gw") == gameweek)).select(
        "element_id", "name", "team", "position", "value"
    )
    first = projected[gameweeks[0]] if gameweeks else None
    minutes = (
        {int(r["element_id"]): r for r in first.select("element_id", "p_blank", "p_short", "p_full").to_dicts()}
        if first is not None
        else {}
    )
    xp_by_gw = {gw: dict(zip(df["element_id"].to_list(), df["xp"].to_list())) for gw, df in projected.items()}

    players = []
    for row in identity.sort("element_id").iter_rows(named=True):
        element_id = int(row["element_id"])
        m = minutes.get(element_id, {})
        players.append(
            ProjectionPlayer(
                element_id=element_id,
                name=row["name"],
                team=row["team"],
                position=row["position"],
                price=int(row["value"]) if row.get("value") is not None else None,
                xp=[json_safe(xp_by_gw[gw].get(element_id)) for gw in gameweeks],
                p_blank=json_safe(m.get("p_blank")),
                p_short=json_safe(m.get("p_short")),
                p_full=json_safe(m.get("p_full")),
            )
        )

    logger.info(
        "projections: %d players over gw%s at %s gw%d",
        len(players), f"{gameweeks[0]}-{gameweeks[-1]}" if gameweeks else "(none)", season, gameweek,
    )
    return ProjectionsFile(
        header=build_header(
            rows=len(players),
            source_gameweek=gameweek,
            normalization_basis=load_frontend_config()["normalization"]["basis"],
            generated_at=as_of,
        ),
        season=season,
        gameweek=gameweek,
        gameweeks=gameweeks,
        teams=teams,
        players=players,
    )
