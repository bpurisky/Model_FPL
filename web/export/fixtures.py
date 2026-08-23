"""`fixtures.json` — our Elo difficulty beside FPL's static rating (§5.3.2).

§4.3's instruction is to "report both ... so the difference is visible",
and `analytics/fdr.py:build_fdr_comparison` was written for exactly this
join and has never been called outside its own test. This module is what
calls it.

The contrast is the product. FPL's rating is published once and fixed for
the season: it is available in gameweek 1, when nothing else is, and it
never learns. Elo moves with results: it knows nothing in August and a
great deal by March. Showing them side by side lets a reader see when the
static rating has gone stale — which is the only question either number
can settle on its own.

Three things here are easy to get wrong and two of them are silent.

**FPL reassigns team ids every season.** Verified against the archive:
id 3 is Bournemouth in 2023-24 and again in 2026-27, but Burnley in
2025-26. Elo chained across seasons on the raw id would hand one club's
rating to another with nothing raising, so every match is resolved to a
club *name* first and rated against a synthetic id that is stable across
seasons. Names are what FPL keeps constant; ids are not.

**Fixture ids collide across seasons.** Both 2023-24 and 2026-27 have a
fixture 1, and `build_fdr_comparison` joins on that column — so seeding
Elo from earlier seasons in the same pass would silently join the wrong
FPL difficulty onto the wrong match. Historical fixtures therefore carry
*negative* synthetic ids, leaving the positive space to the current
season alone, and the comparison runs over the positive rows only.

**`finished` is not "has this been played".** It flips only after FPL
confirms the gameweek's data, which lags full time by many hours — as of
this build, all 380 fixtures report `finished: false` while eight have
been played and scored. `collector.schemas.fixture_is_played` is the
predicate that already encodes this (it exists because reading `finished`
alone produced the degenerate gw2 freeze), so Elo learns from it rather
than from `finished`.

An unplayed fixture has no pre-match Elo to report, because Elo did not
exist for a match that has not happened. Those rows carry the rating each
club holds *now*, and say so in `difficulty_basis`, because "what the
model knew going in" and "what the model thinks today" are different
claims and a planning surface must not blur them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from analytics.fdr import EloRatings, build_fdr_comparison, compute_elo_ratings, custom_difficulty
from backtest.backfill import RAW_CACHE_DIR, load_match_results, load_teams
from collector.schemas import fixture_is_played
from web.export.contract import FixtureRow, FixturesFile, build_header, json_safe

logger = logging.getLogger("web.export.fixtures")

CURRENT_SEASON = "2026-27"
REFERENCE_DIR = Path("data/reference")

# Historical fixture ids are pushed into the negative space, one block per
# season, so they cannot collide with each other or with the current
# season's real ids. Only the sign and the separation matter.
_SEASON_BLOCK = 10_000

_MATCH_SCHEMA = {
    "fixture": pl.Int64,
    "event": pl.Int64,
    "team_h": pl.Int64,
    "team_a": pl.Int64,
    "team_h_score": pl.Int64,
    "team_a_score": pl.Int64,
    "kickoff_time": pl.Datetime(time_unit="us", time_zone="UTC"),
}


class _StableTeams:
    """Club name -> an id that means the same club in every season.

    FPL's own ids are reassigned as clubs come up and go down, so they are
    only safe within one season. Assigned on first sight; the values are
    arbitrary and never leave this module.
    """

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def id_for(self, name: str) -> int:
        return self._ids.setdefault(name, len(self._ids) + 1)

    def map_frame(self, df: pl.DataFrame, teams: pl.DataFrame) -> pl.DataFrame:
        """Resolve `team_h`/`team_a` from season-local ids to stable ones."""
        lookup = dict(zip(teams["id"].to_list(), teams["name"].to_list()))
        for column in ("team_h", "team_a"):
            df = df.with_columns(
                pl.col(column)
                .map_elements(lambda i: self.id_for(lookup[i]), return_dtype=pl.Int64)
                .alias(column)
            )
        return df


def historical_matches(teams: _StableTeams, raw_dir: Path = RAW_CACHE_DIR) -> pl.DataFrame:
    """Every completed match from the archive seasons, in stable-id space.

    `data/historical/raw/` is gitignored (it is vaastav's cache, restored
    by `python -m backtest backfill`), so this returns empty in a fresh
    clone rather than failing. The file records how many matches actually
    fed Elo, so a reader can tell an unseeded rating from a seeded one
    instead of having to trust that it was seeded.
    """
    if not raw_dir.exists():
        return pl.DataFrame(schema=_MATCH_SCHEMA)

    frames = []
    for index, season_dir in enumerate(sorted(p for p in raw_dir.iterdir() if p.is_dir())):
        fixtures_csv, teams_csv = season_dir / "fixtures.csv", season_dir / "teams.csv"
        if not (fixtures_csv.exists() and teams_csv.exists()):
            continue
        matches = load_match_results(fixtures_csv)
        matches = teams.map_frame(matches, load_teams(teams_csv))
        frames.append(
            matches.with_columns(
                (-(index + 1) * _SEASON_BLOCK - pl.col("fixture")).alias("fixture")
            ).select(list(_MATCH_SCHEMA)).cast(_MATCH_SCHEMA)
        )
    return pl.concat(frames) if frames else pl.DataFrame(schema=_MATCH_SCHEMA)


def seeded_seasons(raw_dir: Path = RAW_CACHE_DIR) -> list[str]:
    if not raw_dir.exists():
        return []
    return sorted(p.name for p in raw_dir.iterdir() if p.is_dir() and (p / "fixtures.csv").exists())


def archive_team_names(raw_dir: Path = RAW_CACHE_DIR) -> set[str]:
    """Every club the archive seasons contain.

    The test for "is this club's rating measured or assumed" has to be
    made against the *archive*, not against `EloRatings.final`. A promoted
    club that has played one match this season already has an entry there
    — Coventry, Hull and Ipswich each got one from gameweek 1 — so
    membership would report them as rated when their number is still the
    initial league-average value nudged by a single result.
    """
    if not raw_dir.exists():
        return set()
    names: set[str] = set()
    for season_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        teams_csv = season_dir / "teams.csv"
        if teams_csv.exists():
            names |= set(load_teams(teams_csv)["name"].to_list())
    return names


def current_fixtures(reference_dir: Path = REFERENCE_DIR) -> pl.DataFrame:
    return pl.read_parquet(reference_dir / "fixtures.parquet")


def current_teams(reference_dir: Path = REFERENCE_DIR) -> pl.DataFrame:
    return pl.read_parquet(reference_dir / "teams.parquet").select("id", "name")


def played_mask(fixtures: pl.DataFrame) -> pl.Series:
    """`fixture_is_played`, applied row by row.

    Reusing the collector's predicate rather than restating `finished or
    finished_provisional` here: that rule already cost this project a
    gameweek once, and a second copy of it is a second place for it to be
    wrong.
    """
    return pl.Series(
        [fixture_is_played(row) for row in fixtures.iter_rows(named=True)], dtype=pl.Boolean
    )


def current_matches(teams: _StableTeams, fixtures: pl.DataFrame, team_names: pl.DataFrame) -> pl.DataFrame:
    """Completed current-season matches, keeping their real fixture ids."""
    played = fixtures.filter(
        played_mask(fixtures)
        & pl.col("team_h_score").is_not_null()
        & pl.col("team_a_score").is_not_null()
    )
    if played.height == 0:
        return pl.DataFrame(schema=_MATCH_SCHEMA)
    played = played.rename({"id": "fixture"})
    return teams.map_frame(played, team_names).select(list(_MATCH_SCHEMA)).cast(_MATCH_SCHEMA)


def build_fixtures(
    reference_dir: Path = REFERENCE_DIR,
    raw_dir: Path = RAW_CACHE_DIR,
    season: str = CURRENT_SEASON,
) -> FixturesFile:
    """One row per current-season fixture, both difficulties on each."""
    stable = _StableTeams()
    fixtures = current_fixtures(reference_dir)
    team_names = current_teams(reference_dir)

    history = historical_matches(stable, raw_dir)
    current = current_matches(stable, fixtures, team_names)
    matches = pl.concat([history, current]) if history.height else current

    elo = compute_elo_ratings(matches) if matches.height else EloRatings(
        per_fixture=pl.DataFrame(
            schema={
                "fixture": pl.Int64, "event": pl.Int64, "team_h": pl.Int64,
                "team_a": pl.Int64, "elo_home_pre": pl.Float64, "elo_away_pre": pl.Float64,
            }
        ),
        final={},
    )

    # The wiring §4.3 asked for: our per-fixture Elo difficulty joined to
    # FPL's published rating, on `fixture`. Restricted to the positive ids
    # so a historical fixture cannot collide with a current-season one.
    fpl_difficulty = fixtures.select(
        pl.col("id").alias("fixture"), "team_h_difficulty", "team_a_difficulty"
    )
    played_elo = elo.per_fixture.filter(pl.col("fixture") > 0)
    comparison = build_fdr_comparison(
        EloRatings(per_fixture=played_elo, final=elo.final), fpl_difficulty
    )
    pre_match = {
        row["fixture"]: (row["custom_difficulty_home"], row["custom_difficulty_away"])
        for row in comparison.iter_rows(named=True)
    }

    names = dict(zip(team_names["id"].to_list(), team_names["name"].to_list()))
    rows: list[FixtureRow] = []
    for row in fixtures.sort(["event", "kickoff_time", "id"]).iter_rows(named=True):
        home_name, away_name = names[row["team_h"]], names[row["team_a"]]
        if row["id"] in pre_match:
            home, away = pre_match[row["id"]]
            basis = "pre_match"
        else:
            # No pre-match rating exists for a match that has not happened.
            # Today's Elo is the honest stand-in, labelled as such.
            elo_h = elo.final.get(stable.id_for(home_name), 1500.0)
            elo_a = elo.final.get(stable.id_for(away_name), 1500.0)
            home = custom_difficulty(elo_h, elo_a, is_home=True)
            away = custom_difficulty(elo_a, elo_h, is_home=False)
            basis = "current_elo"

        rows.append(
            FixtureRow(
                fixture=int(row["id"]),
                gw=int(row["event"]) if row["event"] is not None else None,
                team_h=home_name,
                team_a=away_name,
                kickoff_time=row["kickoff_time"],
                played=row["id"] in pre_match,
                team_h_difficulty=row["team_h_difficulty"],
                team_a_difficulty=row["team_a_difficulty"],
                custom_difficulty_home=json_safe(home),
                custom_difficulty_away=json_safe(away),
                difficulty_basis=basis,
            )
        )

    archive = archive_team_names(raw_dir)
    unseeded = sorted(set(team_names["name"].to_list()) - archive)

    seeds = seeded_seasons(raw_dir)
    logger.info(
        "elo over %d matches (%d seeded from %s, %d from %s); %d club(s) unseeded: %s",
        matches.height, history.height, seeds or "no archive", current.height, season,
        len(unseeded), unseeded or "none",
    )

    return FixturesFile(
        header=build_header(
            rows=len(rows),
            source_gameweek=int(fixtures["event"].max()) if fixtures.height else None,
            normalization_basis="elo_pre_match_with_carry_forward",
        ),
        season=season,
        elo_matches=matches.height,
        elo_seeded_from=seeds,
        unseeded_teams=unseeded,
        fixtures=rows,
    )
