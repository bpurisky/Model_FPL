"""§5.11.1 over `fixtures.json`, concentrated on the three ways this file
could ship a confident wrong difficulty without anything raising: a team id
that means a different club, a fixture id that collides across seasons, and
a played match read as unplayed."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from analytics.fdr import INITIAL_ELO
from web.export.fixtures import (
    _StableTeams,
    archive_team_names,
    build_fixtures,
    current_matches,
    historical_matches,
    played_mask,
)

FIXTURE_COLUMNS = [
    "id", "event", "team_h", "team_a", "kickoff_time", "finished",
    "finished_provisional", "started", "team_h_score", "team_a_score",
    "team_h_difficulty", "team_a_difficulty",
]


def _fixtures(rows: list[dict]) -> pl.DataFrame:
    for row in rows:
        unknown = set(row) - set(FIXTURE_COLUMNS)
        assert not unknown, f"unknown key(s) in test row: {unknown}"
    return pl.DataFrame(
        {c: [row.get(c) for row in rows] for c in FIXTURE_COLUMNS},
        schema={
            "id": pl.Int64, "event": pl.Int64, "team_h": pl.Int64, "team_a": pl.Int64,
            "kickoff_time": pl.Datetime(time_unit="us", time_zone="UTC"),
            "finished": pl.Boolean, "finished_provisional": pl.Boolean,
            "started": pl.Boolean, "team_h_score": pl.Int64, "team_a_score": pl.Int64,
            "team_h_difficulty": pl.Int64, "team_a_difficulty": pl.Int64,
        },
    )


def _fixture(id, h, a, *, gw=1, hs=None, as_=None, finished=False, provisional=False, hd=3, ad=3, day=1):
    return {
        "id": id, "event": gw, "team_h": h, "team_a": a,
        "kickoff_time": datetime(2026, 8, day, 14, 0, tzinfo=timezone.utc),
        "finished": finished, "finished_provisional": provisional, "started": True,
        "team_h_score": hs, "team_a_score": as_,
        "team_h_difficulty": hd, "team_a_difficulty": ad,
    }


def _teams(names: dict[int, str]) -> pl.DataFrame:
    return pl.DataFrame(
        {"id": list(names), "name": list(names.values())},
        schema={"id": pl.Int64, "name": pl.Utf8},
    )


# --- team ids are not stable across seasons ------------------------------


def test_the_same_id_in_two_seasons_resolves_to_two_different_clubs():
    """The silent one. FPL reassigns ids as clubs come up and go down —
    id 3 is Bournemouth in 2023-24 and Burnley in 2025-26 — so Elo chained
    on the raw id would hand one club's rating to another and nothing
    would raise."""
    stable = _StableTeams()

    old = stable.map_frame(
        _fixtures([_fixture(1, 3, 1, hs=1, as_=0)]).rename({"id": "fixture"}),
        _teams({3: "Burnley", 1: "Arsenal"}),
    )
    new = stable.map_frame(
        _fixtures([_fixture(1, 3, 1, hs=1, as_=0)]).rename({"id": "fixture"}),
        _teams({3: "Bournemouth", 1: "Arsenal"}),
    )

    assert old["team_h"][0] != new["team_h"][0], "Burnley and Bournemouth must not share a rating"
    assert old["team_a"][0] == new["team_a"][0], "Arsenal is Arsenal in both"


def test_the_same_club_under_different_ids_keeps_one_rating():
    stable = _StableTeams()

    first = stable.map_frame(
        _fixtures([_fixture(1, 3, 1)]).rename({"id": "fixture"}),
        _teams({3: "Bournemouth", 1: "Arsenal"}),
    )
    later = stable.map_frame(
        _fixtures([_fixture(1, 9, 1)]).rename({"id": "fixture"}),
        _teams({9: "Bournemouth", 1: "Arsenal"}),
    )

    assert first["team_h"][0] == later["team_h"][0]


# --- fixture ids collide across seasons ----------------------------------


def test_historical_fixture_ids_cannot_collide_with_the_current_season(tmp_path):
    """`build_fdr_comparison` joins on `fixture`, and every season has a
    fixture 1. If a seeded historical row kept its own id, the join would
    attach the wrong FPL difficulty to the wrong match — a wrong number
    that looks entirely well-formed."""
    for season in ("2024-25", "2025-26"):
        season_dir = tmp_path / season
        season_dir.mkdir()
        _fixtures([_fixture(1, 1, 2, hs=1, as_=0, finished=True)]).rename(
            {"id": "id"}
        ).write_csv(season_dir / "fixtures.csv")
        _teams({1: "Arsenal", 2: "Aston Villa"}).write_csv(season_dir / "teams.csv")

    history = historical_matches(_StableTeams(), tmp_path)

    assert history.height == 2
    assert all(f < 0 for f in history["fixture"].to_list()), "historical ids must be negative"
    assert history["fixture"].n_unique() == 2, "and distinct from each other"


# --- 'finished' is not 'played' -------------------------------------------


def test_a_provisionally_finished_match_counts_as_played():
    """As of this build all 380 fixtures report `finished: false` while
    eight have been played and scored: `finished` lags full time by many
    hours. Reading it alone is what produced the degenerate gw2 freeze."""
    fixtures = _fixtures([
        _fixture(1, 1, 2, hs=3, as_=0, finished=False, provisional=True),
        _fixture(2, 3, 4, finished=False, provisional=False),
    ])

    assert played_mask(fixtures).to_list() == [True, False]


def test_only_played_matches_reach_elo():
    fixtures = _fixtures([
        _fixture(1, 1, 2, hs=3, as_=0, provisional=True),
        _fixture(2, 1, 2, provisional=False),
    ])

    matches = current_matches(_StableTeams(), fixtures, _teams({1: "Arsenal", 2: "Aston Villa"}))

    assert matches.height == 1
    assert matches["fixture"].to_list() == [1]


def test_a_played_match_with_no_score_is_excluded():
    """`fixture_is_played` can be true while the scores are still null;
    Elo cannot rate a result it does not have."""
    fixtures = _fixtures([_fixture(1, 1, 2, hs=None, as_=None, provisional=True)])

    matches = current_matches(_StableTeams(), fixtures, _teams({1: "Arsenal", 2: "Aston Villa"}))

    assert matches.height == 0


# --- the two bases are different claims ----------------------------------


def test_played_and_upcoming_fixtures_declare_different_bases(tmp_path):
    """A played fixture reports the rating each club carried into it; an
    upcoming one reports what they hold today. Blurring the two would let
    a planning surface present hindsight as foresight."""
    reference = tmp_path / "reference"
    reference.mkdir()
    _fixtures([
        _fixture(1, 1, 2, hs=3, as_=0, provisional=True, gw=1, day=1),
        _fixture(2, 2, 1, gw=2, day=8),
    ]).write_parquet(reference / "fixtures.parquet")
    _teams({1: "Arsenal", 2: "Aston Villa"}).write_parquet(reference / "teams.parquet")

    file = build_fixtures(reference_dir=reference, raw_dir=tmp_path / "absent")

    by_id = {f.fixture: f for f in file.fixtures}
    assert by_id[1].played is True
    assert by_id[1].difficulty_basis == "pre_match"
    assert by_id[2].played is False
    assert by_id[2].difficulty_basis == "current_elo"


def test_an_unplayed_fixture_reflects_the_result_that_came_before_it(tmp_path):
    """The point of a dynamic rating: Arsenal beating Villa 3-0 must make
    Arsenal's next fixture easier for them than the neutral 3.0 that FPL's
    static rating would still be showing."""
    reference = tmp_path / "reference"
    reference.mkdir()
    _fixtures([
        _fixture(1, 1, 2, hs=3, as_=0, provisional=True, gw=1, day=1),
        _fixture(2, 1, 2, gw=2, day=8),
    ]).write_parquet(reference / "fixtures.parquet")
    _teams({1: "Arsenal", 2: "Aston Villa"}).write_parquet(reference / "teams.parquet")

    file = build_fixtures(reference_dir=reference, raw_dir=tmp_path / "absent")
    upcoming = [f for f in file.fixtures if not f.played][0]
    played = [f for f in file.fixtures if f.played][0]

    assert upcoming.custom_difficulty_home < played.custom_difficulty_home


# --- provenance -----------------------------------------------------------


def test_a_missing_archive_yields_an_empty_seed_rather_than_a_failure(tmp_path):
    """`data/historical/raw/` is a restorable cache, not committed data.
    A fresh clone must still build the file — and must say that its Elo
    had nothing to learn from."""
    reference = tmp_path / "reference"
    reference.mkdir()
    _fixtures([_fixture(1, 1, 2, gw=1)]).write_parquet(reference / "fixtures.parquet")
    _teams({1: "Arsenal", 2: "Aston Villa"}).write_parquet(reference / "teams.parquet")

    file = build_fixtures(reference_dir=reference, raw_dir=tmp_path / "absent")

    assert file.elo_seeded_from == []
    assert file.elo_matches == 0
    assert archive_team_names(tmp_path / "absent") == set()
    # With nothing rated, both sides sit at the initial rating, so the only
    # difference between them is home advantage.
    assert file.fixtures[0].custom_difficulty_home < file.fixtures[0].custom_difficulty_away


def test_clubs_absent_from_the_archive_are_named(tmp_path):
    """A promoted club that has played once already has an Elo entry, so
    membership in the ratings dict would report it as rated when its
    number is still the league mean nudged by a single result. The check
    has to be made against the archive."""
    reference = tmp_path / "reference"
    reference.mkdir()
    _fixtures([_fixture(1, 1, 2, hs=1, as_=0, provisional=True)]).write_parquet(
        reference / "fixtures.parquet"
    )
    _teams({1: "Arsenal", 2: "Coventry City"}).write_parquet(reference / "teams.parquet")

    archive = tmp_path / "raw"
    (archive / "2025-26").mkdir(parents=True)
    _fixtures([_fixture(1, 1, 3, hs=1, as_=1, finished=True)]).write_csv(
        archive / "2025-26" / "fixtures.csv"
    )
    _teams({1: "Arsenal", 3: "Burnley"}).write_csv(archive / "2025-26" / "teams.csv")

    file = build_fixtures(reference_dir=reference, raw_dir=archive)

    assert file.unseeded_teams == ["Coventry City"]


# --- the committed file ---------------------------------------------------


def test_the_committed_fixtures_file_is_complete_and_serializable():
    path = Path("data/web/v1/fixtures.json")
    if not path.exists():  # pragma: no cover
        pytest.skip("fixtures.json not generated yet -- run `python -m web.export fixtures`")

    d = json.loads(path.read_text(encoding="utf-8"))

    assert len(d["fixtures"]) == d["header"]["rows"]
    assert {f["difficulty_basis"] for f in d["fixtures"]} <= {"pre_match", "current_elo"}
    for f in d["fixtures"]:
        assert (f["difficulty_basis"] == "pre_match") == f["played"]
        for field in ("custom_difficulty_home", "custom_difficulty_away"):
            assert f[field] is None or math.isfinite(f[field])
            # Both ratings share the 1-5 axis; ours is continuous, FPL's
            # is an integer, and a value outside the range would mean the
            # two could not be plotted together.
            assert f[field] is None or 1.0 <= f[field] <= 5.0


def test_the_committed_fixtures_carry_both_ratings():
    """§4.3's whole instruction: report both so the difference is visible.
    A file with only one of them would satisfy nothing."""
    path = Path("data/web/v1/fixtures.json")
    if not path.exists():  # pragma: no cover
        pytest.skip("fixtures.json not generated yet")

    d = json.loads(path.read_text(encoding="utf-8"))
    both = [
        f for f in d["fixtures"]
        if f["team_h_difficulty"] is not None and f["custom_difficulty_home"] is not None
    ]

    assert len(both) == len(d["fixtures"])
    # And they must not be the same number wearing two labels — if our Elo
    # merely reproduced FPL's rating there would be no reason to compute it.
    assert any(abs(f["custom_difficulty_home"] - f["team_h_difficulty"]) > 0.5 for f in both)


def test_the_unseeded_clubs_are_the_promoted_ones():
    path = Path("data/web/v1/fixtures.json")
    if not path.exists():  # pragma: no cover
        pytest.skip("fixtures.json not generated yet")

    d = json.loads(path.read_text(encoding="utf-8"))
    teams = {f["team_h"] for f in d["fixtures"]} | {f["team_a"] for f in d["fixtures"]}

    assert len(teams) == 20
    assert set(d["unseeded_teams"]) <= teams
    assert INITIAL_ELO == 1500.0  # the value an unseeded club is rated at
