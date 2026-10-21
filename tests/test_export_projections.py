"""§5.11.1 over `projections.json`.

The horizon logic is tested on synthetic data, where a blank and a double
can be constructed on purpose: the live schedule has neither in the next
six gameweeks right now, and a test that depends on the calendar would
pass vacuously until the week it mattered. The build against the real
data checks the shape of the file and that it agrees with `players.json`'s
own single-gameweek projection.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import polars as pl
import pytest

from analytics.scoring import load_scoring_config
from web.export.contract import ProjectionsFile
from web.export.players import target_roster
from web.export.projections import (
    HORIZON,
    build_projections,
    horizon_gameweeks,
    project_horizon,
    team_fixtures,
)

CONFIG = load_scoring_config(Path("config/scoring_2025_26.yaml"))
BASE_DAY = datetime(2025, 1, 1, tzinfo=timezone.utc)
PATH = Path("data/web/v1/projections.json")
PANEL = Path("data/web/v1/panel.parquet")


def _row(element_id, team, gw, position="MID", goals=0):
    return {
        "element_id": element_id, "gw": gw, "minutes": 90, "goals_scored": goals, "assists": 0,
        "expected_goals": float(goals), "expected_assists": 0.0, "clean_sheets": 0, "goals_conceded": 1,
        "saves": 0, "bonus": 0, "yellow_cards": 0, "red_cards": 0, "own_goals": 0,
        "penalties_missed": 0, "penalties_saved": 0, "defensive_contribution": None,
        "position": position, "team": team, "is_promoted_club": False,
        "kickoff_time": BASE_DAY + timedelta(days=gw),
    }


TEAMS = {1: ("Team A", "AAA"), 2: ("Team B", "BBB"), 3: ("Team C", "CCC"), 4: ("Team D", "DDD")}


def _synthetic():
    """A and B meet in gw3 and gw4, C and D only in gw3 (a blank for both
    in gw4), and gw5 is a double for A and C, who play each other twice."""
    train = pl.DataFrame([_row(i, TEAMS[i][0], gw, goals=1) for i in TEAMS for gw in (1, 2)])
    roster = target_roster(train.with_columns(pl.lit("s").alias("season")), "s", 2)
    fixtures = pl.DataFrame(
        {
            "gw": [3, 3, 4, 5, 5],
            "team_h": ["Team A", "Team C", "Team A", "Team A", "Team C"],
            "team_a": ["Team B", "Team D", "Team B", "Team C", "Team A"],
        }
    )
    strengths = pl.DataFrame(
        {"team": [t[0] for t in TEAMS.values()], "attack": [1.4] * 4, "defence": [1.2] * 4}
    )
    return train, roster, fixtures, strengths


def _projected():
    train, roster, fixtures, strengths = _synthetic()
    return project_horizon(train, roster, CONFIG, fixtures, strengths, [3, 4, 5])


def _xp(projected, gw, team):
    return projected[gw].filter(pl.col("team") == team)["xp"].item()


# --- the horizon ----------------------------------------------------------


def test_the_horizon_is_six_gameweeks_and_stops_at_the_end_of_the_season():
    assert horizon_gameweeks(7, 38) == [8, 9, 10, 11, 12, 13]
    assert len(horizon_gameweeks(7, 38)) == HORIZON
    assert horizon_gameweeks(36, 38) == [37, 38]
    assert horizon_gameweeks(38, 38) == []
    assert horizon_gameweeks(7, None) == []


# --- blanks and doubles ---------------------------------------------------


def test_a_blank_gameweek_projects_exactly_zero_for_the_club_with_no_fixture():
    projected = _projected()
    assert _xp(projected, 4, "Team C") == 0.0
    assert _xp(projected, 4, "Team D") == 0.0
    assert _xp(projected, 4, "Team A") > 0.0


def test_a_double_gameweek_sums_both_fixtures():
    """Team A plays once in gw4 and twice in gw5 against the same
    strengths, so its gw5 total is roughly one fixture's worth twice over.
    The bound is loose on purpose: clean sheets and goals conceded are not
    exactly additive, but a total near one fixture's worth would mean the
    second was dropped."""
    projected = _projected()
    single, double = _xp(projected, 4, "Team A"), _xp(projected, 5, "Team A")
    assert double > 1.5 * single
    assert projected[5].filter(pl.col("team") == "Team A")["n_fixtures"].item() == 2


def test_xp_is_non_negative_in_every_gameweek():
    for df in _projected().values():
        assert (df["xp"] >= 0).all()


def test_team_fixtures_lists_blanks_as_empty_and_doubles_as_two():
    ref = pl.DataFrame(
        {
            "event": [3, 5, 5], "team_h": [1, 1, 3], "team_a": [2, 3, 1],
            "kickoff_time": [BASE_DAY + timedelta(days=d) for d in (3, 5, 6)],
            "team_h_difficulty": [2, 4, 3], "team_a_difficulty": [5, 3, 2],
        }
    )
    teams = pl.DataFrame(
        {"id": list(TEAMS), "name": [t[0] for t in TEAMS.values()], "short_name": [t[1] for t in TEAMS.values()]}
    )
    built = team_fixtures(ref, teams, [3, 4, 5])
    out = {t.team: {g.gw: g.fixtures for g in t.gameweeks} for t in built}

    assert out["Team A"][3][0].opponent == "BBB" and out["Team A"][3][0].home is True
    assert out["Team A"][3][0].difficulty == 2
    assert out["Team B"][3][0].home is False and out["Team B"][3][0].difficulty == 5
    assert out["Team A"][4] == []  # blank
    assert [f.opponent for f in out["Team A"][5]] == ["CCC", "CCC"]  # double
    assert [f.home for f in out["Team A"][5]] == [True, False]
    assert all(v == [] for v in out["Team D"].values())
    assert all(list(v) == [3, 4, 5] for v in out.values())


# --- the minutes head -----------------------------------------------------


def test_the_minutes_distribution_sums_to_one():
    first = _projected()[3]
    assert all(abs(t - 1.0) < 1e-9 for t in (first["p_blank"] + first["p_short"] + first["p_full"]).to_list())


# --- the file -------------------------------------------------------------


@lru_cache(maxsize=1)
def _built() -> ProjectionsFile:
    return build_projections()


def _need_panel():
    if not PANEL.exists():  # pragma: no cover
        pytest.skip("panel.parquet is a build artifact; run `python -m web.export panel`")


def test_every_player_has_one_xp_per_gameweek_and_the_roster_ids_match():
    _need_panel()
    file = _built()
    assert 1 <= len(file.gameweeks) <= HORIZON
    assert file.gameweeks == list(range(file.gameweek + 1, file.gameweek + 1 + len(file.gameweeks)))
    panel = pl.read_parquet(PANEL).filter((pl.col("season") == file.season) & (pl.col("gw") == file.gameweek))
    assert sorted(p.element_id for p in file.players) == sorted(panel["element_id"].unique().to_list())
    for player in file.players:
        assert len(player.xp) == len(file.gameweeks)
        assert all(x is not None and x >= 0 for x in player.xp)
    assert file.header.rows == len(file.players)


def test_every_club_carries_an_entry_for_each_horizon_gameweek():
    _need_panel()
    file = _built()
    assert len(file.teams) == 20
    for team in file.teams:
        assert [g.gw for g in team.gameweeks] == file.gameweeks
    assert {p.team for p in file.players} <= {t.team for t in file.teams}


def test_the_first_gameweek_agrees_with_players_json():
    """The same model, so the same number. `players.json` is rewritten only
    when its body changes, so the committed copy can lag the data on disk;
    both are built fresh here rather than compared against the file."""
    _need_panel()
    from web.export.players import build_players

    as_of = datetime.now(timezone.utc)
    players = build_players(as_of=as_of)
    file = build_projections(as_of=as_of)
    assert players.projected_gameweek == file.gameweeks[0]
    totals = {p.element_id: p.projection.total for p in players.players if p.projection}
    for player in file.players:
        assert player.xp[0] == pytest.approx(totals[player.element_id], abs=1e-9)


def test_the_committed_file_validates_against_the_contract():
    if not PATH.exists():  # pragma: no cover
        pytest.skip("projections.json not committed")
    file = ProjectionsFile.model_validate(json.loads(PATH.read_text(encoding="utf-8")))
    assert file.header.contract_version == 1
    for player in file.players:
        assert len(player.xp) == len(file.gameweeks)
