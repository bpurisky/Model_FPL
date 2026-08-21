"""§5's live data bridge (squad/live.py). Focused on the bug found while
first running this against the real API (2026-08-21): an in-progress
gameweek reports `minutes: 0` for any team whose fixture hasn't kicked off
yet, indistinguishable in the raw payload from a genuine blank -- which
would otherwise crater a not-yet-played star's minutes-distribution
projection on fabricated absence (Haaland, verified live, minutes: 0 purely
because Man City's fixture hadn't started).
"""

from __future__ import annotations

from datetime import datetime, timezone

from collector.schemas import BootstrapStatic, Element, Event, Team
from squad.live import build_target_roster, build_train_df

DEADLINE = datetime(2026, 8, 21, 17, 30, tzinfo=timezone.utc)


def _team(team_id: int, name: str, short_name: str) -> Team:
    return Team(id=team_id, name=name, short_name=short_name)


def _element(eid: int, team: int, element_type: int = 4, now_cost: int = 100) -> Element:
    return Element(
        id=eid, web_name=f"P{eid}", team=team, element_type=element_type, now_cost=now_cost,
        selected_by_percent="10.0", transfers_in_event=0, transfers_out_event=0, form="0.0", status="a",
    )


def _bootstrap() -> BootstrapStatic:
    teams = [_team(1, "Team A", "TMA"), _team(2, "Team B", "TMB"), _team(7, "Coventry City", "COV")]
    elements = [
        _element(1, team=1, element_type=4),  # Team A forward -- fixture finished
        _element(2, team=2, element_type=4),  # Team B forward -- fixture NOT finished
        _element(3, team=7, element_type=2),  # Coventry defender -- promoted club
    ]
    events = [Event(id=1, name="Gameweek 1", deadline_time=DEADLINE, finished=False, is_current=True)]
    return BootstrapStatic(events=events, teams=teams, elements=elements)


def _raw_elements() -> list[dict]:
    base = {
        "minutes": 0, "goals_scored": 0, "assists": 0, "clean_sheets": 0, "goals_conceded": 0,
        "saves": 0, "bonus": 0, "yellow_cards": 0, "red_cards": 0, "own_goals": 0,
        "penalties_missed": 0, "penalties_saved": 0, "defensive_contribution": 0,
    }
    return [
        {"id": 1, **{**base, "minutes": 90, "goals_scored": 2}},  # played a full match
        {"id": 2, **base},  # fixture hasn't kicked off -- minutes: 0 is NOT a real blank
        {"id": 3, **{**base, "minutes": 45}},
    ]


def _fixtures(team1_finished: bool, team2_started: bool) -> list[dict]:
    return [
        {"event": 1, "team_h": 1, "team_a": 7, "finished": team1_finished},
        {"event": 1, "team_h": 2, "team_a": 999, "finished": False},  # team 2's own match: not finished either way
    ]


def test_train_df_excludes_teams_without_a_finished_fixture():
    bootstrap = _bootstrap()
    bootstrap_raw = {"elements": _raw_elements()}
    fixtures_raw = _fixtures(team1_finished=True, team2_started=False)

    df = build_train_df(bootstrap, bootstrap_raw, fixtures_raw, gw=1)

    included = set(df["element_id"].to_list())
    assert included == {1, 3}  # team 1 (finished) and team 7/Coventry (finished, same fixture)
    assert 2 not in included  # team 2's fixture hasn't finished -- must not appear as a false blank


def test_train_df_empty_when_no_fixtures_finished():
    bootstrap = _bootstrap()
    bootstrap_raw = {"elements": _raw_elements()}
    fixtures_raw = _fixtures(team1_finished=False, team2_started=False)

    df = build_train_df(bootstrap, bootstrap_raw, fixtures_raw, gw=1)

    assert df.height == 0


def test_train_df_carries_real_stats_for_finished_teams():
    bootstrap = _bootstrap()
    bootstrap_raw = {"elements": _raw_elements()}
    fixtures_raw = _fixtures(team1_finished=True, team2_started=False)

    df = build_train_df(bootstrap, bootstrap_raw, fixtures_raw, gw=1)

    row = df.filter(df["element_id"] == 1).row(0, named=True)
    assert row["minutes"] == 90
    assert row["goals_scored"] == 2
    assert row["gw"] == 1


def test_target_roster_flags_promoted_club_by_short_name():
    bootstrap = _bootstrap()
    roster = build_target_roster(bootstrap)

    coventry_row = roster.filter(roster["element_id"] == 3).row(0, named=True)
    other_row = roster.filter(roster["element_id"] == 1).row(0, named=True)
    assert coventry_row["is_promoted_club"] is True
    assert other_row["is_promoted_club"] is False
