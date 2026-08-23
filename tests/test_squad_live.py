"""§5's live data bridge (squad/live.py).

Two things under test here. First, the bug found while first running this
against the real API (2026-08-21): an in-progress gameweek reports
`minutes: 0` for any team whose fixture hasn't kicked off yet,
indistinguishable in the raw payload from a genuine blank -- which would
otherwise crater a not-yet-played star's minutes-distribution projection on
fabricated absence (Haaland, verified live, minutes: 0 purely because Man
City's fixture hadn't started).

Second, `build_train_df`'s per-gameweek sourcing. Bootstrap-static reports
season-cumulative totals, so reading them as one gameweek's stats is exact
only while one gameweek exists. The history now comes from the actuals
store and only the latest gameweek is reconstructed, as cumulative minus
what is already recorded -- a subtraction that is exact when the store is
complete and dangerously plausible when it is not, which is why the
missing-gameweek case is tested as carefully as the happy path.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import polars as pl
import pytest

from backtest.leakage import LeakageError, assert_no_leakage
from collector.schemas import BootstrapStatic, Element, Event, Team
from squad.live import (
    FLOAT_STAT_COLUMNS,
    INT_STAT_COLUMNS,
    SIGNED_STAT_COLUMNS,
    STAT_COLUMNS,
    TRAIN_SCHEMA,
    build_target_roster,
    build_train_df,
    training_feature_availability,
)

DEADLINE = datetime(2026, 8, 21, 17, 30, tzinfo=timezone.utc)


def _actual(element_id: int, gw: int, position: str = "FWD", team: str = "Team A", promoted: bool = False, **stats) -> dict:
    return {
        "gw": gw, "element_id": element_id, "position": position, "team": team, "is_promoted_club": promoted,
        **{col: 0 for col in STAT_COLUMNS}, **stats,
    }


def _actuals(*rows: dict) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=TRAIN_SCHEMA)


NO_ACTUALS = _actuals()


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
    """bootstrap-static's `elements`: season-cumulative, not per-gameweek.

    The FLOAT_STAT_COLUMNS are strings here because that is what FPL
    actually serves ("0.28", not 0.28) on both bootstrap-static and
    /event/{gw}/live/ -- verified 2026-08-22. Using real floats would let
    a missing `float()` in the reconstruction path pass these tests and
    then concatenate strings in production.
    """
    base = {**{col: 0 for col in INT_STAT_COLUMNS}, **{col: "0.0" for col in FLOAT_STAT_COLUMNS}}
    return [
        {"id": 1, **{**base, "minutes": 90, "goals_scored": 2, "total_points": 12, "expected_goals": "1.25"}},
        {"id": 2, **base},  # fixture hasn't kicked off -- minutes: 0 is NOT a real blank
        {"id": 3, **{**base, "minutes": 45, "total_points": 2, "expected_goals": "0.40"}},
    ]


def _fixtures(team1_finished: bool, team2_started: bool, team1_provisional: bool = False) -> list[dict]:
    return [
        {
            "event": 1, "team_h": 1, "team_a": 7,
            "finished": team1_finished,
            "finished_provisional": team1_finished or team1_provisional,
        },
        # team 2's own match: not played on either flag
        {"event": 1, "team_h": 2, "team_a": 999, "finished": False, "finished_provisional": False},
    ]


def test_train_df_excludes_teams_without_a_finished_fixture():
    bootstrap = _bootstrap()
    bootstrap_raw = {"elements": _raw_elements()}
    fixtures_raw = _fixtures(team1_finished=True, team2_started=False)

    df = build_train_df(bootstrap, bootstrap_raw, fixtures_raw, gw=1, actuals=NO_ACTUALS)

    included = set(df["element_id"].to_list())
    assert included == {1, 3}  # team 1 (finished) and team 7/Coventry (finished, same fixture)
    assert 2 not in included  # team 2's fixture hasn't finished -- must not appear as a false blank


def test_train_df_empty_when_no_fixtures_finished():
    bootstrap = _bootstrap()
    bootstrap_raw = {"elements": _raw_elements()}
    fixtures_raw = _fixtures(team1_finished=False, team2_started=False)

    df = build_train_df(bootstrap, bootstrap_raw, fixtures_raw, gw=1, actuals=NO_ACTUALS)

    assert df.height == 0


def test_train_df_carries_real_stats_for_finished_teams():
    bootstrap = _bootstrap()
    bootstrap_raw = {"elements": _raw_elements()}
    fixtures_raw = _fixtures(team1_finished=True, team2_started=False)

    df = build_train_df(bootstrap, bootstrap_raw, fixtures_raw, gw=1, actuals=NO_ACTUALS)

    row = df.filter(df["element_id"] == 1).row(0, named=True)
    assert row["minutes"] == 90
    assert row["goals_scored"] == 2
    assert row["gw"] == 1


def test_train_df_includes_a_provisionally_finished_fixture():
    """The regression that motivated collector/schemas.py:fixture_is_played.

    A match that has been played but whose bonus FPL hasn't confirmed yet
    reports `finished: false` with `finished_provisional: true`, and stays
    that way for many hours — long enough to cover the whole window in
    which a recommendation or a freeze actually gets made. Gating on
    `finished` alone put every player on the pooled prior for that entire
    period, which is how papertrade/freezes/gw2.json ended up with one
    identical projection for all 600 players.
    """
    bootstrap = _bootstrap()
    bootstrap_raw = {"elements": _raw_elements()}
    fixtures_raw = _fixtures(team1_finished=False, team2_started=False, team1_provisional=True)

    df = build_train_df(bootstrap, bootstrap_raw, fixtures_raw, gw=1, actuals=NO_ACTUALS)

    included = set(df["element_id"].to_list())
    assert included == {1, 3}
    assert df.filter(df["element_id"] == 1).row(0, named=True)["minutes"] == 90
    assert 2 not in included  # still excluded: neither flag set on team 2's fixture


def _gw2_fixtures(team1_played: bool = True) -> list[dict]:
    """Both gw1 fixtures done, gw2 in progress: team 1 v Coventry played,
    team 2's not kicked off."""
    return [
        {"event": 1, "team_h": 1, "team_a": 7, "finished": True, "finished_provisional": True},
        {"event": 1, "team_h": 2, "team_a": 999, "finished": True, "finished_provisional": True},
        {"event": 2, "team_h": 1, "team_a": 7, "finished": False, "finished_provisional": team1_played},
        {"event": 2, "team_h": 2, "team_a": 999, "finished": False, "finished_provisional": False},
    ]


def test_train_df_reconstructs_the_latest_gw_as_a_delta_not_a_cumulative_total():
    """The whole point of the rewrite. Element 1's bootstrap totals are
    90 minutes / 2 goals *for the season*; gw1 already accounts for 90 and
    2 of that, so gw2's own row must be 0 and 0 -- not another 90 and 2."""
    bootstrap = _bootstrap()
    bootstrap_raw = {"elements": [{**el, "minutes": 135, "goals_scored": 3, "total_points": 14} if el["id"] == 1 else el
                                  for el in _raw_elements()]}
    actuals = _actuals(
        _actual(1, gw=1, minutes=90, goals_scored=2, total_points=12),
        _actual(3, gw=1, position="DEF", team="Coventry City", promoted=True, minutes=45, total_points=2),
    )

    df = build_train_df(bootstrap, bootstrap_raw, _gw2_fixtures(), gw=2, actuals=actuals)

    gw2 = df.filter(pl.col("gw") == 2).filter(pl.col("element_id") == 1).row(0, named=True)
    assert gw2["minutes"] == 45  # 135 cumulative - 90 recorded in gw1
    assert gw2["goals_scored"] == 1  # 3 - 2
    assert gw2["total_points"] == 2  # 14 - 12


def test_train_df_keeps_recorded_history_alongside_the_reconstructed_gw():
    bootstrap = _bootstrap()
    bootstrap_raw = {"elements": _raw_elements()}
    actuals = _actuals(_actual(1, gw=1, minutes=90, goals_scored=2, total_points=12))

    df = build_train_df(bootstrap, bootstrap_raw, _gw2_fixtures(), gw=2, actuals=actuals)

    assert sorted(df.filter(pl.col("element_id") == 1)["gw"].to_list()) == [1, 2]
    # Element 2's team hasn't played gw2 and has no gw1 record: absent entirely,
    # rather than present as a fabricated blank.
    assert 2 not in set(df["element_id"].to_list())


def test_train_df_drops_the_latest_gw_when_an_earlier_one_is_unrecorded():
    """A missed `record-actuals` run. Subtracting an incomplete history
    would fold gw1 and gw2 into one row and report it as gw2 -- wrong, and
    silent. Dropping gw2 leaves the projection thin but honest."""
    bootstrap = _bootstrap()
    bootstrap_raw = {"elements": _raw_elements()}
    fixtures_raw = _gw2_fixtures() + [
        {"event": 3, "team_h": 1, "team_a": 7, "finished": False, "finished_provisional": True},
    ]
    actuals = _actuals(_actual(1, gw=1, minutes=90, total_points=12))  # gw2 never recorded

    df = build_train_df(bootstrap, bootstrap_raw, fixtures_raw, gw=3, actuals=actuals)

    assert df["gw"].to_list() == [1]


def test_train_df_returns_a_recorded_gw_verbatim_without_touching_bootstrap():
    """Once `record-actuals` has confirmed the gameweek there is nothing to
    reconstruct, and bootstrap's cumulative totals must not get a second
    vote."""
    bootstrap = _bootstrap()
    bootstrap_raw = {"elements": [{**el, "minutes": 9999} for el in _raw_elements()]}
    actuals = _actuals(
        _actual(1, gw=1, minutes=90, total_points=12),
        _actual(1, gw=2, minutes=60, total_points=5),
    )

    df = build_train_df(bootstrap, bootstrap_raw, _gw2_fixtures(), gw=2, actuals=actuals)

    assert df["minutes"].to_list() == [90, 60]


def test_train_df_never_trains_on_a_gameweek_after_the_one_asked_for():
    """papertrade/freeze.py calls this with gw = target - 1, so a
    late-arriving actuals row for the target gameweek must not leak in
    (§6.5's leakage criterion)."""
    bootstrap = _bootstrap()
    bootstrap_raw = {"elements": _raw_elements()}
    actuals = _actuals(
        _actual(1, gw=1, minutes=90, total_points=12),
        _actual(1, gw=2, minutes=60, total_points=5),  # the gameweek being predicted
    )

    df = build_train_df(bootstrap, bootstrap_raw, _gw2_fixtures(), gw=1, actuals=actuals)

    assert df["gw"].to_list() == [1]


def test_train_df_clamps_a_negative_delta_from_a_retroactive_revision():
    """FPL's dubious goals panel can shrink a total after it was recorded.
    A negative event count would poison every trailing mean downstream."""
    bootstrap = _bootstrap()
    bootstrap_raw = {"elements": _raw_elements()}  # element 1: 2 goals cumulative
    actuals = _actuals(_actual(1, gw=1, minutes=90, goals_scored=3, total_points=12))  # was recorded with 3

    df = build_train_df(bootstrap, bootstrap_raw, _gw2_fixtures(), gw=2, actuals=actuals)

    gw2 = df.filter(pl.col("gw") == 2).filter(pl.col("element_id") == 1).row(0, named=True)
    assert gw2["goals_scored"] == 0  # 2 - 3 clamped, not -1


def test_train_df_schema_matches_the_actuals_store():
    """build_train_df concatenates rows read out of the store with rows it
    builds itself, so a column-order drift between the two would be a
    silent mis-assignment."""
    from papertrade.actuals import _SCHEMA

    assert list(_SCHEMA) == list(TRAIN_SCHEMA)


def test_target_roster_flags_promoted_club_by_short_name():
    bootstrap = _bootstrap()
    roster = build_target_roster(bootstrap)

    coventry_row = roster.filter(roster["element_id"] == 3).row(0, named=True)
    other_row = roster.filter(roster["element_id"] == 1).row(0, named=True)
    assert coventry_row["is_promoted_club"] is True
    assert other_row["is_promoted_club"] is False


def test_train_df_reconstructs_float_stats_by_exact_subtraction():
    """The expected_* columns arrive from FPL as decimal strings and are
    recovered the same way the integer counts are -- cumulative minus
    what's already recorded. A missing float() here would concatenate
    rather than subtract, which is why _raw_elements() serves them as
    strings."""
    bootstrap = _bootstrap()
    bootstrap_raw = {"elements": _raw_elements()}  # element 1: xG "1.25" cumulative
    actuals = _actuals(_actual(1, gw=1, minutes=90, expected_goals=0.5))

    df = build_train_df(bootstrap, bootstrap_raw, _gw2_fixtures(), gw=2, actuals=actuals)

    gw2 = df.filter(pl.col("gw") == 2).filter(pl.col("element_id") == 1).row(0, named=True)
    assert gw2["expected_goals"] == pytest.approx(0.75)  # 1.25 - 0.50


def test_train_df_does_not_warn_on_float_rounding_noise(caplog):
    """Subtracting two 2dp values parsed from strings routinely lands a
    hair below zero with nothing revised at all. Those floor to 0.0
    silently; only a revision bigger than _FLOAT_CLAMP_EPS is a real
    clamp worth warning about."""
    bootstrap = _bootstrap()
    bootstrap_raw = {"elements": _raw_elements()}  # element 1: xG "1.25" cumulative
    # Recorded a hair above cumulative -- arithmetic noise, not a revision.
    actuals = _actuals(_actual(1, gw=1, minutes=90, expected_goals=1.25 + 1e-12))

    with caplog.at_level(logging.WARNING, logger="squad.live"):
        df = build_train_df(bootstrap, bootstrap_raw, _gw2_fixtures(), gw=2, actuals=actuals)

    gw2 = df.filter(pl.col("gw") == 2).filter(pl.col("element_id") == 1).row(0, named=True)
    assert gw2["expected_goals"] == 0.0
    assert "clamped" not in caplog.text


def test_train_df_warns_on_a_real_float_revision(caplog):
    """The other side of the tolerance: a revision FPL actually made is
    still reported, so a store row silently disagreeing with the API
    doesn't pass unnoticed."""
    bootstrap = _bootstrap()
    bootstrap_raw = {"elements": _raw_elements()}  # element 1: xG "1.25" cumulative
    actuals = _actuals(_actual(1, gw=1, minutes=90, expected_goals=1.9))  # recorded well above

    with caplog.at_level(logging.WARNING, logger="squad.live"):
        df = build_train_df(bootstrap, bootstrap_raw, _gw2_fixtures(), gw=2, actuals=actuals)

    gw2 = df.filter(pl.col("gw") == 2).filter(pl.col("element_id") == 1).row(0, named=True)
    assert gw2["expected_goals"] == 0.0
    assert "clamped" in caplog.text


def test_train_df_stat_columns_are_typed_consistently():
    """INT_STAT_COLUMNS and FLOAT_STAT_COLUMNS partition STAT_COLUMNS, and
    TRAIN_SCHEMA types each accordingly. A column landing in neither list
    would be dropped from the store without any test noticing."""
    assert INT_STAT_COLUMNS + FLOAT_STAT_COLUMNS == STAT_COLUMNS
    assert not set(INT_STAT_COLUMNS) & set(FLOAT_STAT_COLUMNS)
    assert all(TRAIN_SCHEMA[c] == pl.Int64 for c in INT_STAT_COLUMNS)
    assert all(TRAIN_SCHEMA[c] == pl.Float64 for c in FLOAT_STAT_COLUMNS)


# --------------------------------------------------------------------------
# point-in-time discipline on the live path (§6.5 criterion 3)
# --------------------------------------------------------------------------


def _dated_fixtures(gw1_kickoff: str, gw2_kickoff: str) -> list[dict]:
    return [
        {"event": 1, "team_h": 1, "team_a": 7, "finished": True, "finished_provisional": True,
         "kickoff_time": gw1_kickoff},
        {"event": 1, "team_h": 2, "team_a": 999, "finished": True, "finished_provisional": True,
         "kickoff_time": gw1_kickoff},
        {"event": 2, "team_h": 1, "team_a": 7, "finished": False, "finished_provisional": True,
         "kickoff_time": gw2_kickoff},
    ]


def test_training_features_are_dated_from_the_end_of_the_match():
    """A row's stats become knowable when its fixture ends, not when it
    kicks off."""
    bootstrap = _bootstrap()
    train_df = _actuals(_actual(1, gw=1, minutes=90))
    fixtures = _dated_fixtures("2026-08-15T14:00:00Z", "2026-08-22T14:00:00Z")

    features = training_feature_availability(train_df, bootstrap, fixtures)

    assert len(features) == 1
    assert features[0].element_id == 1
    assert features[0].available_at == datetime(2026, 8, 15, 16, 0, tzinfo=timezone.utc)  # 14:00 + 2h


def test_no_leakage_when_training_fixtures_finished_before_the_deadline():
    bootstrap = _bootstrap()
    train_df = _actuals(_actual(1, gw=1, minutes=90))
    fixtures = _dated_fixtures("2026-08-15T14:00:00Z", "2026-08-22T14:00:00Z")
    deadline = datetime(2026, 8, 22, 12, 30, tzinfo=timezone.utc)

    assert_no_leakage(training_feature_availability(train_df, bootstrap, fixtures), deadline)


def test_leakage_raises_when_a_rearranged_fixture_was_played_after_the_deadline():
    """The case the gameweek filter cannot catch. build_train_df filters
    `gw <= target - 1`, but gameweek *numbering* is not chronology: a
    postponed gw1 fixture replayed after gw2's deadline sits in a row
    labelled gw1 while containing a match that had not been played when
    the gw2 prediction was made."""
    bootstrap = _bootstrap()
    train_df = _actuals(_actual(1, gw=1, minutes=90))
    # element 1 is on team 1, whose "gw1" fixture was rearranged to after the deadline
    fixtures = _dated_fixtures("2026-08-25T14:00:00Z", "2026-08-22T14:00:00Z")
    deadline = datetime(2026, 8, 22, 12, 30, tzinfo=timezone.utc)

    with pytest.raises(LeakageError, match=r"train_df\[gw1\]"):
        assert_no_leakage(training_feature_availability(train_df, bootstrap, fixtures), deadline)


def test_training_features_skip_a_fixture_with_no_kickoff_time():
    """An unscheduled fixture cannot be shown to have been played before
    any deadline, and carries no stats yet, so it contributes no feature
    rather than a silently-safe one."""
    bootstrap = _bootstrap()
    train_df = _actuals(_actual(1, gw=1, minutes=90))
    fixtures = [{"event": 1, "team_h": 1, "team_a": 7, "finished": False,
                 "finished_provisional": False, "kickoff_time": None}]

    assert training_feature_availability(train_df, bootstrap, fixtures) == []


def test_train_df_does_not_clamp_a_legitimately_negative_bps():
    """BPS carries explicit penalties (yellow -3, red -9, own goal -6), so
    a negative value is real football, not a revision. Verified live
    2026-08-22: eight players held a negative season BPS after gw1 alone.
    Flooring it at zero would erase a real signal and fire a misleading
    "FPL revised this gameweek" warning on every run."""
    bootstrap = _bootstrap()
    raw = [{**el, "bps": -3, "total_points": -1} if el["id"] == 1 else el for el in _raw_elements()]
    bootstrap_raw = {"elements": raw}

    df = build_train_df(bootstrap, bootstrap_raw, _fixtures(team1_finished=True, team2_started=False),
                        gw=1, actuals=NO_ACTUALS)

    row = df.filter(pl.col("element_id") == 1).row(0, named=True)
    assert row["bps"] == -3
    assert row["total_points"] == -1


def test_train_df_still_clamps_a_negative_count(caplog):
    """The clamp is not removed, only narrowed: a negative goal count is
    still impossible and still indicates a revision."""
    bootstrap = _bootstrap()
    bootstrap_raw = {"elements": _raw_elements()}  # element 1: 2 goals cumulative
    actuals = _actuals(_actual(1, gw=1, minutes=90, goals_scored=3))  # recorded with 3

    with caplog.at_level(logging.WARNING, logger="squad.live"):
        df = build_train_df(bootstrap, bootstrap_raw, _gw2_fixtures(), gw=2, actuals=actuals)

    row = df.filter((pl.col("gw") == 2) & (pl.col("element_id") == 1)).row(0, named=True)
    assert row["goals_scored"] == 0
    assert "clamped" in caplog.text


def test_signed_stat_columns_are_a_subset_of_the_integer_stats():
    assert SIGNED_STAT_COLUMNS <= set(INT_STAT_COLUMNS)
