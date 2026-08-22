"""§6.1 freeze protocol: immutability enforcement and the shadow-team
bootstrap. `run_freeze` itself (the live end-to-end orchestration) is
exercised by actually running it against the real API, not unit-tested
here — there's too much live state (bootstrap-static, fixtures, an entry's
picks) to usefully fake without just re-implementing squad/live.py's own
already-tested pieces.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from papertrade.freeze import (
    FREEZE_WINDOW_HOURS,
    FreezeTooEarly,
    assert_before_deadline,
    assert_immutable_in_git,
    assert_within_freeze_window,
    bootstrap_shadow_state,
    latest_frozen_gw,
    load_freeze,
    write_freeze,
)

DEADLINE = datetime(2026, 8, 28, 17, 30, tzinfo=timezone.utc)


def test_write_freeze_refuses_to_overwrite(tmp_path):
    write_freeze(2, {"gameweek": 2}, freezes_dir=tmp_path)

    with pytest.raises(FileExistsError, match="gw2"):
        write_freeze(2, {"gameweek": 2, "tampered": True}, freezes_dir=tmp_path)

    # the original content survives the attempted overwrite untouched
    assert load_freeze(2, freezes_dir=tmp_path) == {"gameweek": 2}


def test_load_freeze_round_trips_write_freeze(tmp_path):
    payload = {"gameweek": 3, "shadow_recommendation": {"captain": 11}}
    write_freeze(3, payload, freezes_dir=tmp_path)
    assert load_freeze(3, freezes_dir=tmp_path) == payload


def test_latest_frozen_gw_finds_the_highest_and_handles_none(tmp_path):
    assert latest_frozen_gw(tmp_path) is None
    write_freeze(2, {}, freezes_dir=tmp_path)
    write_freeze(4, {}, freezes_dir=tmp_path)
    write_freeze(3, {}, freezes_dir=tmp_path)
    assert latest_frozen_gw(tmp_path) == 4


def test_assert_immutable_in_git_passes_for_one_commit_before_deadline():
    assert_immutable_in_git([DEADLINE - timedelta(hours=1)], DEADLINE)  # must not raise


def test_assert_immutable_in_git_rejects_a_commit_after_the_deadline():
    with pytest.raises(AssertionError, match="after its deadline"):
        assert_immutable_in_git([DEADLINE + timedelta(minutes=5)], DEADLINE)


def test_assert_immutable_in_git_rejects_more_than_one_commit():
    with pytest.raises(AssertionError, match="modified 2 times"):
        assert_immutable_in_git([DEADLINE - timedelta(hours=2), DEADLINE - timedelta(hours=1)], DEADLINE)


def test_assert_immutable_in_git_rejects_no_commits():
    with pytest.raises(AssertionError, match="no commits"):
        assert_immutable_in_git([], DEADLINE)


def test_assert_within_freeze_window_rejects_a_freeze_a_week_early():
    """The gw2 regression, as data: it was frozen 2026-08-21T21:03Z against
    a 2026-08-28T17:30Z deadline. Under the window guard that run is a
    no-op, and gw2 stays freezable until it can be decided on real
    information."""
    frozen_at = datetime(2026, 8, 21, 21, 3, tzinfo=timezone.utc)
    with pytest.raises(FreezeTooEarly, match="too early"):
        assert_within_freeze_window(frozen_at, DEADLINE, gw=2)


def test_assert_within_freeze_window_allows_a_run_inside_the_window():
    assert_within_freeze_window(DEADLINE - timedelta(hours=1), DEADLINE, gw=2)  # must not raise


def test_assert_within_freeze_window_opens_exactly_on_the_boundary():
    assert_within_freeze_window(DEADLINE - timedelta(hours=FREEZE_WINDOW_HOURS), DEADLINE, gw=2)  # must not raise


def test_assert_within_freeze_window_rejects_one_second_before_it_opens():
    just_early = DEADLINE - timedelta(hours=FREEZE_WINDOW_HOURS, seconds=1)
    with pytest.raises(FreezeTooEarly):
        assert_within_freeze_window(just_early, DEADLINE, gw=2)


def test_freeze_window_leaves_room_for_a_retry_at_the_scheduled_cadence():
    """The window and .github/workflows/papertrade.yml's cron are one
    decision, not two: a two-hourly job must get more than one shot inside
    the window, or a single failed run silently costs a gameweek."""
    cron_interval_hours = 2
    assert FREEZE_WINDOW_HOURS // cron_interval_hours >= 3


def test_assert_before_deadline_passes_when_before():
    assert_before_deadline(DEADLINE - timedelta(hours=1), DEADLINE, gw=2)  # must not raise


def test_assert_before_deadline_rejects_a_late_run():
    with pytest.raises(RuntimeError, match="gw2"):
        assert_before_deadline(DEADLINE + timedelta(minutes=1), DEADLINE, gw=2)


def test_assert_before_deadline_rejects_exactly_at_deadline():
    with pytest.raises(RuntimeError, match="already passed"):
        assert_before_deadline(DEADLINE, DEADLINE, gw=2)


def _picks_raw(elements: list[int]) -> dict:
    return {
        "picks": [
            {"element": eid, "position": i + 1, "multiplier": 2 if i == 0 else (1 if i < 11 else 0),
             "is_captain": i == 0, "is_vice_captain": i == 1}
            for i, eid in enumerate(elements)
        ],
        "entry_history": {"bank": 5, "value": 1000},
        "active_chip": None,
    }


def test_bootstrap_shadow_state_uses_current_prices_for_priceless_picks():
    picks_raw = _picks_raw(list(range(101, 116)))
    now_cost_by_id = {eid: 50 + i for i, eid in enumerate(range(101, 116))}

    state = bootstrap_shadow_state(picks_raw, DEADLINE, now_cost_by_id)

    assert len(state.players) == 15
    assert state.bank == 5
    for sp in state.players:
        assert sp.purchase_price == sp.selling_price == now_cost_by_id[sp.element_id]
    captain = next(sp for sp in state.players if sp.is_captain)
    assert captain.element_id == 101
