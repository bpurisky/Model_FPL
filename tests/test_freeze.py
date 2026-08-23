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
    HIT_ELIGIBILITY_GWS,
    FreezeTooEarly,
    assert_before_deadline,
    assert_immutable_in_git,
    assert_within_freeze_window,
    bootstrap_shadow_state,
    latest_frozen_gw,
    load_freeze,
    transfer_cap,
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


# --- transfer cap (the optimizer's licence to pay hits) -------------------


def test_transfer_cap_binds_at_one_gameweek_of_history():
    """The gw2 case, and the reason this exists: one gameweek in, the
    uncapped optimizer proposed 11 transfers for -40 points against
    projections that are trailing means over a single observation."""
    cap = transfer_cap(n_train_gws=1, free_transfers=1)

    assert cap["applied"] is True
    assert cap["max_transfers"] == 1  # the free allowance exactly: no hits


def test_transfer_cap_lifts_once_the_model_reaches_its_own_window():
    """Uncapped means max_transfers=None -- optimize_squad may pay hits
    again, because the trailing means now average over the window the
    event model was actually validated at."""
    cap = transfer_cap(n_train_gws=HIT_ELIGIBILITY_GWS, free_transfers=2)

    assert cap["applied"] is False
    assert cap["max_transfers"] is None


def test_transfer_cap_is_inclusive_at_the_threshold():
    assert transfer_cap(HIT_ELIGIBILITY_GWS - 1, 1)["applied"] is True
    assert transfer_cap(HIT_ELIGIBILITY_GWS, 1)["applied"] is False


def test_transfer_cap_with_no_free_transfers_forces_a_hold():
    """Intended, not an edge case: with nothing free and a model below its
    own window, every available move costs 4 points against a projection
    that has not earned them."""
    cap = transfer_cap(n_train_gws=2, free_transfers=0)

    assert cap["applied"] is True
    assert cap["max_transfers"] == 0


def test_transfer_cap_records_its_evidence_either_way():
    """A capped hold and a chosen hold are the same squad on disk. §6.3's
    squad-level series is only readable if the freeze says which it was."""
    for n in (0, 1, HIT_ELIGIBILITY_GWS, HIT_ELIGIBILITY_GWS + 3):
        cap = transfer_cap(n_train_gws=n, free_transfers=1)
        assert cap["n_train_gws"] == n
        assert cap["threshold_gws"] == HIT_ELIGIBILITY_GWS
        assert set(cap) == {"applied", "max_transfers", "n_train_gws", "threshold_gws"}


def test_hit_eligibility_threshold_is_the_models_own_window():
    """Not a number chosen here. If DEFAULT_WINDOW moves, this moves with
    it -- the claim is 'below the window the model was tuned at', not
    'below five'."""
    from analytics.projections import DEFAULT_WINDOW

    assert HIT_ELIGIBILITY_GWS == DEFAULT_WINDOW
