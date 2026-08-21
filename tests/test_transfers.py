"""§5.2 free transfer derivation — every hit calculation downstream depends
on this, so it gets tested against a full simulated season including chip
weeks, not just isolated single-gameweek cases."""

from __future__ import annotations

from datetime import datetime, timezone

from collector.schemas import EntryChip, EntryTransfer
from squad.transfers import derive_free_transfers, hit_cost


def _dt(gw: int) -> datetime:
    return datetime(2026, 8, 14 + gw, tzinfo=timezone.utc)


def _transfer(event: int, n: int = 1) -> list[EntryTransfer]:
    return [
        EntryTransfer(
            element_in=100 + i, element_in_cost=50, element_out=200 + i, element_out_cost=50,
            entry=1, event=event, time=_dt(event),
        )
        for i in range(n)
    ]


def test_no_transfers_accrues_and_caps_at_max_banked():
    states = derive_free_transfers(transfers=[], chips=[], through_event=8, max_banked=5)
    available = {s.event: s.available_before for s in states}
    # 1 at event 2, growing by 1/week, capped at 5 from event 6 onward
    assert available == {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 5, 8: 5}


def test_using_one_free_transfer_leaves_bank_flat_next_week():
    transfers = _transfer(event=3, n=1)
    states = derive_free_transfers(transfers, chips=[], through_event=4)
    by_event = {s.event: s for s in states}

    assert by_event[3].available_before == 2  # accrued from event 2
    assert by_event[3].used == 1
    assert by_event[3].hit_count == 0
    assert by_event[4].available_before == 2  # 2 - 1 used + 1 accrued


def test_transfers_beyond_free_allowance_are_hits():
    transfers = _transfer(event=2, n=3)  # only 1 FT available at event 2
    states = derive_free_transfers(transfers, chips=[], through_event=3)
    by_event = {s.event: s for s in states}

    assert by_event[2].available_before == 1
    assert by_event[2].hit_count == 2  # 3 made, 1 free
    assert hit_cost(by_event[2]) == 8
    assert by_event[3].available_before == 1  # floored at 0 leftover, +1 accrual


def test_wildcard_week_is_free_and_leaves_bank_untouched():
    transfers = _transfer(event=4, n=6)  # a big wildcard rebuild
    chips = [EntryChip(name="wildcard", event=4)]
    states = derive_free_transfers(transfers, chips, through_event=5)
    by_event = {s.event: s for s in states}

    assert by_event[4].hit_count == 0
    assert by_event[4].used == 0
    # banked FT going into event 4 was 3 (event2=1,3=2,4=3); wildcard doesn't
    # touch it, so event 5 just accrues +1 from whatever was there.
    assert by_event[4].available_before == 3
    assert by_event[5].available_before == 4


def test_free_hit_week_also_leaves_bank_untouched():
    transfers = _transfer(event=3, n=4)
    chips = [EntryChip(name="freehit", event=3)]
    states = derive_free_transfers(transfers, chips, through_event=4)
    by_event = {s.event: s for s in states}

    assert by_event[3].hit_count == 0
    # 2 banked going into event 3 (accrued from event 2), untouched by the
    # free hit, so event 4 sees 2 + 1 accrual.
    assert by_event[3].available_before == 2
    assert by_event[4].available_before == 3


def test_full_season_walk_with_mixed_activity_and_a_chip():
    # event: 2 (0 transfers), 3 (1 transfer), 4 (0), 5 (wildcard, 5 transfers), 6 (2 transfers)
    transfers = _transfer(2, 0) + _transfer(3, 1) + _transfer(5, 5) + _transfer(6, 2)
    chips = [EntryChip(name="wildcard", event=5)]
    states = derive_free_transfers(transfers, chips, through_event=6)
    by_event = {s.event: s for s in states}

    assert by_event[2].available_before == 1
    assert by_event[3].available_before == 2
    assert by_event[3].used == 1
    assert by_event[4].available_before == 2  # 2-1+1
    assert by_event[5].available_before == 3  # 2+1, untouched by transfer count until wildcard check
    assert by_event[5].hit_count == 0
    assert by_event[6].available_before == 4  # wildcard didn't consume the 3 banked, +1 accrual
    assert by_event[6].hit_count == max(0, 2 - 4)  # plenty of free transfers, no hit
    assert by_event[6].hit_count == 0
