"""§5.1 squad reconstruction and §5.3 selling-price rules."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from collector.schemas import EntryPick, EntryPicksPayload, EntryTransfer
from squad.reconstruct import reconstruct_squad, squad_state_from_dict, squad_state_to_dict


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _pick(element: int, position: int, purchase_price: int, selling_price: int, is_captain: bool = False) -> EntryPick:
    return EntryPick(
        element=element,
        position=position,
        multiplier=2 if is_captain else 1,
        is_captain=is_captain,
        is_vice_captain=False,
        purchase_price=purchase_price,
        selling_price=selling_price,
    )


def _picks_payload(picks: list[EntryPick], bank: int = 5) -> EntryPicksPayload:
    return EntryPicksPayload(picks=picks, entry_history={"bank": bank, "value": 1000}, active_chip=None)


def _transfer(element_in: int, element_out: int, event: int, time: str, in_cost: int = 60, out_cost: int = 55) -> EntryTransfer:
    return EntryTransfer(
        element_in=element_in,
        element_in_cost=in_cost,
        element_out=element_out,
        element_out_cost=out_cost,
        entry=1,
        event=event,
        time=_dt(time),
    )


SNAPSHOT_TS = _dt("2026-08-21T18:00:00")


def test_no_transfers_returns_picks_as_is():
    picks = _picks_payload([_pick(101, 1, 50, 52), _pick(102, 2, 45, 45)], bank=5)
    state = reconstruct_squad(picks, SNAPSHOT_TS, transfers=[], as_of=SNAPSHOT_TS)

    assert {p.element_id for p in state.players} == {101, 102}
    assert state.bank == 5
    saka = next(p for p in state.players if p.element_id == 101)
    assert saka.selling_price == 52  # taken straight from the API, not recomputed


def test_transfer_after_snapshot_is_applied_and_bank_updates():
    picks = _picks_payload([_pick(101, 5, 50, 52)], bank=10)
    t = _transfer(element_in=201, element_out=101, event=3, time="2026-08-23T09:00:00", in_cost=60, out_cost=52)

    state = reconstruct_squad(picks, SNAPSHOT_TS, transfers=[t], as_of=_dt("2026-08-24T00:00:00"))

    assert {p.element_id for p in state.players} == {201}
    incoming = state.players[0]
    assert incoming.purchase_price == 60
    assert incoming.selling_price == 60  # no price info yet, so no profit assumed
    assert incoming.squad_position == 5  # inherits the outgoing player's slot
    assert state.bank == 10 + (52 - 60)  # bank moves by out_cost - in_cost


def test_transfer_before_snapshot_is_ignored():
    picks = _picks_payload([_pick(101, 1, 50, 52)], bank=5)
    t = _transfer(element_in=201, element_out=101, event=1, time="2026-08-20T09:00:00")

    state = reconstruct_squad(picks, SNAPSHOT_TS, transfers=[t], as_of=_dt("2026-08-24T00:00:00"))

    assert {p.element_id for p in state.players} == {101}
    assert state.bank == 5


def test_transfer_after_as_of_is_not_applied():
    picks = _picks_payload([_pick(101, 1, 50, 52)], bank=5)
    t = _transfer(element_in=201, element_out=101, event=3, time="2026-08-25T09:00:00")

    state = reconstruct_squad(picks, SNAPSHOT_TS, transfers=[t], as_of=_dt("2026-08-24T00:00:00"))

    assert {p.element_id for p in state.players} == {101}


def test_multiple_transfers_applied_in_chronological_order():
    picks = _picks_payload([_pick(101, 1, 50, 50)], bank=0)
    t1 = _transfer(element_in=201, element_out=101, event=3, time="2026-08-23T09:00:00", in_cost=50, out_cost=50)
    t2 = _transfer(element_in=301, element_out=201, event=3, time="2026-08-23T10:00:00", in_cost=55, out_cost=52)

    # deliberately out of order — the function must sort by time, not trust input order
    state = reconstruct_squad(picks, SNAPSHOT_TS, transfers=[t2, t1], as_of=_dt("2026-08-24T00:00:00"))

    assert {p.element_id for p in state.players} == {301}
    assert state.bank == 0 + (50 - 50) + (52 - 55)


def test_as_of_before_snapshot_raises():
    picks = _picks_payload([_pick(101, 1, 50, 50)])
    with pytest.raises(ValueError, match="predates"):
        reconstruct_squad(picks, SNAPSHOT_TS, transfers=[], as_of=_dt("2026-08-20T00:00:00"))


def test_selling_an_untracked_player_raises():
    picks = _picks_payload([_pick(101, 1, 50, 50)])
    bogus_transfer = _transfer(element_in=201, element_out=999, event=3, time="2026-08-23T09:00:00")

    with pytest.raises(ValueError, match="999"):
        reconstruct_squad(picks, SNAPSHOT_TS, transfers=[bogus_transfer], as_of=_dt("2026-08-24T00:00:00"))


def test_current_prices_refreshes_selling_price_for_held_and_transferred_players():
    picks = _picks_payload([_pick(101, 1, 50, 50), _pick(102, 2, 40, 40)], bank=0)
    t = _transfer(element_in=201, element_out=102, event=3, time="2026-08-23T09:00:00", in_cost=40, out_cost=40)

    # 101 unchanged since buy: no profit. 201 has risen 4 ticks since it was bought at 40 -> now 44.
    state = reconstruct_squad(
        picks, SNAPSHOT_TS, transfers=[t], as_of=_dt("2026-08-24T00:00:00"),
        current_prices={101: 50, 201: 44},
    )

    by_id = {p.element_id: p for p in state.players}
    assert by_id[101].selling_price == 50  # no move
    assert by_id[201].selling_price == 40 + (44 - 40) // 2  # = 42, half profit rounded down


def test_current_prices_full_drop_passes_through():
    picks = _picks_payload([_pick(101, 1, 50, 50)], bank=0)
    state = reconstruct_squad(
        picks, SNAPSHOT_TS, transfers=[], as_of=SNAPSHOT_TS, current_prices={101: 45},
    )
    assert state.players[0].selling_price == 45


def _priceless_pick(element: int, position: int, is_captain: bool = False) -> EntryPick:
    """A pick as the real 2026/27 gw1 API actually returns it (verified live,
    2026-08-21): no purchase_price/selling_price at all."""
    return EntryPick(
        element=element, position=position, multiplier=2 if is_captain else 1,
        is_captain=is_captain, is_vice_captain=False,
    )


def test_missing_price_falls_back_to_current_prices():
    picks = _picks_payload([_priceless_pick(101, 1), _priceless_pick(102, 2)], bank=5)
    state = reconstruct_squad(
        picks, SNAPSHOT_TS, transfers=[], as_of=SNAPSHOT_TS, current_prices={101: 60, 102: 45},
    )

    by_id = {p.element_id: p for p in state.players}
    # exact, not approximate: zero elapsed time since the snapshot means the
    # price genuinely cannot have moved, so purchase == selling == now_cost.
    assert by_id[101].purchase_price == by_id[101].selling_price == 60
    assert by_id[102].purchase_price == by_id[102].selling_price == 45


def test_missing_price_without_current_prices_raises():
    picks = _picks_payload([_priceless_pick(101, 1)], bank=5)
    with pytest.raises(ValueError, match="101"):
        reconstruct_squad(picks, SNAPSHOT_TS, transfers=[], as_of=SNAPSHOT_TS)


def test_squad_state_json_round_trip():
    picks = _picks_payload([_pick(101, 1, 50, 52, is_captain=True), _pick(102, 2, 40, 40)], bank=7)
    state = reconstruct_squad(picks, SNAPSHOT_TS, transfers=[], as_of=SNAPSHOT_TS)

    restored = squad_state_from_dict(squad_state_to_dict(state))

    assert restored.bank == state.bank
    assert restored.as_of == state.as_of
    assert restored.players == state.players
