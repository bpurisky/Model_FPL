"""§5.1 squad reconstruction. `/api/entry/{id}/event/{gw}/picks/` 404s until
a deadline passes, and `/api/my-team/{id}/` needs session auth we don't
have. So "the current squad" is never a lookup — it's the last completed
deadline's picks, plus every transfer since with a later timestamp, applied
in order. Get this derivation wrong and everything downstream (budget,
free-transfer count, the optimizer) is silently wrong too.

Selling price (§5.3) for a player still sitting in the last picks snapshot
comes straight from the API — never recomputed. A player bought by one of
the transfers we're replaying has no picks snapshot to source it from yet,
so its selling price is derived with FPL's own public rule (half the
profit banked, rounded down; a drop passes through in full) against
whatever `current_prices` the caller can supply — defaulting to "no price
move known yet", which is the correct assumption in the absence of one.

A picks payload can also carry no price fields at all (`collector/schemas.py`
verified this live against a real 2026/27 gw1 payload — see its comment on
`EntryPick.purchase_price`/`selling_price`). When that happens, both are
taken from `current_prices` too, same as an unsnapshotted post-transfer
buy — which is exact, not approximate, for any player who has never been
part of a replayed transfer, because zero elapsed time means zero possible
price movement between "picked" and "priced from current_prices".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from collector.schemas import EntryPicksPayload, EntryTransfer


@dataclass(frozen=True)
class SquadPlayer:
    element_id: int
    purchase_price: int
    selling_price: int
    squad_position: int  # picks' own 1-15 ordering: 1-11 starting XI slot, 12-15 bench
    multiplier: int
    is_captain: bool
    is_vice_captain: bool


@dataclass(frozen=True)
class SquadState:
    as_of: datetime
    players: tuple[SquadPlayer, ...]
    bank: int  # 0.1m units, same unit as now_cost/selling_price everywhere in the API


def squad_state_to_dict(state: SquadState) -> dict[str, Any]:
    """§6.1: the shadow team's state needs to round-trip through a frozen
    JSON file so next week's freeze run can pick up where the last one
    left off without needing to replay the whole season."""
    return {"as_of": state.as_of.isoformat(), "bank": state.bank, "players": [asdict(p) for p in state.players]}


def squad_state_from_dict(data: dict[str, Any]) -> SquadState:
    as_of = datetime.fromisoformat(data["as_of"])
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    return SquadState(as_of=as_of, bank=data["bank"], players=tuple(SquadPlayer(**p) for p in data["players"]))


def apply_price_move(purchase_price: int, now_cost: int) -> int:
    """§5.3's public rule: a rise only banks half the profit, rounded down
    to the nearest 0.1m tick (so a single 0.1m rise nets the seller nothing —
    two are required); a drop passes through in full."""
    if now_cost <= purchase_price:
        return now_cost
    return purchase_price + (now_cost - purchase_price) // 2


def reconstruct_squad(
    picks: EntryPicksPayload,
    picks_snapshot_ts: datetime,
    transfers: Sequence[EntryTransfer],
    as_of: datetime,
    current_prices: Mapping[int, int] | None = None,
) -> SquadState:
    """`picks` is the payload from the most recent post-deadline snapshot,
    `picks_snapshot_ts` the time it was taken. `transfers` may include
    history from earlier gameweeks too — only entries strictly after the
    snapshot and at-or-before `as_of` are replayed. `current_prices`
    (element_id -> now_cost) lets a caller refresh every held player's
    selling price against the latest reference snapshot; omit it to keep
    the API-reported selling prices as-is for anyone not touched by a
    replayed transfer.
    """
    if as_of < picks_snapshot_ts:
        raise ValueError("as_of predates the picks snapshot it's meant to extend")

    players: dict[int, SquadPlayer] = {}
    for pick in picks.picks:
        purchase_price, selling_price = pick.purchase_price, pick.selling_price
        if purchase_price is None or selling_price is None:
            if current_prices is None or pick.element not in current_prices:
                raise ValueError(
                    f"picks payload has no price for element {pick.element}, and no current_prices "
                    "fallback was supplied to derive one"
                )
            purchase_price = selling_price = current_prices[pick.element]
        players[pick.element] = SquadPlayer(
            element_id=pick.element,
            purchase_price=purchase_price,
            selling_price=selling_price,
            squad_position=pick.position,
            multiplier=pick.multiplier,
            is_captain=pick.is_captain,
            is_vice_captain=pick.is_vice_captain,
        )
    bank = int(picks.entry_history["bank"])

    applicable = sorted(
        (t for t in transfers if picks_snapshot_ts < t.time <= as_of),
        key=lambda t: t.time,
    )
    for t in applicable:
        if t.element_out not in players:
            raise ValueError(
                f"transfer at {t.time} sells element {t.element_out}, which isn't in the reconstructed squad "
                "(out-of-order transfer history, or a snapshot gap)"
            )
        outgoing = players.pop(t.element_out)
        bank += t.element_out_cost - t.element_in_cost
        players[t.element_in] = SquadPlayer(
            element_id=t.element_in,
            purchase_price=t.element_in_cost,
            selling_price=t.element_in_cost,
            squad_position=outgoing.squad_position,
            multiplier=outgoing.multiplier,
            is_captain=outgoing.is_captain,
            is_vice_captain=outgoing.is_vice_captain,
        )

    if current_prices:
        players = {
            eid: (
                sp
                if eid not in current_prices
                else _replace_selling_price(sp, apply_price_move(sp.purchase_price, current_prices[eid]))
            )
            for eid, sp in players.items()
        }

    ordered = tuple(sorted(players.values(), key=lambda p: p.squad_position))
    return SquadState(as_of=as_of, players=ordered, bank=bank)


def _replace_selling_price(player: SquadPlayer, selling_price: int) -> SquadPlayer:
    return SquadPlayer(
        element_id=player.element_id,
        purchase_price=player.purchase_price,
        selling_price=selling_price,
        squad_position=player.squad_position,
        multiplier=player.multiplier,
        is_captain=player.is_captain,
        is_vice_captain=player.is_vice_captain,
    )
