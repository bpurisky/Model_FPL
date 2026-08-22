"""§5.2 free transfer derivation. No public endpoint exposes this — it's
walked from transfer history and chip usage. Every hit-cost calculation
downstream (and therefore every optimizer recommendation) depends on this
being right, so it's isolated here with its own heavy test coverage rather
than folded into squad/reconstruct.py.

Rule, as implemented: one free transfer accrues per gameweek, banked up to
`max_banked` (read from config/scoring_*.yaml's free_transfers.max_banked,
not hardcoded here — see the caller). A wildcard or free hit makes that
gameweek's transfers free and leaves the banked count untouched for the
following gameweek, per current official rules; it neither resets nor
consumes it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from collector.schemas import EntryChip, EntryTransfer

# Real API chip name strings for the two "unlimited transfers this week" chips.
UNLIMITED_TRANSFER_CHIPS = {"wildcard", "freehit"}


@dataclass(frozen=True)
class FreeTransferState:
    event: int
    available_before: int  # banked FT going into this event, before that event's transfers
    used: int  # transfers made this event that drew on the free-transfer bank
    hit_count: int  # transfers beyond the free allowance (0 during a chip week)


def accrue_free_transfers(available_before: int, made: int, max_banked: int = 5, is_chip_week: bool = False) -> int:
    """The single-gameweek step of the rule (module docstring): one FT
    accrues, banked up to `max_banked`, unless it's a wildcard/free-hit week
    (unlimited transfers, but the bank neither resets nor consumes).
    Exposed on its own — not just inlined into `derive_free_transfers`'s
    loop — because `papertrade/freeze.py` needs to advance the shadow
    team's own free-transfer count one week at a time without replaying its
    whole transfer history each run.
    """
    leftover = available_before if is_chip_week else max(0, available_before - made)
    return min(max_banked, leftover + 1)


def derive_free_transfers(
    transfers: list[EntryTransfer],
    chips: list[EntryChip],
    through_event: int,
    start_event: int = 2,
    max_banked: int = 5,
) -> list[FreeTransferState]:
    """One state per gameweek from `start_event` to `through_event`
    inclusive. `start_event` defaults to 2 — gameweek 1's squad is a free
    build, not a transfer, so free-transfer accounting begins at 2 with a
    single banked transfer, matching current official rules.
    """
    transfer_counts = Counter(t.event for t in transfers)
    chip_events = {c.event for c in chips if c.name in UNLIMITED_TRANSFER_CHIPS}

    states: list[FreeTransferState] = []
    available = 1
    for event in range(start_event, through_event + 1):
        made = transfer_counts.get(event, 0)
        is_chip_week = event in chip_events
        used = 0 if is_chip_week else made
        hits = 0 if is_chip_week else max(0, made - available)
        states.append(FreeTransferState(event=event, available_before=available, used=used, hit_count=hits))
        available = accrue_free_transfers(available, made, max_banked, is_chip_week)
    return states


def hit_cost(state: FreeTransferState, cost_per_hit: int = 4) -> int:
    return state.hit_count * cost_per_hit
