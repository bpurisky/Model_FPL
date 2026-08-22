"""§6.1 freeze protocol: before every deadline, write predictions to a
timestamped, immutable file — per-player projections, the shadow team's
recommended XI, recommended transfers, and recommended captain. Never
edited retroactively. Two independent guards enforce that:

1. `write_freeze` refuses to overwrite an existing file for a gameweek —
   the primary, load-bearing enforcement, exercised on every run.
2. `assert_immutable_in_git` (backed by real git history) can catch a
   retroactive edit that somehow bypassed #1 — e.g. a human editing the
   file directly and committing — by asserting a frozen file's git history
   has exactly one commit, at or before its gameweek's deadline.

The shadow team (§6.2) is bootstrapped once, from the real entry's actual
gameweek-1 squad (pre-model, so not something to second-guess — see
`squad/shadow.py`'s module docstring), and from gameweek 2 on is advanced
purely by replaying the *previous* freeze file's `shadow_state_after` —
never by re-deriving it from the live API, so a freeze file is a complete,
self-contained record of everything needed to reproduce what happened next.
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from analytics.scoring import load_scoring_config
from collector.client import FPLClient
from collector.config import CollectorConfig
from collector.schemas import parse_bootstrap_static, parse_entry_picks, parse_fixtures, resolve_next_event
from squad.live import build_difficulty_table, build_player_pool, build_projections, build_target_roster, build_train_df
from squad.optimize import optimize_squad
from squad.reconstruct import SquadState, reconstruct_squad, squad_state_from_dict, squad_state_to_dict
from squad.shadow import apply_recommendation
from squad.transfers import accrue_free_transfers

logger = logging.getLogger("papertrade.freeze")

FREEZES_DIR = Path("papertrade/freezes")
BOOTSTRAP_EVENT = 1  # gw1: the real, pre-model squad the shadow team starts from (§6.2)

# How long before a deadline a gameweek becomes freezable
# (see assert_within_freeze_window for why this bound exists at all).
FREEZE_WINDOW_HOURS = 6


def freeze_path(gw: int, freezes_dir: Path = FREEZES_DIR) -> Path:
    return freezes_dir / f"gw{gw}.json"


def write_freeze(gw: int, payload: dict[str, Any], freezes_dir: Path = FREEZES_DIR) -> Path:
    freezes_dir.mkdir(parents=True, exist_ok=True)
    path = freeze_path(gw, freezes_dir)
    if path.exists():
        raise FileExistsError(f"freeze for gw{gw} already exists at {path} — a frozen prediction is never overwritten")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_freeze(gw: int, freezes_dir: Path = FREEZES_DIR) -> dict[str, Any]:
    return json.loads(freeze_path(gw, freezes_dir).read_text(encoding="utf-8"))


def latest_frozen_gw(freezes_dir: Path = FREEZES_DIR) -> int | None:
    if not freezes_dir.exists():
        return None
    gws = [int(p.stem.removeprefix("gw")) for p in freezes_dir.glob("gw*.json")]
    return max(gws) if gws else None


def git_commit_times(path: Path) -> list[datetime]:
    """UTC commit timestamps that have ever touched `path`, oldest first."""
    result = subprocess.run(
        ["git", "log", "--follow", "--format=%cI", "--", str(path)],
        capture_output=True, text=True, check=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return sorted(datetime.fromisoformat(line) for line in lines)


def assert_immutable_in_git(commit_times: list[datetime], deadline: datetime) -> None:
    """§6.1's actual acceptance test: a frozen file's git history must be
    exactly one commit, made at or before its gameweek's deadline. Pure and
    testable independent of real git — `git_commit_times` is the thin
    wrapper that feeds it real data.
    """
    if len(commit_times) == 0:
        raise AssertionError("frozen file has no commits — not yet committed, or path is wrong")
    if len(commit_times) > 1:
        raise AssertionError(f"frozen file was modified {len(commit_times)} times — must be committed exactly once")
    if commit_times[0] > deadline:
        raise AssertionError(f"frozen file's only commit ({commit_times[0]}) is after its deadline ({deadline})")


class FreezeTooEarly(RuntimeError):
    """Not a failure — the scheduled job ran outside the freeze window.

    Distinct from the errors `run_freeze` raises for genuinely broken
    conditions, so `papertrade/__main__.py:cmd_freeze` can treat it the way
    it already treats an already-frozen gameweek: a logged no-op, not a
    red workflow run on every one of the six days a week it applies.
    """


def assert_within_freeze_window(
    now: datetime, deadline: datetime, gw: int, window_hours: int = FREEZE_WINDOW_HOURS
) -> None:
    """§6.1 says "before every deadline"; this pins down how long before.

    The first version of the weekly automation froze whatever gameweek
    bootstrap-static called `next`, on a daily cron, with no lower bound.
    The moment one deadline passed the following gameweek became `next`,
    so every freeze landed roughly a week early, with the least information
    it would ever have. gw2 is the surviving evidence: frozen
    2026-08-21T21:03Z against a 2026-08-28T17:30Z deadline, six days and
    twenty hours ahead of it.

    That is not a small inefficiency. The shadow team (§6.2) exists to
    answer whether the model beats the operator's judgment, and the
    operator gets to decide at the deadline with a full week of team news;
    handing the model a week-stale view makes the comparison meaningless
    in the model's disfavour. Freezing inside a window before the deadline
    is what makes the two tracks comparable.

    The window is hours rather than minutes because the job needs room to
    retry: a missed or failed run inside a six-hour window still has later
    runs to land on, whereas a thirty-minute window turns one flaky
    GitHub Actions run into a permanently unfrozen gameweek. It also keeps
    the API calls well clear of §2.2's final-ten-minutes blackout.
    """
    opens_at = deadline - timedelta(hours=window_hours)
    if now < opens_at:
        raise FreezeTooEarly(
            f"gw{gw}'s freeze window opens at {opens_at} ({window_hours}h before its "
            f"{deadline} deadline); it is {now} — too early to freeze"
        )


def assert_before_deadline(now: datetime, deadline: datetime, gw: int) -> None:
    """§0.3's point-in-time discipline, enforced for the automated weekly
    loop specifically: a freeze run that's late (workflow skipped, retried,
    or just slow) must never write a "pre-deadline" prediction file using
    data from after the deadline it claims to precede. Pure and testable on
    its own, unlike the rest of `run_freeze`'s live orchestration.
    """
    if now >= deadline:
        raise RuntimeError(f"gw{gw}'s deadline ({deadline}) has already passed — refusing to freeze late")


def bootstrap_shadow_state(picks_raw: dict, deadline: datetime, now_cost_by_id: dict[int, int]) -> SquadState:
    """The shadow team's starting point: the real entry's gw1 squad exactly
    as picked at the deadline, no transfers replayed — gw1 predates the
    model (§6.2), so it is not a recommendation to evaluate, just the
    common ancestor the real and shadow tracks diverge from at gw2.
    """
    picks = parse_entry_picks(picks_raw, logger)
    return reconstruct_squad(picks, deadline, transfers=[], as_of=deadline, current_prices=now_cost_by_id)


async def run_freeze(
    cfg: CollectorConfig,
    entry_id: int,
    gw: int | None = None,
    freezes_dir: Path = FREEZES_DIR,
    scoring_config_path: str = "config/scoring_2026_27.yaml",
    hit_cost: int = 4,
) -> Path:
    """Freezes gameweek `gw` (defaults to bootstrap-static's resolved next
    event) for the shadow team, advancing it from whatever the previous
    freeze file (or, for gw2, the real gw1 squad) left it in.
    """
    async with FPLClient(**cfg.api.client_kwargs()) as client:
        bootstrap_raw = await client.get_json("/bootstrap-static/")
        fixtures_raw = await client.get_json("/fixtures/")
        bootstrap = parse_bootstrap_static(bootstrap_raw, logger)
        parse_fixtures(fixtures_raw, logger)

        gw = gw or resolve_next_event(bootstrap, bootstrap_raw)
        if gw is None:
            raise RuntimeError("could not resolve the next gameweek from bootstrap-static")
        if gw < 2:
            raise ValueError(f"gw{gw} predates free transfers (§5.2 starts at gw2) — nothing to freeze")

        gw_deadline = next(ev.deadline_time for ev in bootstrap.events if ev.id == gw)
        now = datetime.now(timezone.utc)
        assert_within_freeze_window(now, gw_deadline, gw)
        assert_before_deadline(now, gw_deadline, gw)

        now_cost_by_id = {e.id: e.now_cost for e in bootstrap.elements}

        previous_gw = latest_frozen_gw(freezes_dir)
        if previous_gw is not None and previous_gw >= gw:
            raise FileExistsError(f"gw{gw} is already frozen (latest freeze on disk is gw{previous_gw})")

        if previous_gw is None:
            picks_raw = await client.get_json(f"/entry/{entry_id}/event/{BOOTSTRAP_EVENT}/picks/")
            deadline = next(ev.deadline_time for ev in bootstrap.events if ev.id == BOOTSTRAP_EVENT)
            shadow_state = bootstrap_shadow_state(picks_raw, deadline, now_cost_by_id)
            free_transfers = 1  # gw2's opening balance, per §5.2
        else:
            prev = load_freeze(previous_gw, freezes_dir)
            shadow_state = squad_state_from_dict(prev["shadow_state_after"])
            free_transfers = prev["free_transfers_after"]

    horizon = [gw, gw + 1, gw + 2]
    pool = build_player_pool(bootstrap)
    pool_by_id = {p.element_id: p for p in pool}
    target_roster = build_target_roster(bootstrap)
    train_df = build_train_df(bootstrap, bootstrap_raw, fixtures_raw, gw=gw - 1)
    difficulty_table = build_difficulty_table(bootstrap, fixtures_raw, horizon)
    scoring_config = load_scoring_config(Path(scoring_config_path))
    projections = build_projections(train_df, target_roster, scoring_config, difficulty_table, horizon)

    result = optimize_squad(shadow_state, pool, projections, horizon=horizon, free_transfers=free_transfers, hit_cost=hit_cost)

    shadow_state_after = apply_recommendation(shadow_state, result, pool_by_id, now_cost_by_id, next_gw=gw, as_of=now)
    free_transfers_after = accrue_free_transfers(free_transfers, len(result.transfers_out), scoring_config["free_transfers"]["max_banked"])
    deadline = gw_deadline

    payload = {
        "gameweek": gw,
        "frozen_at": now.isoformat(),
        "deadline": deadline.isoformat(),
        "horizon": horizon,
        "projections": {str(g): {str(eid): pts for eid, pts in p.items()} for g, p in projections.items()},
        "now_cost_snapshot": now_cost_by_id,
        "shadow_recommendation": {
            "transfers_out": sorted(result.transfers_out),
            "transfers_in": sorted(result.transfers_in),
            "starting_xi": sorted(result.starting_xi[gw]),
            "captain": result.captain[gw],
            "bench_order": list(result.bench_order),
            "hits_taken": result.hits_taken,
            "bank_after": result.bank_after,
        },
        "free_transfers_after": free_transfers_after,
        "shadow_state_after": squad_state_to_dict(shadow_state_after),
    }
    return write_freeze(gw, payload, freezes_dir)
