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

from analytics.projections import DEFAULT_WINDOW
from analytics.scoring import load_scoring_config
from collector.client import FPLClient
from collector.config import CollectorConfig
from collector.schemas import parse_bootstrap_static, parse_entry_picks, parse_fixtures, resolve_next_event
from backtest.leakage import assert_no_leakage
from squad.live import (
    build_difficulty_table,
    build_player_pool,
    build_projections,
    build_target_roster,
    build_train_df,
    training_feature_availability,
)
from squad.optimize import optimize_squad
from squad.reconstruct import SquadState, reconstruct_squad, squad_state_from_dict, squad_state_to_dict
from squad.shadow import apply_recommendation
from squad.transfers import accrue_free_transfers

logger = logging.getLogger("papertrade.freeze")

FREEZES_DIR = Path("papertrade/freezes")
BOOTSTRAP_EVENT = 1  # gw1: the real, pre-model squad the shadow team starts from (§6.2)

# How long before a deadline a gameweek becomes freezable
# (see assert_within_freeze_window for why this bound exists at all, and
# why 6 wasn't enough in practice).
FREEZE_WINDOW_HOURS = 24

# The optimizer may not pay for transfer hits until the model has at least
# as many gameweeks of current-season history as its own trailing window.
#
# Not a taste call about aggression. optimize_squad maximizes projected
# points net of hit cost and treats each projection as a point estimate
# with no uncertainty attached, so the thinner the history the more
# willing it is to buy noise. Verified live on 2026-08-23, one gameweek
# into the season: uncapped it proposed 11 transfers for -40 points,
# benching Haaland and B.Fernandes behind players who happened to return
# in gw1; capped to the free transfer it proposed one, and kept them.
#
# The threshold is the model's own DEFAULT_WINDOW rather than a number
# picked here: below it every trailing mean is averaging fewer
# observations than the window the event model was tuned and backtested
# against (§4.2, §4.4), so the projections the optimizer is paying 4
# points a time to act on are outside the regime where they were shown
# to beat the baselines at all.
HIT_ELIGIBILITY_GWS = DEFAULT_WINDOW


def transfer_cap(n_train_gws: int, free_transfers: int) -> dict[str, Any]:
    """How many transfers the optimizer may make this gameweek, and why.

    Returns the record that goes into the freeze verbatim.
    `max_transfers` of None means uncapped — optimize_squad is free to
    pay hits because the history is deep enough to be worth paying for.
    Below the threshold the cap is the free allowance exactly, which is
    the same statement as "no hits": a hit is by definition a transfer
    beyond it.

    Note a free_transfers of 0 caps at 0, forcing a hold. That is the
    intended reading rather than an edge case — with no free transfer and
    a model below its own window, every available move costs 4 points
    against a projection that has not earned them.
    """
    applied = n_train_gws < HIT_ELIGIBILITY_GWS
    return {
        "applied": applied,
        "max_transfers": free_transfers if applied else None,
        "n_train_gws": n_train_gws,
        "threshold_gws": HIT_ELIGIBILITY_GWS,
    }


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


def model_git_sha() -> str | None:
    """The commit the model was at when a freeze was written, or None if
    that cannot be established (no git, detached checkout, git missing).

    Every number in a freeze has to be traceable to the exact code that
    produced it. A projection without a sha is a number with no way back
    to its own definition, which is precisely what §8's "every reported
    metric must regenerate from committed data" rules out — and Phase 5's
    export contract restates as non-negotiable for anything it renders.

    Returns None rather than raising: an unavailable sha must not stop a
    freeze from being written inside its window. A missing sha is recorded
    as missing, which is honest; a skipped freeze loses the gameweek.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        logger.warning("could not resolve model_git_sha: %s", exc)
        return None
    return result.stdout.strip() or None


def git_is_dirty() -> bool | None:
    """Whether the working tree had uncommitted changes when the freeze
    was written. Recorded alongside the sha because a dirty tree means the
    sha does *not* fully describe the code that ran, and a reader
    comparing a freeze against that commit later would otherwise have no
    way to know that."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    return bool(result.stdout.strip())


def _git_log_entries(path: Path, *extra: str) -> list[tuple[str, datetime]]:
    """(sha, commit time) for `path`, newest first — git log's own order.

    `--no-renames` so a rename's delete side is reported against this path
    as a plain deletion rather than being paired away as an R — the
    counting below depends on seeing it, and rename detection is a
    user-configurable default (`diff.renames`) this must not be at the
    mercy of.

    The sha is carried because commit *times* cannot order commits.
    `%cI` has one-second resolution, and a retirement followed promptly by
    a re-freeze lands both in the same second — verified by constructing
    exactly that sequence, where filtering on time discarded the new
    freeze along with the old one and reported a committed file as having
    no commits at all.
    """
    result = subprocess.run(
        ["git", "log", "--no-renames", "--format=%H %cI", *extra, "--", str(path)],
        capture_output=True, text=True, check=True,
    )
    entries = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        sha, _, iso = line.partition(" ")
        entries.append((sha, datetime.fromisoformat(iso)))
    return entries


def git_commit_times(path: Path) -> list[datetime]:
    """UTC commit times for the file *currently* at `path`, oldest first.

    Two deliberate departures from "every commit that ever touched this
    path", and both come from a real event in this repo's history rather
    than from a hypothetical.

    **`--follow` is gone.** It traces content across renames, which answers
    "where did this data come from" — a different question from "was the
    file at this path written once and never edited", which is the only
    one §6.1 asks.

    **Commits at or before the path's most recent deletion are excluded.**
    `papertrade/freezes/gw2.json` was written seven days early, outside
    its window, and retired to `tests/fixtures/degenerate_freeze/` on
    2026-08-23. That retirement is a deletion of this path, so the path
    already carries two commits before its real freeze has been written at
    all. Counting them would fail the immutability assertion for the one
    reason it is not meant to catch: nothing was edited, a bad file was
    removed. Retiring a premature freeze is exactly the corrective action
    §6.1 should permit, and refusing to allow it would make the guard
    argue for keeping known-degenerate data.
    """
    deleted = {sha for sha, _ in _git_log_entries(path, "--diff-filter=D")}
    current: list[datetime] = []
    for sha, when in _git_log_entries(path):  # newest first
        if sha in deleted:
            break  # everything from here back belongs to a previous file
        current.append(when)
    return sorted(current)


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
    retry: a missed or failed run inside the window still has later runs to
    land on, whereas a thirty-minute window turns one flaky GitHub Actions
    run into a permanently unfrozen gameweek. It also keeps the API calls
    well clear of §2.2's final-ten-minutes blackout.

    Six hours was the original choice and gw2 2026-27 is the evidence it
    wasn't enough: `.github/workflows/papertrade.yml`'s two-hourly cron
    went 2026-08-28T06:00:59Z -> 2026-08-28T20:05:02Z, a 14.1-hour gap,
    entirely swallowing gw2's 11:30-17:30 UTC window (deadline 17:30) with
    no run landing inside it. GitHub deprioritizes/drops `schedule`
    triggers on low-activity repos, so the real inter-run gap is not the
    cron's nominal cadence -- observed gaps across this workflow's history
    routinely run 3-14 hours. Once a gameweek's deadline passes with no
    freeze on disk, `resolve_next_event` has already moved on and
    `assert_before_deadline` correctly refuses to freeze it late, so a
    missed window is not a retryable failure -- it is a permanently lost
    gameweek for §6.5's evaluation. 24 hours gives several retry slots even
    against the worst observed gaps, at the cost of the shadow team's
    team-news freshness being at most a day stale rather than freezing
    right at the deadline -- still nowhere near the week-stale failure mode
    above.
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
    manual_correction: str | None = None,
) -> Path:
    """Freezes gameweek `gw` (defaults to bootstrap-static's resolved next
    event) for the shadow team, advancing it from whatever the previous
    freeze file (or, for gw2, the real gw1 squad) left it in.

    Transfers are capped at the free allowance until the model has
    HIT_ELIGIBILITY_GWS gameweeks of current-season history; the cap and
    the evidence for it are recorded in the freeze under `transfer_cap`
    whether or not it bound.

    `manual_correction` records, permanently and in the gameweek's own
    freeze, that a human intervened in this run — §6.5 criterion 4 asks
    for 13 consecutive gameweeks *without* one. It is a free-text reason
    rather than a boolean so the record says what was corrected, and it
    defaults to None, which is the positive claim that nothing was.
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

    # §6.5 criterion 3, on the live path rather than only the historical
    # walk-forward harness. This raises rather than warns (see
    # backtest/leakage.py) — a freeze built on a feature that was not
    # available before its own deadline is not a prediction, and writing
    # it would put a permanently unusable record into an immutable file.
    #
    # Non-retrofittable: it is a claim about what was true *during* the
    # live period, so it has to run at freeze time or not at all.
    leakage_features = training_feature_availability(train_df, bootstrap, fixtures_raw)
    assert_no_leakage(leakage_features, gw_deadline, context=f"live freeze gw{gw}")
    latest_feature_at = max((f.available_at for f in leakage_features), default=None)
    logger.info(
        "leakage check passed for gw%d: %d feature(s), latest available_at=%s, deadline=%s",
        gw, len(leakage_features), latest_feature_at, gw_deadline,
    )

    difficulty_table = build_difficulty_table(bootstrap, fixtures_raw, horizon)
    scoring_config = load_scoring_config(Path(scoring_config_path))
    projections = build_projections(train_df, target_roster, scoring_config, difficulty_table, horizon)

    # Cap transfers at the free allowance while the history is too thin for
    # the model to be paying hits on it (see HIT_ELIGIBILITY_GWS). Recorded
    # in the payload either way, so a reader can tell a held squad that the
    # optimizer chose from one it was not allowed to change.
    n_train_gws = int(train_df["gw"].n_unique()) if train_df.height else 0
    cap = transfer_cap(n_train_gws, free_transfers)
    if cap["applied"]:
        logger.info(
            "gw%d: %d gameweek(s) of history < %d, capping transfers at the free allowance (%d) so no hits are paid",
            gw, n_train_gws, HIT_ELIGIBILITY_GWS, cap["max_transfers"],
        )

    result = optimize_squad(
        shadow_state, pool, projections, horizon=horizon,
        free_transfers=free_transfers, max_transfers=cap["max_transfers"], hit_cost=hit_cost,
    )

    shadow_state_after = apply_recommendation(shadow_state, result, pool_by_id, now_cost_by_id, next_gw=gw, as_of=now)
    free_transfers_after = accrue_free_transfers(free_transfers, len(result.transfers_out), scoring_config["free_transfers"]["max_banked"])
    deadline = gw_deadline

    payload = {
        "gameweek": gw,
        "frozen_at": now.isoformat(),
        "deadline": deadline.isoformat(),
        "horizon": horizon,
        # Provenance and the two §6.5 criteria that can only be established
        # at freeze time. All three are non-retrofittable: freezes are
        # immutable (§6.1), so a field absent here is absent forever for
        # this gameweek, and every one of these is a claim about what was
        # true when the prediction was made rather than about the data.
        "model_git_sha": model_git_sha(),
        "git_dirty": git_is_dirty(),
        "leakage_check": {
            "ran": True,
            "passed": True,  # assert_no_leakage raises, so reaching here means it passed
            "n_features": len(leakage_features),
            "latest_feature_available_at": latest_feature_at.isoformat() if latest_feature_at else None,
            "deadline": gw_deadline.isoformat(),
        },
        # §6.5 criterion 4. Human-declared, because a manual correction is
        # by definition something a human did outside the pipeline and no
        # automated check can observe it. `None` is the claim that this
        # gameweek's squad reconstruction ran untouched.
        "manual_correction": manual_correction,
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
        # Whether the optimizer was free to pay hits this gameweek, and on
        # what evidence. Without this a capped hold and a chosen hold are
        # indistinguishable in the record, and §6.3's squad-level series
        # would be read as the optimizer's judgement when it was a
        # constraint (see HIT_ELIGIBILITY_GWS).
        "transfer_cap": cap,
        "free_transfers_after": free_transfers_after,
        "shadow_state_after": squad_state_to_dict(shadow_state_after),
    }
    return write_freeze(gw, payload, freezes_dir)
