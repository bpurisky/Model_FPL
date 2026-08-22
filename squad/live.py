"""Live data bridge for `python -m squad recommend` (§5's phase goal:
"given a team ID, reconstruct squad state and recommend the highest-value
legal move"). Fetches straight from the FPL API rather than reading Phase
0's collected raw/distilled snapshots — those live on whatever schedule the
hourly collector runs on and `data/raw/` is gitignored (§2.3: insurance, not
archive), so there's no guarantee a fresh-enough own-entry snapshot exists
on disk when a recommendation is asked for right now. `squad/reconstruct.py`
and `squad/transfers.py` only need parsed payload objects, which this module
gets directly, without a round trip through disk.

analytics/projections.py's trailing-window machinery (analytics/features.py)
expects one row per (element_id, gw) so it can take the last N gameweeks.
`build_train_df` sources those per-gameweek splits from
`data/current_season/2026-27.parquet` — the append-only actuals store
`papertrade/actuals.py` writes from `/event/{gw}/live/`, which reports each
gameweek's own stats rather than a season-cumulative total.

Bootstrap-static is still read, but only for the one gameweek that store
cannot yet hold: the most recent one, which at freeze time is typically
played but not yet confirmed (`record-actuals` gates on event-level
`finished`, which lags full time by many hours). That gameweek is recovered
exactly, as cumulative-so-far minus the sum of the gameweeks already
recorded — no blending, and no 600-request `/element-summary/` batch.

Until 2026-08-22 this module read bootstrap-static's cumulative totals as if
they were gw1's, which was exact only while gw1 was the only gameweek in
existence; the moment gw2 completed it would have silently blended two
gameweeks into one row and defeated the trailing WINDOW itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from analytics.fdr import compute_elo_ratings, upcoming_team_difficulty
from analytics.projections import project_points
from analytics.scoring import load_scoring_config
from backtest.backfill import RAW_CACHE_DIR, load_match_results, load_teams
from collector.client import FPLClient
from collector.config import CollectorConfig
from collector.schemas import (
    BootstrapStatic,
    fixture_is_played,
    parse_bootstrap_static,
    parse_entry_history,
    parse_entry_picks,
    parse_entry_transfers,
    parse_fixtures,
    resolve_current_event,
    resolve_next_event,
)
from squad.optimize import Player
from squad.reconstruct import SquadState, reconstruct_squad
from squad.transfers import derive_free_transfers

logger = logging.getLogger("squad.live")

POSITION_BY_ELEMENT_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# config/promoted_clubs.yaml's "2026-27" entry, by short_name rather than
# full name — bootstrap-static's team names ("Coventry City") don't match
# that config file's vaastav-sourced short forms ("Coventry"), but
# short_name codes are unambiguous either way.
PROMOTED_CLUB_SHORT_NAMES = {"COV", "HUL", "IPS"}

_ELO_BOOTSTRAP_SEASON = "2025-26"

# Per-gw stat columns needed by analytics/projections.py's trailing-rate
# heads, plus total_points for backtest/baselines.py. Read straight off
# bootstrap-static's raw element dicts when a gameweek has to be
# reconstructed (the trimmed `Element` pydantic model in
# collector/schemas.py deliberately only covers Phase 0's
# distilled-trending columns, §2.3, not these).
STAT_COLUMNS = [
    "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded", "saves", "bonus",
    "yellow_cards", "red_cards", "own_goals", "penalties_missed", "penalties_saved", "defensive_contribution",
    "total_points",
]

# The schema of both `build_train_df`'s output and the actuals store it
# reads (papertrade/actuals.py imports this). One definition rather than
# two because build_train_df concatenates rows straight off that file with
# rows it reconstructs here, so the two must agree column-for-column;
# it lives in this module only because papertrade already depends on
# squad.live and the reverse would be circular.
TRAIN_SCHEMA = {
    "gw": pl.Int64, "element_id": pl.Int64, "position": pl.Utf8, "team": pl.Utf8, "is_promoted_club": pl.Boolean,
    **{c: pl.Int64 for c in STAT_COLUMNS},
}


@dataclass(frozen=True)
class LiveData:
    squad: SquadState
    pool: list[Player]
    free_transfers: int
    current_event: int
    next_event: int
    train_df: pl.DataFrame
    target_roster: pl.DataFrame
    difficulty_table: pl.DataFrame
    scoring_config: dict[str, Any]
    ownership: dict[int, float]
    web_names: dict[int, str]
    now_cost_by_id: dict[int, int]
    data_gw: int
    teams_with_played_data: int
    teams_total: int
    # Distinct gameweeks actually present in train_df — not `data_gw`,
    # which is only the latest one. They differ whenever a `record-actuals`
    # run was missed, and the gap is exactly what makes a projection
    # thinner than it looks (see build_train_df's missing-gameweek path).
    history_gws: int


def _team_name(bootstrap: BootstrapStatic, team_id: int) -> str:
    return next(t.name for t in bootstrap.teams if t.id == team_id)


def _team_short_name(bootstrap: BootstrapStatic, team_id: int) -> str:
    return next(t.short_name for t in bootstrap.teams if t.id == team_id)


def build_target_roster(bootstrap: BootstrapStatic) -> pl.DataFrame:
    """element_id, position, team (name), is_promoted_club — the shape
    analytics/projections.py's target_roster needs."""
    rows = [
        {
            "element_id": e.id,
            "position": POSITION_BY_ELEMENT_TYPE[e.element_type],
            "team": _team_name(bootstrap, e.team),
            "is_promoted_club": _team_short_name(bootstrap, e.team) in PROMOTED_CLUB_SHORT_NAMES,
        }
        for e in bootstrap.elements
    ]
    return pl.DataFrame(rows)


def _recorded_totals(history: pl.DataFrame, before_gw: int) -> dict[int, dict[str, int]]:
    """element_id -> its stats summed over every recorded gameweek before
    `before_gw`. This is exactly the quantity subtracted out of
    bootstrap-static's season-cumulative totals to recover `before_gw`'s
    own stats, so it must cover every earlier gameweek or not be used at
    all — `build_train_df` checks that before calling this."""
    prior = history.filter(pl.col("gw") < before_gw)
    if prior.height == 0:
        return {}
    summed = prior.group_by("element_id").agg([pl.col(col).sum() for col in STAT_COLUMNS])
    return {row["element_id"]: row for row in summed.to_dicts()}


def _reconstruct_gw(
    bootstrap: BootstrapStatic,
    bootstrap_raw: dict[str, Any],
    fixtures_raw: list[dict[str, Any]],
    gw: int,
    history: pl.DataFrame,
) -> pl.DataFrame:
    """`gw`'s own per-player stats, derived as bootstrap-static's
    season-cumulative totals minus everything already recorded for the
    gameweeks before it. Only teams whose gw-`gw` fixture has actually been
    played contribute a row — see `build_train_df` for why that gate
    exists and why it is not the raw `finished` flag."""
    played_teams = {
        team
        for f in fixtures_raw
        if f["event"] == gw and fixture_is_played(f)
        for team in (f["team_h"], f["team_a"])
    }
    team_by_element = {e.id: e.team for e in bootstrap.elements}
    cumulative_by_id = {el["id"]: el for el in bootstrap_raw["elements"]}
    recorded = _recorded_totals(history, gw)

    rows = []
    clamped = 0
    for row in build_target_roster(bootstrap).to_dicts():
        element_id = row["element_id"]
        if team_by_element[element_id] not in played_teams:
            continue
        cumulative = cumulative_by_id[element_id]
        already = recorded.get(element_id, {})
        stats = {}
        for col in STAT_COLUMNS:
            value = cumulative[col] - already.get(col, 0)
            if value < 0:
                # FPL revises finished gameweeks after the fact (the
                # dubious goals panel, bonus recalculation), so a total
                # already written to the append-only store can end up
                # larger than the current cumulative one. A negative event
                # count is not a thing; 0 is the only honest floor, and
                # the revision is by construction small.
                clamped += 1
                value = 0
            stats[col] = value
        rows.append({"gw": gw, **row, **stats})

    if clamped:
        logger.warning(
            "gw%d: clamped %d negative stat deltas to 0 — a gameweek already in the actuals "
            "store appears to have been revised by FPL since it was recorded",
            gw, clamped,
        )
    return pl.DataFrame(rows, schema=TRAIN_SCHEMA)


def build_train_df(
    bootstrap: BootstrapStatic,
    bootstrap_raw: dict[str, Any],
    fixtures_raw: list[dict[str, Any]],
    gw: int,
    actuals: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """This season's history as one row per (element_id, gameweek), for
    every gameweek up to and including `gw`.

    Gameweeks the actuals store has already recorded are read from it
    verbatim. `gw` itself usually has not been — `record-actuals` gates on
    event-level `finished`, which lags full time by many hours, so at
    freeze time the latest gameweek is typically played but unconfirmed.
    That one gameweek is reconstructed from bootstrap-static's cumulative
    totals minus the recorded ones (`_reconstruct_gw`), which is exact
    whenever every earlier gameweek is in the store. When it is not — a
    missed `record-actuals` run — the gameweek is dropped rather than
    guessed at, because subtracting an incomplete history would silently
    fold two gameweeks into one row: the failure mode this function was
    rewritten to remove, and one that produces no error at all.

    Only teams whose gw-`gw` fixture has actually been played contribute a
    reconstructed row. "Played" is `collector.schemas.fixture_is_played`,
    not the raw `finished` flag — read that function's docstring before
    changing this, because gating on `finished` alone is precisely the bug
    that made every player fall through to the pooled prior. Excluding
    not-yet-played teams matters: a gameweek in progress reports
    `minutes: 0` for anyone whose fixture hasn't kicked off yet,
    indistinguishable in the raw data from a genuine blank -- verified live
    (2026-08-21) against a real case, Haaland showing `minutes: 0` purely
    because Man City's gw1 fixture hadn't started, which would otherwise
    crater his minutes-distribution projection on fabricated absence. An
    excluded player now falls through to his *own* earlier gameweeks, and
    only to the pooled-prior fallback
    (`analytics.features.fill_missing_with_pooled_prior`) if he has none.

    Anything after `gw` is filtered out rather than assumed absent:
    `papertrade/freeze.py` calls this with `gw = target_gw - 1`, so this
    filter is the last thing standing between a late-arriving actuals row
    and training on the very gameweek being predicted (§6.5's leakage
    criterion).
    """
    if actuals is None:
        # Deferred import: papertrade.actuals imports this module's
        # POSITION_BY_ELEMENT_TYPE/PROMOTED_CLUB_SHORT_NAMES/TRAIN_SCHEMA,
        # so importing it at module scope here would be circular. Only the
        # default path needs it; every caller under test passes a frame.
        from papertrade.actuals import load_actuals

        actuals = load_actuals()

    history = actuals.filter(pl.col("gw") <= gw).select(list(TRAIN_SCHEMA)).sort(["element_id", "gw"])
    recorded_gws = set(history["gw"].to_list())

    if gw in recorded_gws:
        return history

    missing = [g for g in range(1, gw) if g not in recorded_gws]
    if missing:
        logger.warning(
            "gw%d is not in the actuals store and gameweek(s) %s are missing before it, so it "
            "cannot be recovered from cumulative totals without double-counting — omitting it. "
            "Projections fall back to gameweeks %s, or the pooled prior where a player has none. "
            "Run `python -m papertrade record-actuals` and retry.",
            gw, missing, sorted(recorded_gws) or "(none)",
        )
        return history

    reconstructed = _reconstruct_gw(bootstrap, bootstrap_raw, fixtures_raw, gw, history)
    return pl.concat([history, reconstructed]).sort(["element_id", "gw"])


def build_player_pool(bootstrap: BootstrapStatic) -> list[Player]:
    return [
        Player(
            element_id=e.id,
            position=POSITION_BY_ELEMENT_TYPE[e.element_type],
            club=_team_short_name(bootstrap, e.team),
            now_cost=e.now_cost,
        )
        for e in bootstrap.elements
    ]


def build_elo_final_for_current_teams(bootstrap: BootstrapStatic, season: str = _ELO_BOOTSTRAP_SEASON) -> dict[int, float]:
    """Elo carried over from the most recent completed season, remapped
    onto *this* season's numeric team ids. FPL reassigns team ids each
    season (promotion/relegation shuffles the alphabetical numbering,
    §3.2's Coventry/Hull/Ipswich among them) so `season`'s Elo — keyed by
    that season's own ids — is bridged through team NAME, the one thing
    that's stable across seasons, rather than trusted to line up by id.
    A team absent from `season` (newly promoted) is simply absent from the
    returned dict; `upcoming_team_difficulty` already treats that as
    "no history yet" and falls back to the neutral initial rating.
    """
    matches = load_match_results(RAW_CACHE_DIR / season / "fixtures.csv")
    old_teams = load_teams(RAW_CACHE_DIR / season / "teams.csv")
    elo_final = compute_elo_ratings(matches).final
    old_id_to_name = dict(zip(old_teams["id"].to_list(), old_teams["name"].to_list()))
    elo_by_name = {old_id_to_name[tid]: rating for tid, rating in elo_final.items() if tid in old_id_to_name}
    return {t.id: elo_by_name[t.name] for t in bootstrap.teams if t.name in elo_by_name}


def build_difficulty_table(bootstrap: BootstrapStatic, fixtures_raw: list[dict[str, Any]], horizon: list[int]) -> pl.DataFrame:
    elo_final = build_elo_final_for_current_teams(bootstrap)
    fixtures_df = pl.DataFrame(
        [{"event": f["event"], "team_h": f["team_h"], "team_a": f["team_a"]} for f in fixtures_raw if f["event"] in horizon]
    )
    teams_df = pl.DataFrame([{"id": t.id, "name": t.name} for t in bootstrap.teams])
    return upcoming_team_difficulty(elo_final, fixtures_df, teams_df)


async def fetch_live_data(
    cfg: CollectorConfig,
    entry_id: int,
    horizon: list[int] | None = None,
    scoring_config_path: str = "config/scoring_2026_27.yaml",
) -> LiveData:
    async with FPLClient(**cfg.api.client_kwargs()) as client:
        bootstrap_raw = await client.get_json("/bootstrap-static/")
        fixtures_raw = await client.get_json("/fixtures/")
        entry_history_raw = await client.get_json(f"/entry/{entry_id}/history/")
        transfers_raw = await client.get_json(f"/entry/{entry_id}/transfers/")

        bootstrap = parse_bootstrap_static(bootstrap_raw, logger)
        parse_fixtures(fixtures_raw, logger)  # validated for drift; raw dicts used below for event/team_h/team_a
        entry_history = parse_entry_history(entry_history_raw, logger)
        transfers = parse_entry_transfers(transfers_raw, logger)

        current_event = resolve_current_event(bootstrap, bootstrap_raw)
        next_event = resolve_next_event(bootstrap, bootstrap_raw)
        if current_event is None or next_event is None:
            raise RuntimeError("could not resolve current/next gameweek from bootstrap-static")
        if not entry_history.current:
            raise RuntimeError(f"entry {entry_id} has no gameweek history yet (has gw1's deadline passed?)")
        last_played_event = max(h.event for h in entry_history.current)

        picks_raw = await client.get_json(f"/entry/{entry_id}/event/{last_played_event}/picks/")
        picks = parse_entry_picks(picks_raw, logger)

    now_cost_by_id = {e.id: e.now_cost for e in bootstrap.elements}
    ownership = {e.id: float(e.selected_by_percent) for e in bootstrap.elements}
    web_names = {e.id: e.web_name for e in bootstrap.elements}

    picks_snapshot_ts = next(ev.deadline_time for ev in bootstrap.events if ev.id == last_played_event)
    as_of = datetime.now(timezone.utc)
    squad = reconstruct_squad(picks, picks_snapshot_ts, transfers, as_of, current_prices=now_cost_by_id)

    ft_states = derive_free_transfers(
        list(transfers), entry_history.chips, through_event=next_event,
        max_banked=load_scoring_config(Path(scoring_config_path))["free_transfers"]["max_banked"],
    )
    free_transfers = ft_states[-1].available_before if ft_states else 1

    horizon = horizon or [next_event, next_event + 1, next_event + 2]
    train_df = build_train_df(bootstrap, bootstrap_raw, fixtures_raw, gw=last_played_event)
    target_roster = build_target_roster(bootstrap)
    difficulty_table = build_difficulty_table(bootstrap, fixtures_raw, horizon)
    scoring_config = load_scoring_config(Path(scoring_config_path))

    gw_teams = {t for f in fixtures_raw if f["event"] == last_played_event for t in (f["team_h"], f["team_a"])}
    played_teams = {
        t
        for f in fixtures_raw
        if f["event"] == last_played_event and fixture_is_played(f)
        for t in (f["team_h"], f["team_a"])
    }

    return LiveData(
        squad=squad,
        pool=build_player_pool(bootstrap),
        free_transfers=free_transfers,
        current_event=current_event,
        next_event=next_event,
        train_df=train_df,
        target_roster=target_roster,
        difficulty_table=difficulty_table,
        scoring_config=scoring_config,
        ownership=ownership,
        web_names=web_names,
        now_cost_by_id=now_cost_by_id,
        data_gw=last_played_event,
        teams_with_played_data=len(played_teams),
        teams_total=len(gw_teams),
        history_gws=train_df["gw"].n_unique(),
    )


def build_projections(
    train_df: pl.DataFrame,
    target_roster: pl.DataFrame,
    scoring_config: dict[str, Any],
    difficulty_table: pl.DataFrame,
    horizon: list[int],
) -> dict[int, dict[int, float]]:
    """Split out of `LiveData` on purpose — `papertrade/freeze.py` needs to
    project points for the shadow team's own state, which has no `LiveData`
    of its own (that struct is specific to the real, live-fetched entry)."""
    projections: dict[int, dict[int, float]] = {}
    for gw in horizon:
        df = project_points(train_df, target_roster, gw, scoring_config, difficulty_table)
        projections[gw] = dict(zip(df["element_id"].to_list(), df["prediction"].to_list()))
    return projections
