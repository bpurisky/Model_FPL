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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl

from analytics.fdr import compute_elo_ratings, upcoming_team_difficulty
from analytics.projections import project_points
from analytics.scoring import load_scoring_config
from backtest.backfill import RAW_CACHE_DIR, load_match_results, load_teams
from backtest.leakage import Feature
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

# Per-gw stat columns. Read straight off bootstrap-static's raw element
# dicts when a gameweek has to be reconstructed (the trimmed `Element`
# pydantic model in collector/schemas.py deliberately only covers Phase 0's
# distilled-trending columns, §2.3, not these).
#
# Every column here must be *additive over gameweeks* — bootstrap-static
# reports each as a season-cumulative total and `_reconstruct_gw` recovers
# a single gameweek by subtracting the recorded ones, which is only exact
# for a quantity that sums. Verified against the live API on 2026-08-22:
# for all 22 columns, cumulative equalled gw1 exactly while gw1 was the
# only gameweek played. Do not add a column here without checking that.
#
# The first fourteen are what analytics/projections.py's trailing-rate
# heads and backtest/baselines.py consume. The rest are recorded but not
# yet modelled: they are what Phase 5's export contract needs at
# player-gameweek grain, and the store is append-only, so a column not
# captured as each gameweek finishes costs a rewrite of an immutable file
# to recover later. Widening is cheap now and expensive after gw1 lands.
INT_STAT_COLUMNS = [
    "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded", "saves", "bonus",
    "yellow_cards", "red_cards", "own_goals", "penalties_missed", "penalties_saved", "defensive_contribution",
    "total_points",
    "starts", "bps", "clearances_blocks_interceptions", "recoveries", "tackles",
]

# The same, for the columns FPL serves as decimal *strings* on both
# endpoints — `float()` at the boundary, `pl.Float64` in the frame.
#
# NOTE on the ICT four: as of 2026-08-22 FPL publishes 0.0 for influence,
# creativity, threat and ict_index for all 604 elements, on both
# bootstrap-static and /event/{gw}/live/ — the index appears to be
# unpopulated this season, where xG/xA/bps are populated normally. They are
# recorded anyway, verbatim, so that a mid-season change is captured from
# the gameweek it happens rather than not at all. Anything reading them
# must treat 0.0 here as "FPL published nothing", not as a measured zero.
FLOAT_STAT_COLUMNS = [
    "expected_goals", "expected_assists", "expected_goal_involvements", "expected_goals_conceded",
    "influence", "creativity", "threat", "ict_index",
]

STAT_COLUMNS = INT_STAT_COLUMNS + FLOAT_STAT_COLUMNS

# Stats that can legitimately be negative, and so must never be floored at
# zero by `_reconstruct_gw`'s revision clamp.
#
# BPS carries explicit penalties -- yellow -3, red -9, own goal -6, error
# leading to a goal -3 -- so a player who came on, got booked and did
# little else genuinely ends the gameweek below zero. Verified live
# 2026-08-22: eight players held a negative *season* BPS after gw1 alone.
# FPL's own scoring does the same to total_points (own goal -2, red -3).
#
# Everything else in STAT_COLUMNS is a count or a non-negative rate, where
# a negative delta cannot be real and does indicate a revision.
SIGNED_STAT_COLUMNS = frozenset({"bps", "total_points"})

# How far below zero a reconstructed float delta may fall before it counts
# as a real FPL revision rather than binary-floating-point noise. FPL
# publishes these to two decimal places, so anything this small is
# arithmetic, not football. See `_reconstruct_gw`.
_FLOAT_CLAMP_EPS = 1e-6

# How long after kickoff a fixture's stats are treated as knowable, for
# the point-in-time assertion in `training_feature_availability`.
#
# Direction of error matters here and it is not symmetric. Too *small* a
# value dates a feature earlier than it really was available, and the
# leakage check then passes something it should have caught. Too large
# only risks a false alarm on a fixture that kicked off shortly before a
# deadline — loud, and reviewable. Two hours is a full match plus
# half-time and stoppage; provisional bonus lands later still, so this is
# deliberately at the strict end of "the match is over."
MATCH_DURATION = timedelta(hours=2)

# The schema of both `build_train_df`'s output and the actuals store it
# reads (papertrade/actuals.py imports this). One definition rather than
# two because build_train_df concatenates rows straight off that file with
# rows it reconstructs here, so the two must agree column-for-column;
# it lives in this module only because papertrade already depends on
# squad.live and the reverse would be circular.
TRAIN_SCHEMA = {
    "gw": pl.Int64, "element_id": pl.Int64, "position": pl.Utf8, "team": pl.Utf8, "is_promoted_club": pl.Boolean,
    **{c: pl.Int64 for c in INT_STAT_COLUMNS},
    **{c: pl.Float64 for c in FLOAT_STAT_COLUMNS},
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
        for col in INT_STAT_COLUMNS:
            value = cumulative[col] - already.get(col, 0)
            if value < 0 and col not in SIGNED_STAT_COLUMNS:
                # FPL revises finished gameweeks after the fact (the
                # dubious goals panel, bonus recalculation), so a total
                # already written to the append-only store can end up
                # larger than the current cumulative one. A negative event
                # count is not a thing; 0 is the only honest floor, and
                # the revision is by construction small.
                clamped += 1
                value = 0
            stats[col] = value
        for col in FLOAT_STAT_COLUMNS:
            # FPL serves these as decimal strings ("0.28"); parse before
            # subtracting or this silently concatenates.
            value = float(cumulative[col]) - already.get(col, 0.0)
            if value < 0.0:
                # Same revision case as above, but the floor is applied on
                # a tolerance rather than at exactly 0: subtracting two
                # 2dp values parsed from strings routinely lands a hair
                # below zero (~1e-16) with nothing revised at all, and
                # counting those would make the warning cry wolf on every
                # single run. Only a real revision clears FLOAT_CLAMP_EPS.
                if value < -_FLOAT_CLAMP_EPS:
                    clamped += 1
                value = 0.0
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


def training_feature_availability(
    train_df: pl.DataFrame,
    bootstrap: BootstrapStatic,
    fixtures_raw: list[dict[str, Any]],
    match_duration: timedelta = MATCH_DURATION,
) -> list[Feature]:
    """One `backtest.leakage.Feature` per (element_id, gw) row of
    `train_df`, carrying the moment that row's stats first became
    knowable: the end of that player's team's last fixture in that
    gameweek.

    Per *player* rather than per gameweek because teams within one
    gameweek kick off days apart, and a Monday-night fixture is genuinely
    available later than a Saturday-lunchtime one. Per-player also makes
    `assert_no_leakage`'s error message name the offending element.

    Why this is worth asserting when `build_train_df` already filters
    `gw <= target - 1`: that filter is about gameweek *numbering*, and
    numbering is not chronology. A postponed fixture from gameweek 5 can
    be replayed after gameweek 6's deadline, at which point a row labelled
    "gw5" contains a match that had not been played when the prediction
    was made. The gameweek filter cannot see that; a timestamp can. This
    is the case the assertion exists for, and it is not hypothetical in a
    league that rearranges fixtures for cup and European ties.

    A player whose team has no fixture at all in a gameweek gets no
    Feature — there is nothing to have leaked.
    """
    team_id_by_name = {t.name: t.id for t in bootstrap.teams}

    # team id -> gw -> latest kickoff among that team's fixtures that gw.
    latest_kickoff: dict[tuple[int, int], datetime] = {}
    for fixture in fixtures_raw:
        event, kickoff = fixture.get("event"), fixture.get("kickoff_time")
        if event is None or not kickoff:
            # An unscheduled fixture (kickoff_time null) cannot be shown to
            # have been played before any deadline, so it is skipped rather
            # than assumed safe. It carries no stats yet either, so no row
            # in train_df depends on it.
            continue
        ko = kickoff if isinstance(kickoff, datetime) else datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
        for team in (fixture["team_h"], fixture["team_a"]):
            key = (team, event)
            if key not in latest_kickoff or ko > latest_kickoff[key]:
                latest_kickoff[key] = ko

    features: list[Feature] = []
    for row in train_df.select(["element_id", "gw", "team"]).to_dicts():
        team_id = team_id_by_name.get(row["team"])
        if team_id is None:
            continue
        kickoff = latest_kickoff.get((team_id, row["gw"]))
        if kickoff is None:
            continue
        features.append(
            Feature(
                name=f"train_df[gw{row['gw']}]",
                element_id=row["element_id"],
                value=None,
                available_at=kickoff + match_duration,
            )
        )
    return features


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
