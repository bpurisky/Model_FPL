"""Live data bridge for `python -m squad recommend` (§5's phase goal:
"given a team ID, reconstruct squad state and recommend the highest-value
legal move"). Fetches straight from the FPL API rather than reading Phase
0's collected raw/distilled snapshots — those live on whatever schedule the
hourly collector runs on and `data/raw/` is gitignored (§2.3: insurance, not
archive), so there's no guarantee a fresh-enough own-entry snapshot exists
on disk when a recommendation is asked for right now. `squad/reconstruct.py`
and `squad/transfers.py` only need parsed payload objects, which this module
gets directly, without a round trip through disk.

Known limitation, not fixed here: analytics/projections.py's trailing-window
machinery (analytics/features.py) expects one row per (element_id, gw) so it
can take the last N gameweeks. Only gameweek 1 has any 2026/27 result at
all as of this module's first use, so `build_train_df` sources that single
row from bootstrap-static's per-element season-cumulative stats — exact
while only one gameweek exists, because cumulative-so-far *is* gw1's total.
Once gw2 finishes, cumulative totals stop being safe to reuse this way
(they'd blend multiple gameweeks into one row and defeat the trailing
WINDOW itself) — switch to per-gameweek splits from
`/element-summary/{id}/`'s `history`, batched via
`collector.snapshot.run_element_summary_snapshot`'s existing rate-limited
fetch pattern, before running this again after gw2.
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
# heads, read straight off bootstrap-static's raw element dicts (the
# trimmed `Element` pydantic model in collector/schemas.py deliberately
# only covers Phase 0's distilled-trending columns, §2.3, not these).
_STAT_COLUMNS = [
    "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded", "saves", "bonus",
    "yellow_cards", "red_cards", "own_goals", "penalties_missed", "penalties_saved", "defensive_contribution",
]


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
    teams_with_finished_data: int
    teams_total: int


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


def build_train_df(
    bootstrap: BootstrapStatic, bootstrap_raw: dict[str, Any], fixtures_raw: list[dict[str, Any]], gw: int
) -> pl.DataFrame:
    """One row per element for `gw`, for teams whose gw-`gw` fixture has
    actually finished (see module docstring's limitation: this whole
    function is only correct while `gw` is the only gameweek that exists
    at all this season). Excluding not-yet-played teams matters: a gameweek
    in progress reports `minutes: 0` for anyone whose fixture hasn't
    kicked off yet, indistinguishable in the raw data from a genuine blank
    -- verified live (2026-08-21) against a real case, Haaland showing
    `minutes: 0` purely because Man City's gw1 fixture hadn't started,
    which would otherwise crater his minutes-distribution projection on
    fabricated absence. Excluded players fall through to the pooled-prior
    fallback (`analytics.features.fill_missing_with_pooled_prior`) instead
    of a false zero.
    """
    finished_teams = {
        team for f in fixtures_raw if f["event"] == gw and f["finished"] for team in (f["team_h"], f["team_a"])
    }
    target_roster = build_target_roster(bootstrap)
    team_by_element = {e.id: e.team for e in bootstrap.elements}
    stats_by_id = {el["id"]: el for el in bootstrap_raw["elements"]}
    rows = [
        {**row, "gw": gw, **{col: stats_by_id[row["element_id"]][col] for col in _STAT_COLUMNS}}
        for row in target_roster.to_dicts()
        if team_by_element[row["element_id"]] in finished_teams
    ]
    schema = {"element_id": pl.Int64, "position": pl.Utf8, "team": pl.Utf8, "is_promoted_club": pl.Boolean, "gw": pl.Int64}
    schema.update({c: pl.Int64 for c in _STAT_COLUMNS})
    return pl.DataFrame(rows, schema=schema)


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
    finished_teams = {
        t for f in fixtures_raw if f["event"] == last_played_event and f["finished"] for t in (f["team_h"], f["team_a"])
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
        teams_with_finished_data=len(finished_teams),
        teams_total=len(gw_teams),
    )


def build_projections(live: LiveData, horizon: list[int]) -> dict[int, dict[int, float]]:
    projections: dict[int, dict[int, float]] = {}
    for gw in horizon:
        df = project_points(live.train_df, live.target_roster, gw, live.scoring_config, live.difficulty_table)
        projections[gw] = dict(zip(df["element_id"].to_list(), df["prediction"].to_list()))
    return projections
