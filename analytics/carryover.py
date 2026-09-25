"""Last season's rows, re-keyed so this season's trailing windows can reach
back into them.

Each season used to start from nothing: element ids are reassigned every
summer, so a player's gw2 projection rested on one game, and every
trailing window stayed short of its nominal length until gw5 or later —
the stretch of the season where a manager's early transfers are made.
FPL's `code` is stable across seasons (backtest.backfill.backfill_player_codes
for the history, bootstrap-static for the live season), so the previous
season's rows can be renamed to this season's element ids and placed
*before* gw1, where `features.trailing_mean`'s "last N rows" naturally
picks them up and lets them age out as current games arrive.

Point-in-time by construction: every carried row is a match from a season
that finished before this one started.

Player-level rates (minutes, goals, xG, bonus, cards) carry across a
transfer; team-level ones do not. A defender's clean sheets and goals
conceded, and a keeper's saves, describe the back line he played behind,
so those are nulled on carried rows whose club differs from the player's
current one. `polars` means skip nulls, so such a player's team-level
rate comes from this season's games alone (or the pooled prior before he
has any), while his row count — and so the reach of the window — is kept.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from backtest.backfill import NORMALIZED_DIR, PLAYER_CODES_PATH

logger = logging.getLogger("analytics.carryover")

# bootstrap-static's `code`, written by the collector alongside id/team.
# The current season's element_id -> code comes from here, since the
# archive's player_codes.parquet only covers seasons vaastav has published.
REFERENCE_PLAYERS_PATH = Path("data/reference/players.parquet")

# Rates that describe the player's team rather than the player.
TEAM_LEVEL_COLUMNS = ["clean_sheets", "goals_conceded", "saves", "expected_goals_conceded"]


def carry_forward(
    prev_season_df: pl.DataFrame,
    prev_codes: pl.DataFrame,
    current_codes: pl.DataFrame,
    current_teams: pl.DataFrame,
) -> pl.DataFrame:
    """`prev_season_df` re-keyed to the current season.

    `prev_codes` / `current_codes`: element_id -> code for each season.
    `current_teams`: element_id -> team, this season.

    Returns the previous season's rows for players who are in the current
    season, with `element_id` replaced by the current one and `gw` shifted
    to <= 0 (last season's final gameweek becomes gw 0), and team-level
    columns nulled where the player has since changed club. Players new to
    the league have no rows and fall through to the pooled prior as before.
    """
    last_gw = prev_season_df["gw"].max() or 0
    mapping = (
        prev_codes.rename({"element_id": "prev_element_id"})
        .join(current_codes, on="code", how="inner")
        .join(current_teams.rename({"team": "current_team"}), on="element_id", how="inner")
        .select("prev_element_id", "element_id", "current_team")
    )
    carried = (
        prev_season_df.rename({"element_id": "prev_element_id"})
        .join(mapping, on="prev_element_id", how="inner")
        .drop("prev_element_id")
        .with_columns((pl.col("gw") - last_gw).alias("gw"))
    )
    moved = pl.col("team") != pl.col("current_team")
    team_cols = [c for c in TEAM_LEVEL_COLUMNS if c in carried.columns]
    return carried.with_columns(
        [pl.when(moved).then(None).otherwise(pl.col(c)).alias(c) for c in team_cols]
    ).drop("current_team")


def previous_season(season: str) -> str:
    """"2025-26" -> "2024-25"."""
    start = int(season[:4]) - 1
    return f"{start}-{(start + 1) % 100:02d}"


def season_codes(season: str, reference_players: Path = REFERENCE_PLAYERS_PATH) -> pl.DataFrame | None:
    """element_id -> code for `season`: the archive's mapping if it has the
    season, else the collector's reference table (the live season), else
    None."""
    if PLAYER_CODES_PATH.exists():
        archive = pl.read_parquet(PLAYER_CODES_PATH).filter(pl.col("season") == season)
        if archive.height:
            return archive.select("element_id", "code")
    if reference_players.exists():
        reference = pl.read_parquet(reference_players)
        if "code" in reference.columns:
            return reference.select(pl.col("id").alias("element_id"), "code")
    return None


def prior_history(season: str, roster: pl.DataFrame, codes: pl.DataFrame | None = None) -> pl.DataFrame | None:
    """Last season's rows for `roster`'s players, ready to pass as
    `prior_history` to analytics.projections.

    `roster`: element_id -> team for `season`. `codes`: element_id -> code
    for `season`, looked up with `season_codes` when not given (the live
    path passes bootstrap-static's own).

    None — the model then runs on this season alone, as it did before
    carryover existed — when the previous season is not in the archive or
    either season's codes are unknown. Logged, because a projection that
    has quietly lost its history looks exactly like one that never had it.
    """
    prev = previous_season(season)
    prev_path = NORMALIZED_DIR / f"{prev}.parquet"
    codes = codes if codes is not None else season_codes(season)
    prev_codes = season_codes(prev)
    if not prev_path.exists() or codes is None or prev_codes is None:
        logger.warning(
            "no carryover into %s: %s", season,
            f"{prev_path} missing" if not prev_path.exists() else "player codes missing for one of the two seasons",
        )
        return None
    return carry_forward(pl.read_parquet(prev_path), prev_codes, codes, roster.select("element_id", "team"))


def season_start_roster(season_df: pl.DataFrame) -> pl.DataFrame:
    """element_id -> each player's club at his first row of the season: the
    `roster` prior_history needs, for an archived season."""
    return season_df.sort("gw").group_by("element_id").agg(pl.col("team").first())
