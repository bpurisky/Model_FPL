"""Downloads and normalizes historical FPL seasons (§3.1) into
data/historical/{season}.parquet — the committed input the backtest harness
reads. Re-running this script re-derives that Parquet from a fresh download;
nothing downstream depends on the raw cache surviving.

Source: vaastav/Fantasy-Premier-League — one row per player per gameweek,
plus per-season fixtures.csv (kickoff times, FPL's own difficulty ratings)
and teams.csv (id -> name).

Known hazards this module exists to handle (§3.2):
  - `xP` is dropped entirely — contaminated, reflects post-match information.
  - `team` and `position` are read from the per-gameweek row itself, never
    joined from a season-final summary table (players_raw.csv,
    cleaned_players.csv) — that join is exactly how 2026/27's position
    reclassifications would silently leak into historical training data.
  - Promoted clubs (config/promoted_clubs.yaml) get flagged so baselines.py
    can apply a principled prior instead of nulling out their early gameweeks.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
import polars as pl

# pyyaml: see collector/config.py — same justification, same file.
import yaml

logger = logging.getLogger("backtest.backfill")

VAASTAV_RAW = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

SEASONS = ["2023-24", "2024-25", "2025-26"]

RAW_CACHE_DIR = Path("data/historical/raw")
NORMALIZED_DIR = Path("data/historical")
PROMOTED_CLUBS_CONFIG = Path("config/promoted_clubs.yaml")

# Columns kept from merged_gw.csv. Deliberately excludes:
#   - xP (§3.2 — contaminated, dropped entirely)
#   - mng_* (2024-25's short-lived "assistant manager" pick type; not a player)
#   - modified (internal scrape-diff flag, not football data)
_MERGED_GW_COLUMNS = [
    "element",
    "name",
    "team",
    "position",
    "opponent_team",
    "was_home",
    "kickoff_time",
    "round",
    "fixture",
    "minutes",
    "starts",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "value",
    "selected",
    "transfers_in",
    "transfers_out",
]


def load_promoted_clubs(season: str, config_path: Path = PROMOTED_CLUBS_CONFIG) -> list[str]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return config.get(season, [])


def _download(client: httpx.Client, url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("using cached %s", dest)
        return dest
    response = client.get(url, timeout=60.0, follow_redirects=True)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def download_season_files(season: str, client: httpx.Client | None = None) -> dict[str, Path]:
    owns_client = client is None
    client = client or httpx.Client()
    try:
        season_dir = RAW_CACHE_DIR / season
        return {
            "merged_gw": _download(client, f"{VAASTAV_RAW}/{season}/gws/merged_gw.csv", season_dir / "merged_gw.csv"),
            "fixtures": _download(client, f"{VAASTAV_RAW}/{season}/fixtures.csv", season_dir / "fixtures.csv"),
            "teams": _download(client, f"{VAASTAV_RAW}/{season}/teams.csv", season_dir / "teams.csv"),
        }
    finally:
        if owns_client:
            client.close()


def _load_teams(teams_path: Path) -> pl.DataFrame:
    """id -> name, so `opponent_team` can be a readable name like `team` is,
    instead of a per-season-arbitrary numeric id that means nothing without
    this file alongside it."""
    return pl.read_csv(teams_path, columns=["id", "name"]).rename({"id": "opponent_team", "name": "opponent_team_name"})


def _load_fixture_difficulty(fixtures_path: Path) -> pl.DataFrame:
    """One row per fixture: id, team_h_difficulty, team_a_difficulty.

    FPL's own historical difficulty ratings (1-5), already point-in-time
    correct since they're the values that were actually published for that
    fixture — a simpler, defensible opponent-difficulty signal for the
    baseline. The custom Elo-based FDR replacement is Phase 2 (§4.3); this
    just needs *some* documented difficulty scaling to beat.
    """
    return pl.read_csv(
        fixtures_path,
        columns=["id", "team_h_difficulty", "team_a_difficulty"],
        schema_overrides={"id": pl.Int64, "team_h_difficulty": pl.Int64, "team_a_difficulty": pl.Int64},
    ).rename({"id": "fixture"})


def normalize_season(season: str, files: dict[str, Path], promoted_clubs_path: Path = PROMOTED_CLUBS_CONFIG) -> pl.DataFrame:
    raw = pl.read_csv(
        files["merged_gw"],
        columns=_MERGED_GW_COLUMNS,
        schema_overrides={"opponent_team": pl.Int64, "fixture": pl.Int64, "round": pl.Int64},
        try_parse_dates=False,
    )

    fixture_difficulty = _load_fixture_difficulty(files["fixtures"])
    teams = _load_teams(files["teams"])
    promoted_clubs = load_promoted_clubs(season, promoted_clubs_path)

    per_fixture = (
        raw.join(fixture_difficulty, on="fixture", how="left")
        .join(teams, on="opponent_team", how="left")
        .with_columns(
            pl.lit(season).alias("season"),
            pl.col("round").alias("gw"),
            pl.col("kickoff_time").str.to_datetime(time_zone="UTC", strict=False),
            pl.when(pl.col("was_home")).then(pl.col("team_h_difficulty")).otherwise(pl.col("team_a_difficulty")).alias(
                "opponent_difficulty"
            ),
            pl.col("team").is_in(promoted_clubs).alias("is_promoted_club"),
        )
        .rename({"element": "element_id"})
        .drop("round", "team_h_difficulty", "team_a_difficulty", "opponent_team")
        .rename({"opponent_team_name": "opponent_team"})
    )

    # Blank/double gameweeks (§3.2 — a real hazard, not a hypothetical one:
    # 2023-24 gw7 has 983 such rows) mean (element_id, gw) is NOT unique per
    # fixture — a player can have zero or two fixtures in one FPL round when
    # matches get rearranged. FPL sums a manager's points across every
    # fixture played that gameweek, so per-fixture rows are aggregated the
    # same way rather than treated as a schema violation. `opponent_team`
    # and `was_home` become ambiguous for a double; they're joined/nulled
    # rather than arbitrarily picking one fixture and silently dropping data
    # from the other.
    additive = [
        "minutes", "starts", "total_points", "goals_scored", "assists", "clean_sheets", "goals_conceded",
        "own_goals", "penalties_saved", "penalties_missed", "yellow_cards", "red_cards", "saves", "bonus", "bps",
        "influence", "creativity", "threat", "ict_index",
    ]
    carried_first = ["season", "name", "team", "position", "is_promoted_club", "value", "selected", "transfers_in", "transfers_out"]

    df = per_fixture.group_by(["element_id", "gw"]).agg(
        [pl.col(c).sum().alias(c) for c in additive]
        + [pl.col(c).first().alias(c) for c in carried_first]
        + [
            pl.len().alias("n_fixtures"),
            pl.col("kickoff_time").min().alias("kickoff_time"),
            pl.col("opponent_difficulty").mean().alias("opponent_difficulty"),
            pl.col("opponent_team").unique().sort().str.join(" & ").alias("opponent_team"),
            pl.when(pl.col("was_home").n_unique() == 1)
            .then(pl.col("was_home").first())
            .otherwise(pl.lit(None, dtype=pl.Boolean))
            .alias("was_home"),
        ]
    )

    return df.select(
        [
            "season",
            "gw",
            "element_id",
            "name",
            "team",
            "position",
            "opponent_team",
            "opponent_difficulty",
            "was_home",
            "kickoff_time",
            "n_fixtures",
            "is_promoted_club",
            "minutes",
            "starts",
            "total_points",
            "goals_scored",
            "assists",
            "clean_sheets",
            "goals_conceded",
            "own_goals",
            "penalties_saved",
            "penalties_missed",
            "yellow_cards",
            "red_cards",
            "saves",
            "bonus",
            "bps",
            "influence",
            "creativity",
            "threat",
            "ict_index",
            "value",
            "selected",
            "transfers_in",
            "transfers_out",
        ]
    )


def backfill_season(season: str, client: httpx.Client | None = None) -> Path:
    files = download_season_files(season, client)
    df = normalize_season(season, files, PROMOTED_CLUBS_CONFIG)
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = NORMALIZED_DIR / f"{season}.parquet"
    df.write_parquet(out_path)
    logger.info("normalized %s: %d rows -> %s", season, df.height, out_path)
    return out_path


def backfill_all(seasons: list[str] | None = None) -> list[Path]:
    seasons = seasons or SEASONS
    with httpx.Client() as client:
        return [backfill_season(season, client) for season in seasons]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    backfill_all()
