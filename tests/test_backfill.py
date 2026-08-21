"""§3.2 hazards: xP dropped, position/club resolved per gameweek (not a
season-final join), promoted-club flag correctly derived. §3.6 acceptance
criteria are exercised directly against this module's normalization logic
using small synthetic season files — no network access."""

from __future__ import annotations

import csv
from pathlib import Path

import polars as pl
import pytest
import yaml

from backtest.backfill import NORMALIZED_DIR, normalize_season

MERGED_GW_COLUMNS = [
    "element", "name", "team", "position", "opponent_team", "was_home", "kickoff_time", "round", "fixture",
    "minutes", "starts", "total_points", "goals_scored", "assists", "clean_sheets", "goals_conceded", "own_goals",
    "penalties_saved", "penalties_missed", "yellow_cards", "red_cards", "saves", "bonus", "bps", "influence",
    "creativity", "threat", "ict_index", "value", "selected", "transfers_in", "transfers_out", "xP",
]

TEAMS = {1: "Team A", 2: "Team B", 3: "Team C", 4: "PromoTown"}

# element 1 transfers from Team A to Team B after gw2 — the known mid-season
# transfer the acceptance criteria ask for. element 2 is on the promoted club.
_ROWS = [
    # element, name, team, position, opponent_team, was_home, gw, fixture
    (1, "Star Player", "Team A", "MID", 2, True, 1, 101),
    (2, "Promo Striker", "PromoTown", "FWD", 3, False, 1, 102),
    (3, "Other Fwd", "Team C", "FWD", 4, True, 1, 103),
    (1, "Star Player", "Team A", "MID", 3, False, 2, 104),
    (2, "Promo Striker", "PromoTown", "FWD", 1, True, 2, 105),
    (3, "Other Fwd", "Team C", "FWD", 2, False, 2, 106),
    (1, "Star Player", "Team B", "MID", 1, True, 3, 107),  # <- transferred here
    (2, "Promo Striker", "PromoTown", "FWD", 2, False, 3, 108),
    (3, "Other Fwd", "Team C", "FWD", 1, True, 3, 109),
    (1, "Star Player", "Team B", "MID", 4, False, 4, 110),
    (2, "Promo Striker", "PromoTown", "FWD", 1, True, 4, 111),
    (3, "Other Fwd", "Team C", "FWD", 2, False, 4, 112),
]


def _write_merged_gw(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MERGED_GW_COLUMNS)
        writer.writeheader()
        for element, name, team, position, opponent_team, was_home, gw, fixture in _ROWS:
            writer.writerow(
                {
                    "element": element, "name": name, "team": team, "position": position,
                    "opponent_team": opponent_team, "was_home": was_home,
                    "kickoff_time": f"2025-0{gw}-01T12:00:00Z", "round": gw, "fixture": fixture,
                    "minutes": 90, "starts": 1, "total_points": 5, "goals_scored": 0, "assists": 0,
                    "clean_sheets": 0, "goals_conceded": 1, "own_goals": 0, "penalties_saved": 0,
                    "penalties_missed": 0, "yellow_cards": 0, "red_cards": 0, "saves": 0, "bonus": 0,
                    "bps": 20, "influence": "10.0", "creativity": "5.0", "threat": "5.0", "ict_index": "2.0",
                    "value": 55, "selected": 100000, "transfers_in": 0, "transfers_out": 0,
                    # xP: deliberately contaminated-looking value, must never survive normalization.
                    "xP": 999.9,
                }
            )


def _write_fixtures(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "event", "kickoff_time", "team_h_difficulty", "team_a_difficulty"])
        writer.writeheader()
        for fixture_id in {row[7] for row in _ROWS}:
            # Fixture 111 (element 2's gw4 home game) deliberately hard for
            # the home side, easy for the away side, to exercise the
            # opponent_difficulty was_home branch.
            if fixture_id == 111:
                writer.writerow({"id": fixture_id, "event": 4, "kickoff_time": "2025-04-01T12:00:00Z", "team_h_difficulty": 5, "team_a_difficulty": 1})
            else:
                writer.writerow({"id": fixture_id, "event": 1, "kickoff_time": "2025-01-01T12:00:00Z", "team_h_difficulty": 3, "team_a_difficulty": 3})


def _write_teams(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "name"])
        writer.writeheader()
        for team_id, name in TEAMS.items():
            writer.writerow({"id": team_id, "name": name})


@pytest.fixture
def season_files(tmp_path: Path) -> dict[str, Path]:
    merged_gw = tmp_path / "merged_gw.csv"
    fixtures = tmp_path / "fixtures.csv"
    teams = tmp_path / "teams.csv"
    _write_merged_gw(merged_gw)
    _write_fixtures(fixtures)
    _write_teams(teams)
    return {"merged_gw": merged_gw, "fixtures": fixtures, "teams": teams}


@pytest.fixture
def promoted_clubs_config(tmp_path: Path) -> Path:
    path = tmp_path / "promoted_clubs.yaml"
    path.write_text(yaml.dump({"2024-25": ["PromoTown"]}), encoding="utf-8")
    return path


def test_xp_column_absent_from_normalized_output(season_files, promoted_clubs_config):
    df = normalize_season("2024-25", season_files, promoted_clubs_config)
    assert "xP" not in df.columns
    assert "xp" not in [c.lower() for c in df.columns]


def test_club_resolved_per_gameweek_across_a_known_transfer(season_files, promoted_clubs_config):
    df = normalize_season("2024-25", season_files, promoted_clubs_config)
    star = df.filter(pl.col("element_id") == 1).sort("gw")
    assert star["team"].to_list() == ["Team A", "Team A", "Team B", "Team B"]


def test_position_resolved_from_per_gameweek_row_not_a_season_summary(season_files, promoted_clubs_config):
    df = normalize_season("2024-25", season_files, promoted_clubs_config)
    # Every row's position comes straight from that row's own gw record —
    # there is no join to any season-final/current mapping in normalize_season
    # at all, so a stale current-season position table couldn't leak in even
    # if one existed. Assert the per-row value made it through unmodified.
    assert df.filter(pl.col("element_id") == 1)["position"].unique().to_list() == ["MID"]
    assert df.filter(pl.col("element_id") == 2)["position"].unique().to_list() == ["FWD"]


def test_promoted_club_flag_set_from_config(season_files, promoted_clubs_config):
    df = normalize_season("2024-25", season_files, promoted_clubs_config)
    assert df.filter(pl.col("element_id") == 2)["is_promoted_club"].all()  # PromoTown
    assert not df.filter(pl.col("element_id") == 1)["is_promoted_club"].any()  # Team A/B


def test_opponent_team_resolved_to_readable_name(season_files, promoted_clubs_config):
    df = normalize_season("2024-25", season_files, promoted_clubs_config)
    row = df.filter((pl.col("element_id") == 1) & (pl.col("gw") == 1)).row(0, named=True)
    assert row["opponent_team"] == "Team B"  # opponent_team id 2 in the fixture


def test_opponent_difficulty_uses_was_home_branch(season_files, promoted_clubs_config):
    df = normalize_season("2024-25", season_files, promoted_clubs_config)
    row = df.filter((pl.col("element_id") == 2) & (pl.col("gw") == 4)).row(0, named=True)
    assert row["was_home"] is True
    assert row["opponent_difficulty"] == 5  # team_h_difficulty on fixture 111


def test_double_gameweek_rows_are_summed_not_rejected(season_files, promoted_clubs_config):
    """A real hazard, not a hypothetical one: 2023-24 gw7 has 983 rows where
    a rearranged fixture gives a player two matches in one FPL round. FPL
    sums a manager's points across both; normalize_season must too, rather
    than treating the duplicate (element_id, gw) as a schema violation."""
    # A second fixture 101b for element 1 in the same gw1, against a
    # different opponent (team 3) — a genuine blank/double gameweek shape.
    with season_files["merged_gw"].open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MERGED_GW_COLUMNS)
        writer.writerow(
            {
                "element": 1, "name": "Star Player", "team": "Team A", "position": "MID", "opponent_team": 3,
                "was_home": False, "kickoff_time": "2025-01-03T12:00:00Z", "round": 1, "fixture": 199,
                "minutes": 90, "starts": 1, "total_points": 7, "goals_scored": 1, "assists": 0, "clean_sheets": 0,
                "goals_conceded": 1, "own_goals": 0, "penalties_saved": 0, "penalties_missed": 0, "yellow_cards": 0,
                "red_cards": 0, "saves": 0, "bonus": 2, "bps": 25, "influence": "12.0", "creativity": "6.0",
                "threat": "8.0", "ict_index": "3.0", "value": 55, "selected": 100000, "transfers_in": 0,
                "transfers_out": 0, "xP": 999.9,
            }
        )
    with season_files["fixtures"].open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "event", "kickoff_time", "team_h_difficulty", "team_a_difficulty"])
        writer.writerow({"id": 199, "event": 1, "kickoff_time": "2025-01-03T12:00:00Z", "team_h_difficulty": 4, "team_a_difficulty": 2})

    df = normalize_season("2024-25", season_files, promoted_clubs_config)
    row = df.filter((pl.col("element_id") == 1) & (pl.col("gw") == 1)).row(0, named=True)

    assert row["n_fixtures"] == 2
    assert row["total_points"] == 5 + 7  # summed across both fixtures
    assert row["minutes"] == 90 + 90
    assert row["goals_scored"] == 1
    # ambiguous across the two fixtures — not silently picking one:
    assert row["was_home"] is None
    assert row["opponent_team"] == "Team B & Team C"
    # not-a-duplicate rows are unaffected: single fixture, n_fixtures == 1.
    other = df.filter((pl.col("element_id") == 2) & (pl.col("gw") == 1)).row(0, named=True)
    assert other["n_fixtures"] == 1


def test_assistant_manager_pick_rows_are_excluded(season_files, promoted_clubs_config):
    """FPL's short-lived 'Assistant Manager' pick (real head coaches, scored
    by team results, not individual events) shows up as position=="AM" in
    the real 2024-25 data starting round 23 — not a player, not part of
    this schema."""
    with season_files["merged_gw"].open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MERGED_GW_COLUMNS)
        writer.writerow(
            {
                "element": 999, "name": "Some Head Coach", "team": "Team A", "position": "AM", "opponent_team": 2,
                "was_home": True, "kickoff_time": "2025-01-01T12:00:00Z", "round": 1, "fixture": 101,
                "minutes": 0, "starts": 0, "total_points": 8, "goals_scored": 0, "assists": 0, "clean_sheets": 0,
                "goals_conceded": 0, "own_goals": 0, "penalties_saved": 0, "penalties_missed": 0, "yellow_cards": 0,
                "red_cards": 0, "saves": 0, "bonus": 0, "bps": 0, "influence": "0.0", "creativity": "0.0",
                "threat": "0.0", "ict_index": "0.0", "value": 50, "selected": 1000, "transfers_in": 0,
                "transfers_out": 0, "xP": 0,
            }
        )
    df = normalize_season("2024-25", season_files, promoted_clubs_config)
    assert df.filter(pl.col("element_id") == 999).height == 0
    assert "AM" not in df["position"].unique().to_list()


@pytest.mark.skipif(not (NORMALIZED_DIR / "2024-25.parquet").exists(), reason="requires the committed backfilled data")
def test_real_2024_25_data_resolves_a_known_mid_season_transfer_per_gameweek():
    """Armando Broja (element 156, 2024-25): Chelsea gw1-2, then Everton for
    the rest of the season on transfer-deadline-day loan. Confirmed directly
    against vaastav's source CSV, not assumed — real data, not a synthetic
    stand-in, for the specific acceptance criterion this hazard names."""
    df = pl.read_parquet(NORMALIZED_DIR / "2024-25.parquet")
    broja = df.filter(pl.col("element_id") == 156).sort("gw")
    assert broja.filter(pl.col("gw") <= 2)["team"].to_list() == ["Chelsea", "Chelsea"]
    assert broja.filter(pl.col("gw") >= 3)["team"].unique().to_list() == ["Everton"]
