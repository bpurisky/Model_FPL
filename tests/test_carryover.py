"""analytics.carryover: last season's rows, re-keyed onto this season's
element ids through FPL's stable `code`."""

from __future__ import annotations

import polars as pl
import pytest

from analytics.carryover import carry_forward, previous_season


def _prev_rows():
    return pl.DataFrame({
        "element_id": [11, 11, 12, 13],
        "gw": [37, 38, 38, 38],
        "team": ["Arsenal", "Arsenal", "Everton", "Everton"],
        "minutes": [90, 90, 90, 90],
        "expected_goals": [0.4, 0.6, 0.1, 0.2],
        "clean_sheets": [1, 0, 1, 1],
        "goals_conceded": [0, 2, 0, 0],
    })


def test_rows_are_rekeyed_by_code_and_placed_before_gw1():
    prev_codes = pl.DataFrame({"element_id": [11, 12, 13], "code": [500, 600, 700]})
    current_codes = pl.DataFrame({"element_id": [1, 2], "code": [500, 600]})  # 700 left the league
    current_teams = pl.DataFrame({"element_id": [1, 2], "team": ["Arsenal", "Everton"]})
    out = carry_forward(_prev_rows(), prev_codes, current_codes, current_teams).sort(["element_id", "gw"])
    assert out["element_id"].to_list() == [1, 1, 2]
    assert out["gw"].to_list() == [-1, 0, 0]
    assert out["expected_goals"].to_list() == pytest.approx([0.4, 0.6, 0.1])


def test_team_level_rates_do_not_follow_a_player_to_a_new_club():
    prev_codes = pl.DataFrame({"element_id": [11, 12], "code": [500, 600]})
    current_codes = pl.DataFrame({"element_id": [1, 2], "code": [500, 600]})
    current_teams = pl.DataFrame({"element_id": [1, 2], "team": ["Chelsea", "Everton"]})  # 500 moved
    out = carry_forward(_prev_rows(), prev_codes, current_codes, current_teams)
    moved = out.filter(pl.col("element_id") == 1)
    stayed = out.filter(pl.col("element_id") == 2)
    assert moved["clean_sheets"].null_count() == moved.height
    assert moved["goals_conceded"].null_count() == moved.height
    assert moved["expected_goals"].null_count() == 0  # his own chances go with him
    assert stayed["clean_sheets"].to_list() == [1]


def test_previous_season():
    assert previous_season("2026-27") == "2025-26"
    assert previous_season("2000-01") == "1999-00"
