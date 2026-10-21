"""D18: P(haul), the captaincy ranking, the frozen record and its scoring,
and `captaincy.json`.

The maths is tested on hand-built rows, where the answer is known; the
scoring on tmp freezes, including one that predates the captaincy field
(which must be skipped, not crash and not be edited); the file against the
real data.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import polars as pl
import pytest

from analytics.captaincy import SHORTLIST, freeze_record, most_owned, rank_candidates
from analytics.haul import (
    HAUL_THRESHOLD,
    LOW_THRESHOLD,
    BonusModel,
    haul_summary,
    load_bonus_model,
    points_distribution,
)
from analytics.projections import expected_points_from_projection
from analytics.scoring import load_scoring_config
from papertrade.actuals import _SCHEMA
from papertrade.evaluate import brier_score, evaluate_captaincy
from papertrade.freeze import write_freeze
from web.export.captaincy import build_captaincy
from web.export.contract import CaptaincyFile

CONFIG = load_scoring_config(Path("config/scoring_2025_26.yaml"))
PATH = Path("data/web/v1/captaincy.json")
PANEL = Path("data/web/v1/panel.parquet")


def _row(element_id=1, position="MID", n=1, goals=0.4, assists=0.2, p_full=0.9, p_short=0.05, **over):
    """One projected row as `with_scoreline(project_event_vectors(...))`
    leaves it: the per-appearance rates already scaled by P(plays)."""
    p_blank = 1.0 - p_full - p_short
    plays = 1.0 - p_blank
    row = {
        "element_id": element_id, "position": position, "team": "Team A", "n_fixtures": n,
        "p_blank": p_blank, "p_short": p_short, "p_full": p_full,
        "expected_goals_trailing": goals * plays, "expected_assists_trailing": assists * plays,
        "clean_sheets_trailing": 0.0, "goals_conceded_trailing": 0.0, "saves_trailing": 0.0,
        "bonus_trailing": 0.3 * plays, "yellow_cards_trailing": 0.05 * plays, "red_cards_trailing": 0.0,
        "own_goals_trailing": 0.0, "penalties_missed_trailing": 0.0, "penalties_saved_trailing": 0.0,
        "p_dc_threshold_trailing": 0.0,
        "attack_mult": 1.0 * n, "defence_mult": 1.0 * n, "expected_cs": 0.3 * n, "expected_gc_deductions": 0.6 * n,
    }
    row.update(over)
    return row


def _bonus_model() -> BonusModel:
    table = {}
    for position in ("GK", "DEF", "MID", "FWD"):
        table[(position, 0)] = [0.9, 0.06, 0.03, 0.01]
        table[(position, 1)] = [0.5, 0.2, 0.15, 0.15]
        table[(position, 2)] = [0.2, 0.2, 0.25, 0.35]
    return BonusModel(table)


# --- P(haul) --------------------------------------------------------------


def test_the_distribution_is_a_distribution_and_its_mean_is_the_models_xp():
    for position in ("GK", "DEF", "MID", "FWD"):
        for n in (1, 2):
            row = _row(position=position, n=n)
            dist = points_distribution(row, CONFIG)
            assert sum(dist.values()) == pytest.approx(1.0, abs=1e-6)
            mean = sum(x * p for x, p in dist.items())
            assert mean == pytest.approx(expected_points_from_projection(row, CONFIG), abs=1e-3), (position, n)


def test_a_bonus_model_keeps_the_mean_and_fattens_the_tail():
    row = _row(position="FWD", goals=0.5, assists=0.2)
    plain = haul_summary(row, CONFIG)
    shaped = haul_summary(row, CONFIG, _bonus_model())
    assert shaped["mean"] == pytest.approx(plain["mean"], abs=1e-3)
    assert shaped["p_haul"] > plain["p_haul"]


def test_a_blank_gameweek_never_hauls_and_a_double_hauls_more():
    assert haul_summary(_row(n=0), CONFIG) == {"p_haul": 0.0, "p_low": 1.0, "mean": 0.0}
    single, double = haul_summary(_row(n=1), CONFIG), haul_summary(_row(n=2), CONFIG)
    assert double["p_haul"] > single["p_haul"]
    assert double["mean"] == pytest.approx(2 * single["mean"], abs=1e-3)


def test_p_haul_rises_with_the_scoring_rate_and_falls_with_a_doubt_over_minutes():
    low, high = haul_summary(_row(goals=0.2), CONFIG)["p_haul"], haul_summary(_row(goals=0.9), CONFIG)["p_haul"]
    assert 0 < low < high < 1
    sure = haul_summary(_row(p_full=0.95, p_short=0.03), CONFIG)["p_haul"]
    doubtful = haul_summary(_row(p_full=0.4, p_short=0.1), CONFIG)["p_haul"]
    assert doubtful < sure


def test_a_player_with_no_chance_of_playing_has_p_haul_zero():
    out = haul_summary(_row(p_full=0.0, p_short=0.0), CONFIG)
    assert out["p_haul"] == 0.0 and out["p_low"] == 1.0


def test_p_haul_is_deterministic_and_the_threshold_is_a_named_constant():
    assert HAUL_THRESHOLD == 10 and LOW_THRESHOLD == 2
    row = _row(position="FWD", goals=0.6)
    assert haul_summary(row, CONFIG) == haul_summary(row, CONFIG)
    # A forward needs two goals (8) plus a full-minutes 2 to reach 10, so
    # P(>= 2 goals | appears) * P(60+) is a floor.
    floor = (1 - math.exp(-0.6) * (1 + 0.6)) * 0.9
    assert haul_summary(row, CONFIG)["p_haul"] >= floor - 1e-9


def test_bonus_model_reads_only_seasons_before_the_one_projected():
    assert load_bonus_model("2023-24") is None  # nothing earlier in the archive
    model = load_bonus_model("2025-26")
    assert model is not None
    for pmf in model.table.values():
        assert sum(pmf) == pytest.approx(1.0)
    # A goal contribution makes bonus likelier.
    assert model.mean("FWD", 2) > model.mean("FWD", 1) > model.mean("FWD", 0)


# --- ranking --------------------------------------------------------------


def _frame(rows):
    return pl.DataFrame(rows)


def test_candidates_are_ranked_by_xp_and_blanks_are_excluded():
    rows = [
        _row(1, goals=0.2), _row(2, goals=0.9), _row(3, goals=0.5), _row(4, n=0), _row(5, goals=0.9),
    ]
    meta = {i: {"name": f"P{i}", "team": "Team A", "ownership_pct": float(i)} for i in range(1, 6)}
    ranked = rank_candidates(_frame(rows), CONFIG, meta, top_n=3)
    assert [r["element_id"] for r in ranked] == [2, 5, 3]  # the tie at goals=0.9 breaks to the lower id
    assert ranked[0]["xp"] >= ranked[1]["xp"] >= ranked[2]["xp"]
    assert all(0 <= r["p_haul"] <= 1 for r in ranked)
    assert ranked[0]["name"] == "P2" and ranked[0]["ownership_pct"] == 2.0


def test_most_owned_breaks_ties_to_the_lower_id_and_tolerates_no_ownership():
    meta = {7: {"name": "A", "ownership_pct": 40.0}, 3: {"name": "B", "ownership_pct": 40.0}, 9: {"name": "C", "ownership_pct": 5.0}}
    assert most_owned(meta)["element_id"] == 3
    assert most_owned({1: {"name": "A", "ownership_pct": None}}) is None


# --- the frozen record and its scoring ------------------------------------


def _candidates(n=20):
    return [
        {"element_id": i, "name": f"P{i}", "team": "T", "position": "MID", "xp": 10.0 - i * 0.1,
         "p_haul": 0.3 - i * 0.01, "p_low": 0.1, "ownership_pct": 50.0 - i}
        for i in range(1, n + 1)
    ]


def _freeze(gw: int, with_captaincy: bool = True) -> dict:
    payload = {"gameweek": gw, "projections": {str(gw): {str(i): float(i % 5) + 0.5 for i in range(1, 21)}}}
    if with_captaincy:
        payload["captaincy"] = freeze_record(_candidates(), {"element_id": 30, "name": "Template", "team": "T", "ownership_pct": 80.0})
    return payload


def _actuals(gw: int, points: dict[int, int]) -> pl.DataFrame:
    from squad.live import FLOAT_STAT_COLUMNS, INT_STAT_COLUMNS

    return pl.DataFrame(
        [
            {
                "gw": gw, "element_id": eid, "position": "MID", "team": "T", "is_promoted_club": False,
                **{c: 0 for c in INT_STAT_COLUMNS}, **{c: 0.0 for c in FLOAT_STAT_COLUMNS},
                "minutes": 90, "total_points": pts,
            }
            for eid, pts in points.items()
        ],
        schema=_SCHEMA,
    )


def test_the_freeze_record_holds_the_pick_the_shortlist_and_every_candidate():
    record = _freeze(8)["captaincy"]
    assert record["pick"]["element_id"] == 1
    assert [c["element_id"] for c in record["shortlist"]] == list(range(1, SHORTLIST + 1))
    assert len(record["candidates"]) == 20
    assert record["haul_threshold"] == HAUL_THRESHOLD
    assert record["most_owned"]["element_id"] == 30


def test_scoring_doubles_the_captain_and_compares_with_the_baselines(tmp_path):
    write_freeze(8, _freeze(8), freezes_dir=tmp_path)
    # pick (1) scores 4, most-owned (30) 9, best of the shortlist is player 3 with 13; player 2 hauls too.
    actuals = _actuals(8, {1: 4, 2: 10, 3: 13, 4: 2, 5: 0, 30: 9, **{i: 2 for i in range(6, 21)}})
    out = evaluate_captaincy([8], freezes_dir=tmp_path, actuals=actuals)
    [gw] = out["per_gw"]
    assert gw["pick_points"] == 4 and gw["pick_points_x2"] == 8 and gw["pick_hauled"] is False
    assert gw["most_owned_points"] == 9 and gw["most_owned_points_x2"] == 18
    assert gw["best_of_shortlist"]["element_id"] == 3 and gw["best_of_shortlist_points_x2"] == 26
    assert out["summary"]["n_gameweeks"] == 1
    assert out["summary"]["mean_pick_points_x2"] == 8
    # Pooled calibration over all 20 frozen candidates; players 2 and 3 hauled.
    cal = out["summary"]["calibration"]
    assert cal["n"] == 20 and cal["haul_rate"] == pytest.approx(2 / 20)
    expected = sum((c["p_haul"] - (c["element_id"] in (2, 3))) ** 2 for c in _candidates()) / 20
    assert cal["brier"] == pytest.approx(expected)
    assert cal["brier_climatology"] == pytest.approx(0.1 * 0.9)


def test_a_freeze_without_the_captaincy_field_is_skipped_not_fatal_and_not_edited(tmp_path):
    old = write_freeze(6, _freeze(6, with_captaincy=False), freezes_dir=tmp_path)
    before = old.read_bytes()
    write_freeze(8, _freeze(8), freezes_dir=tmp_path)
    actuals = pl.concat([_actuals(6, {1: 5}), _actuals(8, {i: 3 for i in list(range(1, 21)) + [30]})])
    out = evaluate_captaincy([6, 7, 8], freezes_dir=tmp_path, actuals=actuals)
    assert [g["gw"] for g in out["per_gw"]] == [8]
    reasons = {s["gw"]: s["reason"] for s in out["skipped"]}
    assert reasons[6] == "freeze predates the captaincy record" and reasons[7] == "no freeze"
    assert old.read_bytes() == before


def test_a_frozen_gameweek_without_actuals_waits(tmp_path):
    write_freeze(8, _freeze(8), freezes_dir=tmp_path)
    out = evaluate_captaincy([8], freezes_dir=tmp_path, actuals=pl.DataFrame(schema=_SCHEMA))
    assert out["per_gw"] == [] and out["skipped"][0]["reason"] == "no actuals recorded yet"
    assert out["summary"]["calibration"]["brier"] is None


def test_brier_score_of_nothing_is_none_and_of_a_perfect_forecast_is_zero():
    assert brier_score([])["brier"] is None
    perfect = brier_score([{"p_haul": 1.0, "hauled": True}, {"p_haul": 0.0, "hauled": False}])
    assert perfect["brier"] == 0.0


# --- the file -------------------------------------------------------------


@lru_cache(maxsize=1)
def _built(freezes_dir: str) -> CaptaincyFile:
    return build_captaincy(as_of=datetime.now(timezone.utc), freezes_dir=Path(freezes_dir), actuals=pl.DataFrame(schema=_SCHEMA))


def _need_panel():
    if not PANEL.exists():  # pragma: no cover
        pytest.skip("panel.parquet is a build artifact; run `python -m web.export panel`")


def test_the_build_ranks_twenty_candidates_for_the_next_gameweek(tmp_path):
    _need_panel()
    file = _built(str(tmp_path))
    assert file.gameweek == file.header.source_gameweek + 1
    assert len(file.candidates) == 20 and file.header.rows == 20
    xps = [c.xp for c in file.candidates]
    assert xps == sorted(xps, reverse=True)
    assert all(0 <= c.p_haul <= 1 and 0 <= c.p_low <= 1 for c in file.candidates)
    assert all(c.opponents for c in file.candidates)  # none is a blank
    assert file.haul_threshold == HAUL_THRESHOLD
    assert file.track_record.n_gameweeks == 0


def test_the_first_candidates_xp_agrees_with_projections_json(tmp_path):
    """Same model, same number: the pick's xP is the player's xP in the
    projections file for the same gameweek, both built fresh."""
    _need_panel()
    from web.export.projections import build_projections

    as_of = datetime.now(timezone.utc)
    captaincy = build_captaincy(as_of=as_of, freezes_dir=tmp_path, actuals=pl.DataFrame(schema=_SCHEMA))
    projections = build_projections(as_of=as_of)
    assert captaincy.gameweek == projections.gameweeks[0]
    xp = {p.element_id: p.xp[0] for p in projections.players}
    for c in captaincy.candidates:
        assert c.xp == pytest.approx(xp[c.element_id], abs=1e-9)


def test_the_committed_file_validates_against_the_contract():
    if not PATH.exists():  # pragma: no cover
        pytest.skip("captaincy.json not committed")
    file = CaptaincyFile.model_validate(json.loads(PATH.read_text(encoding="utf-8")))
    assert file.header.contract_version == 1
    assert len(file.candidates) == 20
    assert file.haul_threshold == HAUL_THRESHOLD


def test_captaincy_record_reads_names_teams_and_ownership_from_bootstrap():
    from types import SimpleNamespace

    from papertrade.freeze import captaincy_record

    bootstrap = SimpleNamespace(
        teams=[SimpleNamespace(id=1, name="Team A")],
        elements=[
            SimpleNamespace(id=1, web_name="Alpha", team=1, selected_by_percent="12.5"),
            SimpleNamespace(id=2, web_name="Beta", team=1, selected_by_percent="60.1"),
        ],
    )
    vectors = pl.DataFrame([_row(1, goals=0.9), _row(2, goals=0.2)])
    record = captaincy_record(vectors, CONFIG, bootstrap)
    assert record["pick"]["name"] == "Alpha" and record["pick"]["team"] == "Team A"
    assert record["pick"]["ownership_pct"] == 12.5
    assert record["most_owned"]["element_id"] == 2
    assert captaincy_record(None, CONFIG, bootstrap) is None
