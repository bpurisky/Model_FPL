"""§5.11.1 over `players.json`.

Two of these tests exist because the first build got them wrong, and both
failures were invisible from the code: a join that silently kept the
panel's single-gameweek `minutes` where the season total was meant, and
metrics taken at gameweek grain so every player who did not feature in the
latest gameweek had a card of nulls.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import pytest

from web.export.contract import PlayersFile
from web.export.players import ACTUAL_TOTALS, build_players

PLAYERS_PATH = Path("data/web/v1/players.json")


@lru_cache(maxsize=1)
def _built() -> PlayersFile:
    return build_players()


def _committed() -> dict | None:
    if not PLAYERS_PATH.exists():  # pragma: no cover
        return None
    return json.loads(PLAYERS_PATH.read_text(encoding="utf-8"))


# --- the two bugs ----------------------------------------------------------


def test_actuals_are_season_totals_not_the_latest_gameweek():
    """The panel carries `minutes` and `total_points` of its own — for the
    single gameweek each row describes — so an unprefixed join keeps the
    panel's values for exactly those two and the season's for everything
    else. That produced a row reading 0 minutes and 0 points beside 27
    goals, which is not a number anyone would query but is exactly the
    kind of thing a table renders without complaint.
    """
    file = _built()
    scorers = [p for p in file.players if (p.actuals.get("goals_scored") or 0) >= 5]

    assert scorers, "expected some players with real goal tallies"
    for player in scorers:
        assert player.actuals["minutes"] > 0, f"{player.name} scored but logged no minutes"
        assert player.actuals["total_points"] > 0


def test_a_player_who_missed_the_latest_gameweek_still_has_his_metrics():
    """Per-90 rates are null for a gameweek a player did not feature in,
    so metrics taken at gameweek grain would empty the card of anyone
    rested in the last round — the players most worth looking at.

    The rates here are season-to-date and minutes-weighted, which is what
    a player-level file means by "his xG per 90".
    """
    file = _built()
    rested = [
        p for p in file.players
        if (p.actuals.get("minutes") or 0) > 1500 and p.metrics["xg_per90"].value is not None
    ]

    assert len(rested) > 100, "season rates should exist for every regular starter"


def test_the_population_size_is_carried_once_per_group_not_per_player():
    """`n` is a property of the position group. Repeating it per player
    would restate the same 64 numbers thousands of times and invite a
    reader to think two players' z-scores rest on different samples."""
    file = _built()

    assert set(file.population) <= {"GK", "DEF", "MID", "FWD"}
    assert file.population, "no population recorded"
    for metrics in file.population.values():
        for size in metrics.values():
            assert size > 0
    assert not hasattr(file.players[0].metrics["xg_per90"], "n")


# --- the projection --------------------------------------------------------


def test_the_total_equals_the_sum_of_its_components():
    """Computed as that sum rather than by calling a second function, so
    the headline figure cannot disagree with the decomposition panel that
    breaks it down."""
    file = _built()
    projected = [p for p in file.players if p.projection]

    assert projected
    for player in projected:
        parts = [v for v in player.projection.components.values() if v is not None]
        assert player.projection.total == pytest.approx(sum(parts), abs=1e-9)


def test_the_minutes_distribution_is_a_distribution():
    file = _built()

    for player in file.players:
        if not player.projection:
            continue
        parts = [player.projection.p_blank, player.projection.p_short, player.projection.p_full]
        if any(p is None for p in parts):
            continue
        assert sum(parts) == pytest.approx(1.0, abs=1e-9)
        assert all(0.0 <= p <= 1.0 for p in parts)


def test_the_projection_basis_says_whether_a_fixture_is_known():
    """For a completed season there is no next gameweek to have a fixture,
    so difficulty falls back to neutral and the projection stops being
    about a specific match. Reading one as the other would read it wrong."""
    file = _built()

    assert file.projection_basis in {"next_fixture", "fixture_neutral"}
    assert file.projected_gameweek == file.gameweek + 1


def test_every_projected_player_appeared_in_the_latest_gameweek():
    """The roster is who played, so a projection cannot be attached to an
    element the season has already lost."""
    file = _built()
    ids = {p.element_id for p in file.players}

    for player in file.players:
        assert player.element_id in ids


# --- shape and honesty -----------------------------------------------------


def test_a_player_below_the_minutes_floor_has_a_null_z_not_a_zero():
    """A z-score of zero means exactly average, which is a finding. Null
    means the player has not played enough to be positioned at all
    (§5.3.3)."""
    file = _built()
    thin = [p for p in file.players if (p.actuals.get("minutes") or 0) < 200]

    assert thin, "expected some barely-used players"
    assert any(p.metrics["xg_per90"].z is None for p in thin)
    assert not any(p.metrics["xg_per90"].z == 0.0 for p in thin if (p.actuals.get("minutes") or 0) == 0)


def test_one_row_per_element():
    file = _built()
    ids = [p.element_id for p in file.players]

    assert len(ids) == len(set(ids))
    assert len(ids) == file.header.rows


def test_nothing_unserializable_reaches_the_json():
    payload = _built().model_dump_json()

    assert "NaN" not in payload and "Infinity" not in payload
    for player in json.loads(payload)["players"]:
        for metric in player["metrics"].values():
            for field in ("value", "z", "percentile"):
                assert metric[field] is None or math.isfinite(metric[field])


def test_actual_totals_are_the_declared_set():
    file = _built()
    seen = {key for player in file.players for key in player.actuals}

    assert seen <= set(ACTUAL_TOTALS)


def test_the_committed_file_matches_a_fresh_build():
    payload = _committed()
    if payload is None:  # pragma: no cover
        pytest.skip("players.json not generated yet -- run `python -m web.export players`")

    current = json.loads(_built().model_dump_json())
    for p in (payload, current):
        del p["header"]

    assert payload == current, "data/web/v1/players.json is stale -- regenerate it"
