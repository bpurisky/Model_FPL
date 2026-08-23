"""The team-level clean sheet model.

Two kinds of test here. The mechanical ones pin behaviour that must hold
for any input — point-in-time discipline, the direction each factor moves
lambda, a team quantity staying a team quantity. The measured ones assert
the model actually beats the base rate on real data, because a probability
that does not is worse than no column at all: it looks like information.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from analytics.clean_sheet import (
    CALIBRATION,
    FORM_WINDOW,
    HOME_XGC_RATIO,
    LEAGUE_XGC_PER_FIXTURE,
    clean_sheet_probability,
    evaluate_clean_sheet_model,
    team_gameweek_defence,
    with_head_to_head,
    with_trailing_form,
)

SEASONS = ("2023-24", "2024-25", "2025-26")
PLAYER_COLUMNS = [
    "season", "gw", "element_id", "team", "opponent_team", "was_home",
    "n_fixtures", "minutes", "expected_goals", "expected_goals_conceded",
    "goals_conceded",
]


def _archive() -> pl.DataFrame:
    return pl.concat(
        [
            pl.read_parquet(f"data/historical/{s}.parquet").with_columns(pl.lit(s).alias("season"))
            for s in SEASONS
        ],
        how="diagonal_relaxed",
    )


def _players(rows: list[dict]) -> pl.DataFrame:
    for row in rows:
        unknown = set(row) - set(PLAYER_COLUMNS)
        assert not unknown, f"unknown key(s): {unknown}"
    return pl.DataFrame(
        {c: [r.get(c) for r in rows] for c in PLAYER_COLUMNS},
        schema={
            "season": pl.Utf8, "gw": pl.Int64, "element_id": pl.Int64, "team": pl.Utf8,
            "opponent_team": pl.Utf8, "was_home": pl.Boolean, "n_fixtures": pl.Int64,
            "minutes": pl.Int64, "expected_goals": pl.Float64,
            "expected_goals_conceded": pl.Float64, "goals_conceded": pl.Int64,
        },
    )


def _match(gw, team, opp, *, home=True, xgc=1.5, xg=1.5, conceded=1, season="2025-26"):
    """Two players so `max` over the squad is exercised rather than assumed."""
    return [
        {
            "season": season, "gw": gw, "element_id": eid, "team": team,
            "opponent_team": opp, "was_home": home, "n_fixtures": 1, "minutes": 90,
            "expected_goals": xg / 2, "expected_goals_conceded": xgc,
            "goals_conceded": conceded,
        }
        for eid in (1, 2)
    ]


# --- the team quantity -----------------------------------------------------


def test_team_xgc_is_the_squad_maximum_not_a_sum():
    """Per-player xGC is the team's xGC while that player was on the
    pitch, so summing it across eleven players would report eleven times
    the chances the team actually faced."""
    rows = _match(1, "Alpha", "Beta", xgc=1.2)
    frame = team_gameweek_defence(_players(rows))

    assert frame["team_xgc"].to_list() == [pytest.approx(1.2)]


def test_every_player_of_a_team_gets_the_same_probability():
    """A clean sheet is one event for eleven players. Attaching a
    per-player probability would imply eleven different answers to the
    same question."""
    df = _players(_match(1, "Alpha", "Beta") + _match(2, "Alpha", "Gamma"))

    scored = clean_sheet_probability(df)

    assert scored.height == 2, "one row per team-gameweek, not per player"


def test_probability_is_a_poisson_zero():
    df = _players(
        _match(1, "Alpha", "Beta", xgc=1.0) + _match(2, "Alpha", "Gamma", xgc=1.0)
    )

    scored = clean_sheet_probability(df).drop_nulls("clean_sheet_prob")

    for probability in scored["clean_sheet_prob"]:
        assert 0.0 < probability < 1.0


# --- point-in-time (§0.3) ---------------------------------------------------


def test_a_teams_own_result_does_not_inform_its_own_prediction():
    """`shift(1)` before the rolling mean. Without it the model would be
    predicting a match from that match's own xGC, which would look
    spectacular and mean nothing."""
    rows = []
    for gw, xgc in enumerate([0.1, 0.2, 5.0], start=1):
        rows += _match(gw, "Alpha", f"Opp{gw}", xgc=xgc)

    trailing = with_trailing_form(team_gameweek_defence(_players(rows))).sort("gw")

    assert trailing["trailing_xgc"][0] is None, "nothing precedes the first match"
    assert trailing["trailing_xgc"][1] == pytest.approx(0.1)
    assert trailing["trailing_xgc"][2] == pytest.approx(0.15), "the 5.0 has not happened yet"


def test_head_to_head_reads_only_earlier_meetings():
    rows = []
    for gw, xgc in ((1, 0.5), (5, 2.5), (9, 9.9)):
        rows += _match(gw, "Alpha", "Beta", xgc=xgc)

    h2h = with_head_to_head(team_gameweek_defence(_players(rows))).sort("gw")

    assert h2h["h2h_matches"].to_list() == [0, 1, 2]
    assert h2h["h2h_xgc"][0] is None
    assert h2h["h2h_xgc"][1] == pytest.approx(0.5)
    assert h2h["h2h_xgc"][2] == pytest.approx(1.5), "mean of 0.5 and 2.5, not 9.9"


def test_head_to_head_spans_seasons():
    """A stylistic mismatch between two clubs outlives one campaign, and
    restricting to a single season would discard most meetings."""
    rows = _match(1, "Alpha", "Beta", xgc=0.4, season="2023-24") + _match(
        1, "Alpha", "Beta", xgc=3.0, season="2024-25"
    )

    h2h = with_head_to_head(team_gameweek_defence(_players(rows))).sort("season")

    assert h2h["h2h_matches"].to_list() == [0, 1]
    assert h2h["h2h_xgc"][1] == pytest.approx(0.4)


# --- each factor moves lambda the right way --------------------------------


def _probability(**kwargs) -> float:
    """One team's probability for gameweek 4, given three prior matches."""
    history = kwargs.pop("history", 1.5)
    rows = []
    for gw in range(1, 4):
        rows += _match(gw, "Alpha", f"Old{gw}", xgc=history, xg=1.5)
    rows += _match(4, "Alpha", "Beta", **kwargs)
    # the opponent needs its own history for the attack term to exist
    for gw in range(1, 4):
        rows += _match(gw, "Beta", f"Other{gw}", xg=kwargs.get("_opp_xg", 1.5), xgc=1.5)
    rows += _match(4, "Beta", "Alpha", home=not kwargs.get("home", True), xg=1.5, xgc=1.5)
    scored = clean_sheet_probability(_players(rows))
    return scored.filter((pl.col("team") == "Alpha") & (pl.col("gw") == 4))["clean_sheet_prob"][0]


def test_playing_at_home_raises_the_clean_sheet_probability():
    """Measured, not assumed: home sides concede 1.3436 xG against 1.6354
    away across the archive."""
    assert _probability(home=True) > _probability(home=False)


def test_a_leakier_recent_record_lowers_it():
    assert _probability(history=0.4) > _probability(history=2.8)


def test_the_home_ratio_is_below_one_so_the_direction_cannot_silently_flip():
    assert 0 < HOME_XGC_RATIO < 1


# --- shrinkage --------------------------------------------------------------


def test_a_team_with_no_history_sits_at_the_league_mean():
    """Early season is when this model is most used and least informed.
    Pulling toward the league mean is what stops one good result reading
    as a defence."""
    rows = _match(1, "Alpha", "Beta", xgc=0.0)

    scored = clean_sheet_probability(_players(rows))
    expected = math.exp(-LEAGUE_XGC_PER_FIXTURE * CALIBRATION * (HOME_XGC_RATIO ** 0.25))

    assert scored["clean_sheet_prob"][0] == pytest.approx(expected, rel=0.02)


def test_one_clean_sheet_moves_the_estimate_only_part_way():
    """With a shrinkage prior of four, a single match is worth a fifth of
    the estimate — not a fifth of nothing and not all of it."""
    shut_out = _probability(history=0.0)
    league = _probability(history=LEAGUE_XGC_PER_FIXTURE)

    assert shut_out > league
    assert shut_out < math.exp(-0.05), "but nowhere near a certainty"


# --- measured on real data --------------------------------------------------


def test_the_model_beats_the_base_rate_on_the_archive():
    """The claim that justifies the column existing. A probability that
    cannot beat 'everyone gets the league rate' is not information."""
    scored = clean_sheet_probability(_archive()).filter(pl.col("n_fixtures") == 1)

    result = evaluate_clean_sheet_model(scored)

    assert result["n"] > 2000
    assert result["brier"] < result["base_rate_brier"]
    assert result["skill"] > 0.03, f"skill was only {result['skill']:.4f}"


def test_the_probabilities_are_calibrated_across_the_range():
    """Monotone and close. A model whose top quintile does not actually
    keep more clean sheets than its bottom is ordering noise."""
    scored = clean_sheet_probability(_archive()).filter(pl.col("n_fixtures") == 1)

    bins = evaluate_clean_sheet_model(scored)["calibration"]

    actual = [b["actual"] for b in bins]
    assert actual == sorted(actual), f"not monotone: {actual}"
    assert actual[-1] > 2 * actual[0]
    for b in bins:
        assert abs(b["predicted"] - b["actual"]) < 0.05


def test_the_measured_constants_still_match_the_archive():
    """`HOME_XGC_RATIO` and the league means are measurements written down,
    not settings. If the archive gains a season they must be re-derived,
    and this is what says so."""
    teams = team_gameweek_defence(_archive()).filter(pl.col("n_fixtures") == 1)
    home = teams.filter(pl.col("was_home"))["xgc_per_fixture"].mean()
    away = teams.filter(~pl.col("was_home"))["xgc_per_fixture"].mean()

    assert home / away == pytest.approx(HOME_XGC_RATIO, abs=0.01)
    assert teams["xgc_per_fixture"].mean() == pytest.approx(LEAGUE_XGC_PER_FIXTURE, abs=0.02)


def test_the_form_window_is_shorter_than_a_season():
    """A window long enough to span a manager change is not form."""
    assert 3 <= FORM_WINDOW <= 10
