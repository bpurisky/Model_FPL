"""§5.11.1 over `board.json`: the composite, the two surfaces built on it,
and the honesty mechanism §5.4.6 requires around the momentum buckets."""

from __future__ import annotations

import polars as pl
import pytest

from web.export.board import (
    BUCKETS,
    build_board,
    classify,
    composite_expr,
    driving_metrics,
    with_composite,
    with_momentum,
)
from web.export.normalize import load_frontend_config

WEIGHTS = {"MID": {"xg_per90": 0.6, "xa_per90": 0.4}}


def _rows(rows: list[dict]) -> pl.DataFrame:
    columns = {
        "season": pl.Utf8, "gw": pl.Int64, "element_id": pl.Int64, "name": pl.Utf8,
        "team": pl.Utf8, "position": pl.Utf8, "total_points": pl.Int64,
        "xg_per90_z_pos": pl.Float64, "xa_per90_z_pos": pl.Float64,
    }
    for row in rows:
        unknown = set(row) - set(columns)
        assert not unknown, f"unknown key(s): {unknown}"
    return pl.DataFrame({c: [r.get(c) for r in rows] for c in columns}, schema=columns)


def _player(element_id, gw, xg, xa=0.0, *, position="MID", points=2, season="2025-26"):
    return {
        "season": season, "gw": gw, "element_id": element_id, "name": f"P{element_id}",
        "team": "Alpha", "position": position, "total_points": points,
        "xg_per90_z_pos": xg, "xa_per90_z_pos": xa,
    }


# --- the composite ---------------------------------------------------------


def test_composite_is_the_weighted_sum_of_z_scores():
    df = _rows([_player(1, 1, xg=2.0, xa=1.0)])

    out = df.with_columns(composite_expr(WEIGHTS["MID"], set(df.columns)).alias("c"))

    assert out["c"][0] == pytest.approx(0.6 * 2.0 + 0.4 * 1.0)


def test_a_missing_metric_does_not_drag_the_composite_toward_zero():
    """The divisor is the weight actually applied, so a player with no
    history for one metric is scored on what is known about him rather
    than penalised for the gap (§5.3.3)."""
    df = _rows([_player(1, 1, xg=2.0, xa=None)])

    out = df.with_columns(composite_expr(WEIGHTS["MID"], set(df.columns)).alias("c"))

    assert out["c"][0] == pytest.approx(2.0), "scored on xG alone, not 0.6 x 2.0"


def test_each_position_is_scored_with_its_own_profile():
    """§5.4.6 is explicit that this is per position, and §5.7.1 is why: a
    defender will not post a forward's xG."""
    df = _rows([_player(1, 1, xg=2.0, position="MID"), _player(2, 1, xg=2.0, position="FWD")])

    out = with_composite(df, {"MID": {"xg_per90": 1.0}, "FWD": {"xa_per90": 1.0}})

    by_id = dict(zip(out["element_id"].to_list(), out["composite"].to_list()))
    assert by_id[1] == pytest.approx(2.0)
    assert by_id[2] == pytest.approx(0.0), "FWD is scored on xA, which is 0 here"


# --- momentum --------------------------------------------------------------


def test_rising_requires_a_rise_in_every_gameweek_of_the_window():
    """Consistent means monotone, not a fitted slope. A slope can be
    positive while the series zig-zags, and the brief asked for a rise
    across three games."""
    steady = [_player(1, gw, xg=x) for gw, x in enumerate([0.1, 0.2, 0.3], start=1)]
    zigzag = [_player(2, gw, xg=x) for gw, x in enumerate([0.1, 0.9, 0.3], start=1)]

    out = with_momentum(with_composite(_rows(steady + zigzag), {"MID": {"xg_per90": 1.0}}), 3)
    last = out.filter(pl.col("gw") == 3)
    flags = dict(zip(last["element_id"].to_list(), last["is_rising"].to_list()))

    assert flags[1] is True
    assert flags[2] is False, "up then down is not a consistent rise"


def test_a_player_without_enough_history_is_not_classified_as_rising():
    out = with_momentum(
        with_composite(_rows([_player(1, 1, xg=0.5)]), {"MID": {"xg_per90": 1.0}}), 3
    )

    assert out["is_rising"][0] is False
    assert out["is_declining"][0] is False


# --- the two surfaces ------------------------------------------------------


def test_rank_is_within_position_and_gameweek():
    """A board says who to look at now, so rank 1 must mean the best
    goalkeeper this week rather than the best goalkeeper-gameweek of the
    last three seasons."""
    rows = [_player(i, 1, xg=float(i)) for i in range(1, 6)]
    rows += [_player(i + 10, 1, xg=float(i), position="FWD") for i in range(1, 4)]

    out = classify(
        with_momentum(
            with_composite(_rows(rows), {"MID": {"xg_per90": 1.0}, "FWD": {"xg_per90": 1.0}}), 3
        ),
        0.75,
    )

    for position, expected in (("MID", 5), ("FWD", 3)):
        ranks = sorted(out.filter(pl.col("position") == position)["rank"].to_list())
        assert ranks == list(range(1, expected + 1)), f"{position} ranks are not 1..N"


def test_percentile_spans_the_position_group():
    rows = [_player(i, 1, xg=float(i)) for i in range(1, 11)]

    out = classify(with_momentum(with_composite(_rows(rows), {"MID": {"xg_per90": 1.0}}), 3), 0.75)

    assert out["percentile"].min() == pytest.approx(0.0)
    assert out["percentile"].max() == pytest.approx(1.0)


def test_optimal_wins_over_a_momentum_bucket_when_both_apply():
    """A card can only say one thing, and Optimal is the classification
    with measured edge."""
    best = [_player(1, gw, xg=x) for gw, x in enumerate([1.0, 2.0, 9.0], start=1)]
    others = [_player(i, gw, xg=0.0) for i in range(2, 6) for gw in (1, 2, 3)]

    out = classify(
        with_momentum(with_composite(_rows(best + others), {"MID": {"xg_per90": 1.0}}), 3), 0.75
    )
    top = out.filter((pl.col("gw") == 3) & (pl.col("element_id") == 1))

    assert top["is_rising"][0] is True
    assert top["bucket"][0] == "optimal"


# --- the honesty mechanism -------------------------------------------------


def test_momentum_buckets_are_scored_inside_the_pool_they_come_from():
    """Optimal takes the top quartile, so comparing a momentum bucket
    against everyone else would score it against a pool the good players
    were already removed from — it would read as negative however well
    momentum worked."""
    file = build_board()
    by_bucket = {b.bucket: b for b in file.bucket_accuracy}

    assert by_bucket["optimal"].comparison == "all classified players"
    for bucket in ("rising", "declining", "neutral"):
        assert by_bucket[bucket].comparison == "other non-optimal players"


def test_every_bucket_publishes_what_it_was_worth():
    """§5.4.6: if the app is going to classify players as rising, it must
    report how often rising players subsequently outperformed."""
    file = build_board()

    assert {b.bucket for b in file.bucket_accuracy} == set(BUCKETS)
    for entry in file.bucket_accuracy:
        assert entry.n > 0
        assert entry.lift is not None


def test_the_optimal_bucket_actually_outperforms():
    """The claim the ranked list rests on. If this ever fails, the board
    is ordering noise and should not ship."""
    file = build_board()
    optimal = next(b for b in file.bucket_accuracy if b.bucket == "optimal")

    assert optimal.lift > 0.3, f"optimal lift was only {optimal.lift:.3f}"


def test_the_rising_bucket_is_reported_honestly_however_it_measures():
    """No definition tried has made Rising predict, so this asserts the
    reporting rather than the direction: whatever it is worth travels
    with it."""
    file = build_board()
    rising = next(b for b in file.bucket_accuracy if b.bucket == "rising")

    assert rising.forward_points is not None
    assert rising.forward_points_other is not None
    assert rising.lift == pytest.approx(rising.forward_points - rising.forward_points_other)


# --- weights and provenance ------------------------------------------------


def test_the_weights_travel_with_the_board():
    """§5.4.6 requires them rendered on screen — the user must be able to
    read the model's opinion, not only receive its output."""
    file = build_board()
    exported = {w.position: w.weights for w in file.weights}

    assert set(exported) == {"GK", "DEF", "MID", "FWD"}
    assert exported == load_frontend_config()["board"]["position_weights"]


def test_negative_weights_survive_the_contract():
    """Conceding is bad for a defender, and a scoring layer that cannot
    express that is not modelling football."""
    file = build_board()
    defenders = next(w.weights for w in file.weights if w.position == "DEF")

    assert defenders["xgc_per90"] < 0
    assert defenders["goals_conceded_per90"] < 0


def test_drivers_name_what_moved_this_player_not_the_profile():
    """Ranked by contribution — weight times the player's own z-score — so
    a card says something about him rather than reciting the weights."""
    assert driving_metrics(
        {"xg_per90_z_pos": 0.1, "xa_per90_z_pos": 3.0},
        {"xg_per90": 0.6, "xa_per90": 0.4},
        limit=1,
    ) == ["xa_per90"]


def test_a_metric_the_player_has_no_value_for_is_not_a_driver():
    assert driving_metrics(
        {"xg_per90_z_pos": None, "xa_per90_z_pos": 1.0},
        {"xg_per90": 0.9, "xa_per90": 0.1},
    ) == ["xa_per90"]


# --- the committed file ----------------------------------------------------


def test_the_board_describes_the_latest_gameweek_in_the_panel():
    """It follows the panel, so the moment `data/current_season/` holds a
    gameweek this switches to it without anyone remembering to."""
    file = build_board()

    assert file.season >= "2025-26"
    assert file.gameweek >= 1
    assert file.header.source_gameweek == file.gameweek


def test_every_player_carries_a_rank_a_percentile_and_a_bucket():
    file = build_board()

    assert file.players
    for player in file.players:
        assert player.bucket in BUCKETS
        assert player.rank >= 1
        assert 0.0 <= player.percentile <= 1.0
        assert player.composite is not None
