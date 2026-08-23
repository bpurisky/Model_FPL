"""§5.11.2 normalization tests, plus the §5.15 Q5 eligibility decision.

The four properties the spec names: z-scores recover to mean 0 / sd 1
within each position group, below-threshold players are null, `n` matches
the eligible population count, and nothing is coerced to zero on the way.
"""

from __future__ import annotations

import polars as pl
import pytest

from web.export.columns import MATRIX_METRICS, PER90_SOURCES, per90_expr
from web.export.normalize import (
    eligible_mask,
    load_frontend_config,
    minutes_floor,
    normalization_basis,
    normalize,
    with_season_to_date,
)

FLOOR = 45


def _rows(specs: list[tuple[int, str, int, float | None]]) -> pl.DataFrame:
    """(element_id, position, minutes, metric) for a single gameweek."""
    return pl.DataFrame(
        {
            "season": ["2025-26"] * len(specs),
            "gw": [1] * len(specs),
            "element_id": [s[0] for s in specs],
            "position": [s[1] for s in specs],
            "minutes": [s[2] for s in specs],
            "n_fixtures": [1] * len(specs),
            "m": [s[3] for s in specs],
        }
    )


def test_z_scores_recover_to_mean_zero_and_unit_sd():
    df = _rows([(i, "MID", 90, float(v)) for i, v in enumerate([1.0, 2.0, 3.0, 4.0, 5.0], start=1)])

    out = normalize(df, ["m"], per_fixture=FLOOR)
    z = out["m_z_pos"].drop_nulls()

    assert z.mean() == pytest.approx(0.0, abs=1e-12)
    assert z.std() == pytest.approx(1.0, abs=1e-12)


def test_below_the_floor_is_null_not_zero():
    """§5.3.3: a z-score of zero means exactly average, which is a
    finding. Null means unknown, which is not the same claim."""
    df = _rows([(1, "MID", 90, 5.0), (2, "MID", 90, 3.0), (3, "MID", 10, 99.0)])

    out = normalize(df, ["m"], per_fixture=FLOOR).sort("element_id")

    assert out["m_z_pos"][2] is None
    assert out["m_pct_pos"][2] is None


def test_a_cameo_does_not_move_the_positional_mean():
    """The stated reason the floor exists at all (§5.7.2)."""
    without = _rows([(1, "MID", 90, 1.0), (2, "MID", 90, 3.0)])
    with_cameo = _rows([(1, "MID", 90, 1.0), (2, "MID", 90, 3.0), (3, "MID", 10, 500.0)])

    a = normalize(without, ["m"], per_fixture=FLOOR).sort("element_id")["m_z_pos"].to_list()
    b = normalize(with_cameo, ["m"], per_fixture=FLOOR).sort("element_id")["m_z_pos"].to_list()

    assert a == pytest.approx(b[:2])


def test_n_pos_counts_the_eligible_population():
    df = _rows([(1, "MID", 90, 1.0), (2, "MID", 90, 2.0), (3, "MID", 5, 3.0), (4, "DEF", 90, 4.0)])

    out = normalize(df, ["m"], per_fixture=FLOOR).sort("element_id")

    assert out["m_n_pos"][0] == 2  # two eligible MIDs, not three
    assert out["m_n_pos"][3] == 1  # the DEF is its own population


def test_positions_are_normalized_separately():
    """The doctrine itself: a defender's xG says nothing about whether he
    is a good defender, so he is never scored against forwards."""
    df = _rows([(1, "FWD", 90, 10.0), (2, "FWD", 90, 12.0), (3, "DEF", 90, 1.0), (4, "DEF", 90, 3.0)])

    out = normalize(df, ["m"], per_fixture=FLOOR).sort("element_id")
    z = out["m_z_pos"].to_list()

    # the lower of each pair sits at the same z despite wildly different raw values
    assert z[0] == pytest.approx(z[2])
    assert z[1] == pytest.approx(z[3])


def test_a_null_metric_contributes_nothing_to_the_group():
    """Four of the sixteen metrics do not exist before 2025-26. A
    null-filled column contributing a zero to its own mean would corrupt
    every other player's z-score in the group."""
    with_null = _rows([(1, "MID", 90, 1.0), (2, "MID", 90, 3.0), (3, "MID", 90, None)])
    without = _rows([(1, "MID", 90, 1.0), (2, "MID", 90, 3.0)])

    a = normalize(with_null, ["m"], per_fixture=FLOOR).sort("element_id")
    b = normalize(without, ["m"], per_fixture=FLOOR).sort("element_id")

    assert a["m_z_pos"][2] is None
    assert a["m_n_pos"][0] == 2
    assert a["m_z_pos"].to_list()[:2] == pytest.approx(b["m_z_pos"].to_list())


def test_zero_variance_group_is_null_rather_than_zero():
    """If every eligible player posted the same value, nobody is above or
    below the mean. Saying so beats emitting a confident 0.0."""
    df = _rows([(1, "MID", 90, 2.0), (2, "MID", 90, 2.0)])

    out = normalize(df, ["m"], per_fixture=FLOOR)

    assert out["m_z_pos"].drop_nulls().len() == 0
    assert out["m_n_pos"][0] == 2  # the population is still real and still reported


def test_a_lone_eligible_player_gets_no_percentile():
    df = _rows([(1, "GK", 90, 4.0)])

    out = normalize(df, ["m"], per_fixture=FLOOR)

    assert out["m_pct_pos"][0] is None
    assert out["m_n_pos"][0] == 1


def test_percentile_is_tie_averaged():
    df = _rows([(1, "MID", 90, 1.0), (2, "MID", 90, 2.0), (3, "MID", 90, 2.0), (4, "MID", 90, 3.0)])

    out = normalize(df, ["m"], per_fixture=FLOOR).sort("element_id")
    pct = out["m_pct_pos"].to_list()

    assert pct[0] == pytest.approx(0.0)
    assert pct[1] == pytest.approx(pct[2])  # tied players share a percentile
    assert pct[3] == pytest.approx(1.0)


# --- §5.15 Q5: the floor scales with football played --------------------


def test_floor_scales_with_fixtures_played():
    assert minutes_floor(1, FLOOR) == 45
    assert minutes_floor(3, FLOOR) == 135
    assert minutes_floor(10, FLOOR) == 450  # the spec's flat number, reached at GW10


def test_a_double_gameweek_raises_the_bar_and_a_blank_does_not():
    """Counted per team fixture, not per gameweek elapsed, so neither is
    punitive."""
    df = pl.DataFrame(
        {
            "season": ["2025-26"] * 2,
            "gw": [1, 2],
            "element_id": [1, 1],
            "position": ["MID"] * 2,
            "minutes": [90, 90],
            "n_fixtures": [1, 2],  # a double in gw2
            "m": [1.0, 1.0],
        }
    )

    out = with_season_to_date(df)

    assert out["cum_fixtures"].to_list() == [1, 3]
    assert minutes_floor(out["cum_fixtures"], FLOOR).to_list() == [45, 135]


def test_eligibility_is_season_to_date_not_per_gameweek():
    """A player who starts every week clears a rising bar; one who plays
    one match in six does not."""
    df = pl.DataFrame(
        {
            "season": ["2025-26"] * 3,
            "gw": [1, 2, 3],
            "element_id": [1, 1, 1],
            "position": ["MID"] * 3,
            "minutes": [90, 0, 0],
            "n_fixtures": [1, 1, 1],
            "m": [1.0, None, None],
        }
    )

    out = with_season_to_date(df)

    assert eligible_mask(out, FLOOR).to_list() == [True, True, False]


def test_the_configured_floor_is_the_one_the_basis_string_reports():
    """A basis the header claims and the numbers do not share is worse
    than no basis at all (§5.7.4)."""
    configured = load_frontend_config()["normalization"]["minutes_per_fixture_floor"]

    assert normalization_basis() == f"within_position_season_to_date_min{configured}_per_fixture"
    assert str(configured) in normalization_basis()


# --- against the committed panel ----------------------------------------


def test_normalization_holds_over_the_real_panel():
    """The §5.11.2 properties over 26,919 real rows rather than a fixture:
    every (gw, position) group recovers mean 0 / sd 1, and no
    below-floor row carries a value."""
    df = pl.read_parquet("data/historical/2024-25.parquet")
    keys = [k for k in MATRIX_METRICS if k in PER90_SOURCES]
    df = df.with_columns([per90_expr(PER90_SOURCES[k], k) for k in keys])

    out = normalize(df, ["xgi_per90"], per_fixture=FLOOR)

    groups = (
        out.filter(pl.col("xgi_per90_z_pos").is_not_null())
        .group_by(["gw", "position"])
        .agg(
            pl.col("xgi_per90_z_pos").mean().alias("mu"),
            pl.col("xgi_per90_z_pos").std().alias("sd"),
        )
    )
    assert groups.height > 100
    assert float(groups["mu"].abs().max()) < 1e-10
    assert float((groups["sd"] - 1).abs().max()) < 1e-10

    below = out.filter(pl.col("cum_minutes") < pl.col("cum_fixtures") * FLOOR)
    assert below.height > 0
    assert below["xgi_per90_z_pos"].drop_nulls().len() == 0
