"""§5.11.1 over `correlations.json`: the population it is computed on,
the null semantics §5.3.3 requires, and the two ways this file could ship
a confident wrong number without raising."""

from __future__ import annotations

import json
import math
from pathlib import Path
from functools import lru_cache
from itertools import combinations

import polars as pl
import pytest

from backtest.report import spearman_with_significance
from web.export.columns import MATRIX_METRICS
from web.export.correlations import (
    GROUPS,
    build_correlations,
    complete_pairs,
    correlate_group,
    correlation_basis,
    player_season_frame,
)
from web.export.contract import CorrelationsFile
from web.export.panel import build_panel

# A metric that exists in every season and one that does not, so the
# availability split the pooled population creates is exercised.
METRICS = ["xg_per90", "tackles_per90"]


@lru_cache(maxsize=1)
def _real() -> CorrelationsFile:
    """The file built over the committed archive, not over the artifact.

    `data/web/v1/panel.parquet` is gitignored (§5.3.4), so a test reading
    it would pass here and fail in a fresh clone. `data/historical/` *is*
    committed, precisely so the harness never depends on an artifact
    someone happens to have built. Cached because every test below wants
    the same file and rebuilding the panel per test is the slow way to
    learn nothing.
    """
    return build_correlations(panel=build_panel())


def _panel_rows(rows: list[dict]) -> pl.DataFrame:
    """Build from an explicit column list rather than from literals.

    `pl.DataFrame(rows, schema=...)` null-fills missing keys silently, so
    a typo in a key name becomes a column of nulls and a passing test.
    """
    columns = [
        "season", "gw", "element_id", "position", "minutes",
        "cum_minutes", "cum_fixtures", "xg_per90", "tackles_per90",
    ]
    for row in rows:
        unknown = set(row) - set(columns)
        assert not unknown, f"unknown key(s) in test row: {unknown}"
    return pl.DataFrame(
        {c: [row.get(c) for row in rows] for c in columns},
        schema={
            "season": pl.Utf8, "gw": pl.Int64, "element_id": pl.Int64,
            "position": pl.Utf8, "minutes": pl.Int64, "cum_minutes": pl.Int64,
            "cum_fixtures": pl.Int64, "xg_per90": pl.Float64,
            "tackles_per90": pl.Float64,
        },
    )


# --- the null-pair hazard ------------------------------------------------


def test_a_perfect_correlation_survives_half_the_column_being_null():
    """The single most dangerous thing this module does.

    `report.spearman` ranks each series independently and polars leaves
    nulls null, while `spearman_with_significance` reports n as `x.len()`
    — the length including the nulls it just ignored. Without pairwise
    filtering a perfect rank correlation is reported as rho=0.23 over
    n=10: no relationship, with a large sample apparently behind it.
    """
    x = pl.Series("x", [1.0, 2.0, 3.0, 4.0, 5.0, None, None, None, None, None])
    y = pl.Series("y", [1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 8.0, 7.0, 6.0, 5.5])

    unfiltered = spearman_with_significance(x, y)
    filtered = spearman_with_significance(*complete_pairs(x, y))

    assert filtered["rho"] == pytest.approx(1.0)
    assert filtered["n"] == 5
    # The bug this guards, stated so a regression reads as a diff:
    assert unfiltered["n"] == 10
    assert unfiltered["rho"] == pytest.approx(0.232, abs=1e-3)


def test_n_counts_complete_pairs_not_group_size():
    """A metric that only exists in one of the pooled seasons must report
    that season's n, not the pooled population's. Reading a 2025-26-only
    correlation as though it had three seasons behind it is exactly the
    overstatement the pooled grain risks."""
    rows = []
    for season in ("2023-24", "2024-25", "2025-26"):
        for element in range(1, 6):
            rows.append({
                "season": season, "gw": 1, "element_id": element, "position": "DEF",
                "minutes": 90, "cum_minutes": 90, "cum_fixtures": 1,
                "xg_per90": 0.1 * element,
                # absent before 2025-26, exactly as the archive has it
                "tackles_per90": float(element) if season == "2025-26" else None,
            })

    people = player_season_frame(_panel_rows(rows), METRICS, per_fixture=45)
    cells = correlate_group(people, METRICS, "DEF")

    assert people.height == 15
    assert len(cells) == 1
    assert cells[0]["n"] == 5, "n must be the 2025-26 rows only"


# --- the season rate ------------------------------------------------------


def test_season_rate_is_minutes_weighted_not_averaged():
    """The mean of weekly per-90 rates is not the season per-90 rate: it
    weights a cameo like a full match. Weighting by minutes recovers the
    exact season figure, `90 * sum(stat) / sum(minutes)`.

    Constructed so the two answers are far apart — one goal in 9 minutes,
    none in 81 — because a test where they nearly agree would not notice
    the difference it exists to protect.
    """
    rows = [
        # 9 minutes, 1 goal  -> 10.0 per 90
        {"season": "2025-26", "gw": 1, "element_id": 1, "position": "FWD",
         "minutes": 9, "cum_minutes": 9, "cum_fixtures": 1,
         "xg_per90": 10.0, "tackles_per90": None},
        # 81 minutes, 0 goals -> 0.0 per 90
        {"season": "2025-26", "gw": 2, "element_id": 1, "position": "FWD",
         "minutes": 81, "cum_minutes": 90, "cum_fixtures": 2,
         "xg_per90": 0.0, "tackles_per90": None},
    ]

    people = player_season_frame(_panel_rows(rows), METRICS, per_fixture=45)

    # 90 * 1 goal / 90 minutes = 1.0
    assert people["xg_per90"][0] == pytest.approx(1.0, abs=1e-12)
    # the naive average would have claimed 5.0
    assert people["xg_per90"][0] != pytest.approx(5.0)


def test_gameweeks_where_the_metric_is_null_leave_the_rate_alone():
    """A null rate contributes neither to the numerator nor to the minutes
    in the denominator. Treating it as a zero would drag a real season
    rate toward whichever weeks happened to be unmeasured."""
    rows = [
        {"season": "2025-26", "gw": 1, "element_id": 1, "position": "MID",
         "minutes": 90, "cum_minutes": 90, "cum_fixtures": 1,
         "xg_per90": 2.0, "tackles_per90": None},
        {"season": "2025-26", "gw": 2, "element_id": 1, "position": "MID",
         "minutes": 90, "cum_minutes": 180, "cum_fixtures": 2,
         "xg_per90": None, "tackles_per90": None},
    ]

    people = player_season_frame(_panel_rows(rows), METRICS, per_fixture=45)

    assert people["xg_per90"][0] == pytest.approx(2.0)


def test_a_metric_absent_all_season_is_null_not_zero():
    """polars sums an all-null column to 0, not to null, so the guarded
    division is what stops 0/0 reaching the export as NaN."""
    rows = [
        {"season": "2023-24", "gw": 1, "element_id": 1, "position": "MID",
         "minutes": 90, "cum_minutes": 90, "cum_fixtures": 1,
         "xg_per90": 0.5, "tackles_per90": None},
    ]

    people = player_season_frame(_panel_rows(rows), METRICS, per_fixture=45)

    assert people["tackles_per90"][0] is None


# --- eligibility (§5.15 Q5) ----------------------------------------------


def test_the_rolling_minutes_floor_excludes_a_bit_part_player():
    """Same floor as normalization, reached through `minutes_floor` rather
    than restated, so the correlation population and the normalization
    population cannot drift apart."""
    rows = [
        {"season": "2025-26", "gw": 10, "element_id": 1, "position": "MID",
         "minutes": 90, "cum_minutes": 900, "cum_fixtures": 10,
         "xg_per90": 0.5, "tackles_per90": 1.0},   # 90/fixture, clears 45
        {"season": "2025-26", "gw": 10, "element_id": 2, "position": "MID",
         "minutes": 10, "cum_minutes": 300, "cum_fixtures": 10,
         "xg_per90": 0.9, "tackles_per90": 2.0},   # 30/fixture, below 45
    ]

    people = player_season_frame(_panel_rows(rows), METRICS, per_fixture=45)

    assert people["element_id"].to_list() == [1]


# --- null rho is not zero rho (§5.3.3) -----------------------------------


def test_a_metric_with_no_spread_yields_null_rho_beside_a_real_n():
    """`saves_per90` is 0.0 for every outfielder — present, not missing.
    The honest cell is a null rho next to the n that explains it: 284
    complete observations and still no correlation defined. A 0.0 would
    read as "measured, no relationship", which is a different claim."""
    rows = [
        {"season": "2025-26", "gw": 1, "element_id": e, "position": "MID",
         "minutes": 90, "cum_minutes": 90, "cum_fixtures": 1,
         "xg_per90": 0.1 * e, "tackles_per90": 0.0}
        for e in range(1, 8)
    ]

    cells = correlate_group(
        player_season_frame(_panel_rows(rows), METRICS, per_fixture=45), METRICS, "MID"
    )

    assert cells[0]["rho"] is None
    assert cells[0]["p_value"] is None
    assert cells[0]["n"] == 7, "n is still reported — the rows exist"


# --- shape ---------------------------------------------------------------


def test_every_unordered_pair_once_and_no_diagonal():
    file = _real()
    metrics, groups = file.metrics, [g.key for g in file.groups]

    assert len(file.cells) == len(list(combinations(metrics, 2))) * len(groups)
    assert not [c for c in file.cells if c.a == c.b], "no diagonal"
    seen = {(c.group, c.a, c.b) for c in file.cells}
    assert not [k for k in seen if (k[0], k[2], k[1]) in seen], "no mirrored pair"


def test_only_the_all_group_is_flagged_mixed_position():
    """§5.7.5's caution attaches to the pooled-position matrix and to
    nothing else. Flagging a single-position group would spend the amber
    §5.8.2 reserves for real epistemic warnings."""
    file = _real()

    assert [g.key for g in file.groups] == GROUPS
    assert [g.key for g in file.groups if g.mixed_position] == ["all"]


def test_the_matrix_covers_every_registered_metric():
    """§5.15 Q2's sixteen, all present.

    This test previously asserted the opposite — that `clean_sheet_prob`
    and `minutes_reliability` were *absent*, because both were registered
    as `source: "model"` and not computed, and a null cell for an unwritten
    model head would have dressed a build gap as a measurement gap. Both
    now exist (`analytics/clean_sheet.py` and
    `features.trailing_minutes_reliability`), so the guard flips: the file
    must cover the whole registry, and a metric quietly dropping out of it
    is the failure now worth catching.
    """
    file = _real()

    assert set(file.metrics) == set(MATRIX_METRICS)
    assert len(file.metrics) == 16
    for key in ("clean_sheet_prob", "minutes_reliability"):
        assert [c for c in file.cells if key in (c.a, c.b)], f"{key} has no cells"


def test_the_basis_and_threshold_travel_in_the_file():
    """A browser reading today's `frontend.yaml` against a file built last
    week would hatch the wrong cells."""
    file = _real()

    assert file.min_n_cell == 30
    assert file.basis == correlation_basis(45)
    assert file.header.normalization_basis == file.basis


def test_nothing_unserializable_reaches_the_json():
    """`float('nan')` serializes to a bare `NaN` token that `JSON.parse`
    rejects, so a degenerate cell would take the whole hero surface down
    rather than render as not-applicable."""
    payload = _real().model_dump_json()

    assert "NaN" not in payload and "Infinity" not in payload
    reparsed = json.loads(payload)
    for cell in reparsed["cells"]:
        assert cell["rho"] is None or math.isfinite(cell["rho"])


def test_the_pooled_population_is_every_season_in_the_panel():
    file = _real()
    by_key = {g.key: g.n_player_seasons for g in file.groups}

    assert len(file.seasons) > 1, "pooled, not single-season"
    assert by_key["all"] == sum(by_key[p] for p in ("GK", "DEF", "MID", "FWD"))


def test_construction_identities_hold_on_the_real_panel():
    """xGI is xG plus xA by definition, so a pipeline that silently
    misaligned players against metrics would break this before it broke
    anything a reader would notice."""
    file = _real()
    cells = {(c.group, c.a, c.b): c for c in file.cells}

    assert cells[("MID", "xg_per90", "xgi_per90")].rho > 0.8
    assert cells[("MID", "xa_per90", "xgi_per90")].rho > 0.7
    assert cells[("DEF", "xgc_per90", "goals_conceded_per90")].rho > 0.8


def test_committed_correlations_json_matches_a_fresh_build():
    """§5.3.4 commits this file so a fresh clone renders the hero surface
    with no pipeline run. That only holds while the committed numbers are
    the ones the current code produces — otherwise the Lab ships one set
    of correlations and the repo believes another, and nothing raises.

    Exact rather than tolerant, and it can afford to be: the panel is
    rebuilt from the committed archive, so the only moving parts are the
    header's timestamp and git sha, which are excluded.
    """
    path = Path("data/web/v1/correlations.json")
    if not path.exists():  # pragma: no cover
        pytest.skip("correlations.json not generated yet -- run `python -m web.export correlations`")

    committed = json.loads(path.read_text(encoding="utf-8"))
    current = json.loads(_real().model_dump_json())
    for payload in (committed, current):
        del payload["header"]

    assert committed == current, "data/web/v1/correlations.json is stale -- regenerate it"
