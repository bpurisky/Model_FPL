"""squad.simulate's scoring rules: what a team sheet really scored."""

from __future__ import annotations

from squad.simulate import score_week, selling_price

POSITIONS = {
    1: "GK", 2: "DEF", 3: "DEF", 4: "DEF", 5: "MID", 6: "MID", 7: "MID", 8: "MID", 9: "FWD", 10: "FWD", 11: "FWD",
    12: "GK", 13: "DEF", 14: "MID", 15: "FWD",
}
XI = frozenset(range(1, 12))
BENCH = (12, 13, 14, 15)


def _everyone_plays(points: int = 2) -> dict[int, tuple[int, int]]:
    return {e: (points, 90) for e in POSITIONS}


def test_captain_doubles():
    assert score_week(XI, BENCH, captain=9, vice=5, positions=POSITIONS, actual=_everyone_plays()) == 11 * 2 + 2


def test_vice_captain_doubles_when_the_captain_does_not_play():
    actual = _everyone_plays()
    actual[9] = (0, 0)
    actual[5] = (10, 90)
    # 9 is subbed by the first outfield sub that keeps the formation (13)
    assert score_week(XI, BENCH, captain=9, vice=5, positions=POSITIONS, actual=actual) == 9 * 2 + 10 + 2 + 10


def test_autosubs_respect_the_formation():
    """Losing a defender from a three-man back line: the defender on the
    bench comes on even though a midfielder sits ahead of him."""
    bench = (12, 14, 13, 15)
    actual = _everyone_plays()
    actual[2] = (0, 0)
    actual[13] = (7, 90)
    points = score_week(XI, bench, captain=9, vice=5, positions=POSITIONS, actual=actual)
    assert points == 10 * 2 + 7 + 2


def test_goalkeeper_only_replaces_goalkeeper():
    actual = _everyone_plays()
    actual[1] = (0, 0)
    actual[12] = (6, 90)
    assert score_week(XI, BENCH, captain=9, vice=5, positions=POSITIONS, actual=actual) == 10 * 2 + 6 + 2


def test_selling_price_keeps_half_a_rise_and_all_of_a_fall():
    assert selling_price(purchase=50, now=53) == 51
    assert selling_price(purchase=50, now=54) == 52
    assert selling_price(purchase=50, now=48) == 48
