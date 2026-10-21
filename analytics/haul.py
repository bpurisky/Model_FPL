"""P(haul): the chance a player scores a haul in a gameweek, from the
event model's own heads.

**A haul is HAUL_THRESHOLD or more FPL points in the gameweek**, the
definition the captaincy table and the paper trade's calibration both use.

**Method: exact enumeration over the model's per-component expectations.**
The event model (`analytics.projections`) does not produce a points
distribution, only expectations per head: a minutes distribution
(`p_blank`/`p_short`/`p_full`), a goals and an assists rate, a clean-sheet
probability from the scoreline model, and trailing means for the rest. This
module builds the distribution those imply, for each fixture:

- minutes bucket ~ Categorical(p_blank, p_short, p_full), scored as the
  scoring config scores it;
- goals ~ Poisson and assists ~ Poisson, each at the model's expected count
  for the gameweek (`{goals,assists}_trailing * attack_mult`), split evenly
  over the fixtures and conditioned on the player appearing (the trailing
  columns are already scaled by P(plays), so that is divided back out);
- clean sheet ~ Bernoulli(`expected_cs` / fixtures), earned on a 60+ bucket
  only, exactly as `expected_points_by_component` gates it;
- a double gameweek is two independent fixtures, convolved (summed).

The points distribution is enumerated exactly (Poisson counts truncated at
MAX_COUNT with the tail folded into the last count), so there is no
sampling noise and no seed: the same row always gives the same answer. By
construction the distribution's mean equals the model's xP (the tests check
that), so P(haul) is the same model's tail, not a second model. `p_low` is
the same distribution's P(points <= LOW_THRESHOLD).

**Bonus is not a constant.** A hauling match is mostly a goal contribution
plus 1-3 bonus points, so a constant mean bonus badly understates the tail
(an earlier constant-bonus version predicted 0.4% for players who hauled 2.2%
of the time). `BonusModel` supplies P(bonus | position, goals + assists in the
match) from appearances in archived seasons strictly before the projected
one (`load_bonus_model`), and the model's own `bonus_trailing` sets the level:
the history gives the shape, a constant shift keeps the mean at the model's.
Without a bonus model (no earlier season) bonus falls back to its mean.

**What is deliberately ignored.** These components enter as their
expectation, a constant added when the player appears, not as a random
draw: goals-conceded deductions, saves, cards, own goals and penalty
misses/saves, defensive contribution. Goals, assists and the clean sheet are
treated as independent within a fixture although they share a match, and
the Poisson rates are the model's point estimates, with no extra
dispersion for uncertainty in the rate itself (a negative-binomial variant
moved nothing in a 2024-25 backtest, so it is not carried). A goalkeeper's
save count has no variance here, so a goalkeeper's P(haul) is a floor.
Penalty-taker status is not a separate input: it is already inside the
trailing xG. A player the model gives no chance of appearing has P(haul) 0
even if his xP is not exactly 0.

**Calibration, measured not asserted.** Run walk-forward over every
gameweek from 4 on (training strictly before each, bonus shape from earlier
seasons only), for players with xP >= 1: on 2024-25 the top 20 by xP per
gameweek had mean P(haul) 15.4% against a realized 15.6% (n=700), Brier
0.122 against 0.131 for always predicting the base rate; on 2025-26, 11.9%
against 12.4%, Brier 0.106 against 0.109. It is under-confident at the very
top (top 5: 17.9% against 22.3% in 2025-26, n=175) and under-predicts for
low-xP players (all xP >= 1: 2.9% against 4.4%). That is a weaker claim than
"calibrated", and the paper trade scores every frozen P(haul) live (Brier
against the base rate).

Pure and leak-safe: it reads only the projected row it is handed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import polars as pl

HAUL_THRESHOLD = 10
LOW_THRESHOLD = 2
MAX_COUNT = 8  # goals or assists per fixture beyond which the tail is folded in

HISTORICAL_DIR = Path("data/historical")
BONUS_LEVELS = (0, 1, 2, 3)

_EPS = 1e-9


def _poisson(lam: float) -> list[float]:
    """P(K = k) for k = 0..MAX_COUNT, the last entry holding the tail."""
    if lam <= 0:
        return [1.0] + [0.0] * MAX_COUNT
    pmf = [math.exp(-lam) * lam**k / math.factorial(k) for k in range(MAX_COUNT)]
    pmf.append(max(0.0, 1.0 - sum(pmf)))
    return pmf


class BonusModel:
    """P(bonus points | position, goals + assists in the match), from
    appearances in earlier seasons. Bonus is the biggest random component
    the model carries only as a mean, and it is concentrated on the
    matches with a goal contribution, which are the ones that become
    hauls: 1 to 3 points on top of a 5 to 8 point return decides whether a
    match reaches 10. The conditional shape comes from history; its level
    is the model's own (`haul` shifts it so the mean stays `bonus_trailing`).
    """

    def __init__(self, table: dict[tuple[str, int], list[float]]):
        self.table = table

    @staticmethod
    def bucket(goal_contributions: int) -> int:
        return min(goal_contributions, 2)

    def pmf(self, position: str, goal_contributions: int) -> list[float]:
        return self.table[(position, self.bucket(goal_contributions))]

    def mean(self, position: str, goal_contributions: int) -> float:
        return sum(b * p for b, p in zip(BONUS_LEVELS, self.pmf(position, goal_contributions)))


def load_bonus_model(before_season: str, historical_dir: Path = HISTORICAL_DIR) -> BonusModel | None:
    """Fit on every archived season strictly before `before_season` (the
    season being projected, so nothing from it or after it is read). None
    when there is no earlier season."""
    frames = [
        pl.read_parquet(path, columns=["position", "minutes", "goals_scored", "assists", "bonus"])
        for path in sorted(historical_dir.glob("20*-*.parquet"))
        if path.stem < before_season
    ]
    if not frames:
        return None
    played = (
        pl.concat(frames)
        .filter(pl.col("minutes") > 0)
        .with_columns(
            pl.min_horizontal(pl.col("goals_scored") + pl.col("assists"), pl.lit(2)).alias("bucket"),
            pl.col("bonus").clip(0, 3),
        )
    )
    table: dict[tuple[str, int], list[float]] = {}
    for (position, bucket), group in played.group_by("position", "bucket"):
        counts = [int((group["bonus"] == b).sum()) for b in BONUS_LEVELS]
        total = sum(counts)
        table[(position, int(bucket))] = [c / total for c in counts]
    for position in ("GK", "DEF", "MID", "FWD"):
        for bucket in (0, 1, 2):
            table.setdefault((position, bucket), table.get((position, min(bucket, 1)), [1.0, 0.0, 0.0, 0.0]))
    return BonusModel(table)


def _fixture_pmf(row: dict[str, Any], config: dict[str, Any], bonus_model: BonusModel | None) -> dict[float, float]:
    """points -> probability for one fixture. Keys are rounded so equal
    totals reached by different routes merge."""
    n = int(row["n_fixtures"])
    position = row["position"]
    mins = config["minutes"]
    p_blank, p_short, p_full = float(row["p_blank"]), float(row["p_short"]), float(row["p_full"])
    p_play = 1.0 - p_blank

    lam_goals = row["expected_goals_trailing"] * row["attack_mult"] / n / p_play
    lam_assists = row["expected_assists_trailing"] * row["attack_mult"] / n / p_play
    p_cs = min(1.0, max(0.0, row["expected_cs"] / n))

    gc_cfg = config["goals_conceded"]
    constant = 0.0
    if position in gc_cfg["positions"]:
        constant += row["p_full"] * row["expected_gc_deductions"] / n * gc_cfg["points"] / p_play
    bonus_target = row["bonus_trailing"] / p_play
    constant += (
        row["own_goals_trailing"] * config["own_goals"]
        + row["penalties_missed_trailing"] * config["penalties_missed"]
        + row["penalties_saved_trailing"] * config["penalties_saved"]
        + row["yellow_cards_trailing"] * config["yellow_cards"]
        + row["red_cards_trailing"] * config["red_cards"]
    ) / p_play
    if position == "GK":
        saves_cfg = config["saves"]
        expected_saves = row["saves_trailing"] * row["defence_mult"] / n / p_play
        constant += (
            expected_saves / saves_cfg["n"] * saves_cfg["points"]
            if saves_cfg["mode"] == "per_n"
            else expected_saves * saves_cfg["flat_rate"]
        )
    dc_cfg = config.get("defensive_contribution")
    if dc_cfg:
        constant += (row.get("p_dc_threshold_trailing") or 0.0) * dc_cfg["points"] / p_play

    goal_pts, assist_pts = config["goals_scored"][position], config["assists"][position]
    # Attacking returns (and, with a bonus model, the bonus they bring),
    # independent of the minutes bucket given an appearance.
    attack: dict[float, float] = {}
    expected_bonus = 0.0
    for g, pg in enumerate(_poisson(lam_goals)):
        for a, pa in enumerate(_poisson(lam_assists)):
            base = g * goal_pts + a * assist_pts
            if bonus_model is None:
                attack[base] = attack.get(base, 0.0) + pg * pa
                continue
            for level, pb in zip(BONUS_LEVELS, bonus_model.pmf(position, g + a)):
                if pb > 0:
                    attack[base + level] = attack.get(base + level, 0.0) + pg * pa * pb
            expected_bonus += pg * pa * bonus_model.mean(position, g + a)
    # Keep the mean at the model's own bonus rate: the history supplies the
    # shape, the model the level.
    constant += bonus_target - expected_bonus

    pmf: dict[float, float] = {}

    def add(points: float, prob: float) -> None:
        if prob > 0:
            key = round(points, 9)
            pmf[key] = pmf.get(key, 0.0) + prob

    add(float(mins["none"]), p_blank)
    for bucket_p, bucket_pts, can_cs in ((p_short, mins["short"], False), (p_full, mins["full"], True)):
        if bucket_p <= 0:
            continue
        for pts, prob in attack.items():
            base = bucket_pts + pts + constant
            if can_cs and p_cs > 0:
                add(base + config["clean_sheets"][position], bucket_p * prob * p_cs)
                add(base, bucket_p * prob * (1 - p_cs))
            else:
                add(base, bucket_p * prob)
    return pmf


def _convolve(left: dict[float, float], right: dict[float, float]) -> dict[float, float]:
    out: dict[float, float] = {}
    for a, pa in left.items():
        for b, pb in right.items():
            key = round(a + b, 9)
            out[key] = out.get(key, 0.0) + pa * pb
    return out


def points_distribution(
    row: dict[str, Any], config: dict[str, Any], bonus_model: BonusModel | None = None
) -> dict[float, float]:
    """points -> probability for the player's whole gameweek. `row` is one
    row of `project_event_vectors`' output joined with the scoreline
    columns (`with_scoreline`). A blank gameweek, or a player with no
    chance of appearing, is 0 points with probability 1."""
    n = int(row.get("n_fixtures") or 0)
    if n == 0 or 1.0 - float(row["p_blank"]) <= _EPS:
        return {0.0: 1.0}
    one = _fixture_pmf(row, config, bonus_model)
    total = one
    for _ in range(n - 1):
        total = _convolve(total, one)
    return total


def haul_summary(
    row: dict[str, Any], config: dict[str, Any], bonus_model: BonusModel | None = None
) -> dict[str, float]:
    """p_haul, p_low and the distribution's mean for one projected row.

    Thresholds are compared at the half point (a total of 9.5 or more counts
    as a haul, under 2.5 as low): real points are whole numbers, and the
    expectation-only components shift a total by a fraction that would
    otherwise push an exact 10 under the line or an exact 2 over it."""
    dist = points_distribution(row, config, bonus_model)
    mass = sum(dist.values())
    return {
        "p_haul": sum(p for x, p in dist.items() if x >= HAUL_THRESHOLD - 0.5) / mass,
        "p_low": sum(p for x, p in dist.items() if x < LOW_THRESHOLD + 0.5) / mass,
        "mean": sum(x * p for x, p in dist.items()) / mass,
    }
