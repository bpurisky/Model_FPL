/**
 * Window scoring, rotation pairs and blank/double badges for the Fixture
 * Ticker — the pure logic, kept out of the view so it can be tested
 * without a DOM.
 *
 * Everything here is descriptive in §5.0.2's sense: it ranks *clubs* on
 * the schedule they face. It never names a player, a transfer or a squad.
 * "Rotation pair" means two clubs whose fixtures interlock, not a
 * recommendation to own either.
 *
 * **A blank is worse than any fixture.** A club with no match in a
 * gameweek earns nothing that week, so treating the gap as "no
 * difficulty" (zero, or skipping it) would rank a blank-heavy club as
 * the kindest run on the table — the exact error a naive mean makes.
 * Instead a blank counts as one slot at `BLANK_VALUE`, which sits above
 * the hardest rating (5). Consequences, stated rather than hidden:
 *
 * - under the *mean*, a blank always drags a club's window average up,
 *   and a double contributes two fixtures to the average (so a double of
 *   easy fixtures improves it, a double of hard ones worsens it);
 * - under the *sum*, a double adds two ratings and so can total more than
 *   a blank. Sum is the "how much difficulty in total" view and rewards
 *   fewer matches by construction; mean is the default for that reason.
 *
 * A fixture whose rating is null (the Elo column for a club the model
 * could not seed) is *unrated*, not blank: it is left out of the score
 * rather than counted as either easy or hard, and a window with nothing
 * rated scores null.
 *
 * **Attack-versus-defence split: deliberately absent.** `fixtures.json`
 * carries one scalar per side per fixture (FPL's integer, and an Elo
 * expectation from `analytics/fdr.py:custom_difficulty`). There is no
 * separate attacking or defensive strength anywhere in the export, and
 * deriving one is new modelling, not a split of a quantity already
 * computed.
 */

import { gameweekShape, type DifficultyBasis, type TeamFixture } from "./fixtures";

type Schedule = Map<string, TeamFixture[]>;

/** Above FPL's hardest rating (5) and Elo's 1–5 scale ceiling. */
export const BLANK_VALUE = 6;

/** The longest window the ticker offers. */
export const MAX_WINDOW = 8;

export type Aggregate = "mean" | "sum";

const rating = (entry: TeamFixture, basis: DifficultyBasis): number | null =>
  basis === "fpl" ? entry.difficulty : entry.custom;

/**
 * The gameweeks of a window: `length` gameweeks from `from`, drawn from
 * the schedule's own gameweek list so a gap in numbering cannot produce
 * a phantom week. Length is clamped to 1–`MAX_WINDOW`.
 */
export function windowGameweeks(all: number[], from: number, length: number): number[] {
  const clamped = Math.min(MAX_WINDOW, Math.max(1, Math.floor(length)));
  return all.filter((gw) => gw >= from).slice(0, clamped);
}

/**
 * One club's difficulty slots across a window: every rated fixture, plus
 * one `BLANK_VALUE` per blank gameweek.
 */
function slots(
  schedule: Schedule,
  team: string,
  gws: number[],
  basis: DifficultyBasis,
): { values: number[]; rated: boolean } {
  const list = schedule.get(team) ?? [];
  const values: number[] = [];
  let rated = false;
  for (const gw of gws) {
    const inWeek = list.filter((entry) => entry.gw === gw);
    if (inWeek.length === 0) values.push(BLANK_VALUE);
    for (const entry of inWeek) {
      const value = rating(entry, basis);
      if (value === null) continue;
      values.push(value);
      rated = true;
    }
  }
  return { values, rated };
}

/** A club's score over a window, or null when nothing in it is rated. */
export function windowScore(
  schedule: Schedule,
  team: string,
  gws: number[],
  basis: DifficultyBasis,
  aggregate: Aggregate = "mean",
): number | null {
  const { values, rated } = slots(schedule, team, gws, basis);
  // Blanks alone are not a score: a club with no rated fixture at all in
  // the window has nothing to rank on.
  if (!rated) return null;
  const total = values.reduce((acc, value) => acc + value, 0);
  return aggregate === "sum" ? total : total / values.length;
}

/** Clubs ranked kindest first; unscorable clubs last, then by name. */
export function rankClubs(
  schedule: Schedule,
  gws: number[],
  basis: DifficultyBasis,
  aggregate: Aggregate = "mean",
): { team: string; score: number | null }[] {
  return [...schedule.keys()]
    .map((team) => ({ team, score: windowScore(schedule, team, gws, basis, aggregate) }))
    .sort((left, right) => {
      if (left.score === null && right.score === null) return left.team.localeCompare(right.team);
      if (left.score === null) return 1;
      if (right.score === null) return -1;
      return left.score - right.score || left.team.localeCompare(right.team);
    });
}

/**
 * A club's difficulty in one gameweek as a single number: the mean of its
 * rated fixtures that week (a double is its average), `BLANK_VALUE` for a
 * blank, null when it plays but nothing is rated.
 */
export function gameweekValue(
  schedule: Schedule,
  team: string,
  gw: number,
  basis: DifficultyBasis,
): number | null {
  const inWeek = (schedule.get(team) ?? []).filter((entry) => entry.gw === gw);
  if (inWeek.length === 0) return BLANK_VALUE;
  const rated = inWeek.map((entry) => rating(entry, basis)).filter((v): v is number => v !== null);
  if (rated.length === 0) return null;
  return rated.reduce((acc, value) => acc + value, 0) / rated.length;
}

export interface RotationPick {
  gw: number;
  /** The club whose fixture is taken that week; null when neither is rated. */
  team: string | null;
  value: number | null;
}

export interface RotationPair {
  a: string;
  b: string;
  /** Mean over the window of the easier club's value each week. */
  score: number;
  picks: RotationPick[];
}

/**
 * The best pairs of clubs for rotation: each gameweek takes the easier of
 * the two, and the pair's score is the mean of those picks. Lower is
 * kinder. A week where both clubs are blank scores `BLANK_VALUE`; a week
 * where one side is unrated uses the other; a week with neither rated is
 * left out of the mean.
 *
 * Ties on a week go to the club that sorts first by name, and ties on
 * score to the pair that sorts first, so the output is deterministic.
 */
export function rotationPairs(
  schedule: Schedule,
  gws: number[],
  basis: DifficultyBasis,
  top = 10,
): RotationPair[] {
  const teams = [...schedule.keys()].sort((left, right) => left.localeCompare(right));
  const values = new Map(
    teams.map((team) => [team, gws.map((gw) => gameweekValue(schedule, team, gw, basis))]),
  );
  const pairs: RotationPair[] = [];

  for (let i = 0; i < teams.length; i++) {
    for (let j = i + 1; j < teams.length; j++) {
      const a = teams[i]!;
      const b = teams[j]!;
      const va = values.get(a)!;
      const vb = values.get(b)!;
      const picks: RotationPick[] = gws.map((gw, k) => {
        const x = va[k] ?? null;
        const y = vb[k] ?? null;
        if (x === null && y === null) return { gw, team: null, value: null };
        if (y === null || (x !== null && x <= y)) return { gw, team: a, value: x };
        return { gw, team: b, value: y };
      });
      const scored = picks.filter((pick) => pick.value !== null);
      if (scored.length === 0) continue;
      const score = scored.reduce((acc, pick) => acc + pick.value!, 0) / scored.length;
      pairs.push({ a, b, score, picks });
    }
  }

  return pairs
    .sort(
      (left, right) =>
        left.score - right.score ||
        left.a.localeCompare(right.a) ||
        left.b.localeCompare(right.b),
    )
    .slice(0, Math.max(0, top));
}

export interface GameweekBadge {
  gw: number;
  blank: string[];
  double: string[];
}

/** Blank and double clubs for each gameweek of a window, names sorted. */
export function gameweekBadges(schedule: Schedule, gws: number[]): GameweekBadge[] {
  return gws.map((gw) => {
    const shape = gameweekShape(schedule, gw);
    return {
      gw,
      blank: [...shape.blank].sort((l, r) => l.localeCompare(r)),
      double: [...shape.double].sort((l, r) => l.localeCompare(r)),
    };
  });
}
