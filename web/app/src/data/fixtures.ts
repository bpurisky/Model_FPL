/**
 * `fixtures.json`, and the forward-looking questions it answers.
 *
 * This file was exported, committed, schema'd and contract-tested from
 * 5A onward, and until now **nothing read it**. Every surface in the app
 * reads player history; the one export that is entirely about the season
 * being played sat unused. That is the single clearest way the frontend
 * had drifted toward the archive and away from the season the reader is
 * actually in.
 *
 * Two rules govern everything here, and both are about not lying with a
 * schedule:
 *
 * **Fixtures belong to one season, and it is the current one.**
 * `fixtures.json` carries `season` and the export builds it for
 * `CURRENT_SEASON`. Applying its schedule to a row from 2024-25 would
 * invent blanks and doubles that never happened, so every helper here
 * takes the season being displayed and returns nothing when it does not
 * match. A surface showing the archive gets no fixture annotations rather
 * than wrong ones.
 *
 * **A fixture that has been played is not a forecast.** `difficulty_basis`
 * distinguishes what a club carried *into* a match from what it holds
 * today, and "the next N fixtures" means the next N *unplayed* ones. A
 * run computed from matches already in the books would be describing the
 * past while looking like a projection.
 */

import { useEffect, useState } from "react";
import { loadFixtures } from "./load";
import type { FixturesFile, FixtureRow } from "./schema";

let pending: Promise<FixturesFile | null> | null = null;

function file(): Promise<FixturesFile | null> {
  // Failing to `null` rather than throwing: a missing schedule costs the
  // fixture columns, never the surface they sit on.
  pending ??= loadFixtures().catch(() => null);
  return pending;
}

export function useFixtures(): FixturesFile | null {
  const [loaded, setLoaded] = useState<FixturesFile | null>(null);

  useEffect(() => {
    let cancelled = false;
    file().then((result) => {
      if (!cancelled) setLoaded(result);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return loaded;
}

/** One club's involvement in one gameweek. */
export interface TeamFixture {
  gw: number;
  opponent: string;
  home: boolean;
  /** FPL's published 1–5 rating, from this club's point of view. */
  difficulty: number | null;
  /** This repo's Elo-derived rating, on the same 1–5 scale. */
  custom: number | null;
  played: boolean;
  kickoff: string | null;
}

/**
 * The next gameweek with anything still to play.
 *
 * This is the app's only notion of "now". Everything else it renders is
 * keyed to the last gameweek the *panel* holds, which is by construction
 * the last one that finished — so without this the app has no way to
 * refer to the week the reader is actually in.
 *
 * Null once the season is over, which is a real state and not an error.
 */
export function nextGameweek(fixtures: FixturesFile | null): number | null {
  if (!fixtures) return null;
  let earliest: number | null = null;
  for (const row of fixtures.fixtures) {
    if (row.played || row.gw === null) continue;
    if (earliest === null || row.gw < earliest) earliest = row.gw;
  }
  return earliest;
}

/** The next unplayed kickoff, which is the closest thing to a deadline. */
export function nextKickoff(fixtures: FixturesFile | null): string | null {
  if (!fixtures) return null;
  let soonest: string | null = null;
  for (const row of fixtures.fixtures) {
    if (row.played || !row.kickoff_time) continue;
    if (soonest === null || row.kickoff_time < soonest) soonest = row.kickoff_time;
  }
  return soonest;
}

/**
 * Every fixture a club has, by gameweek, from its own point of view.
 *
 * Returns an empty map when `season` is not the one the file describes —
 * see the module docstring. The caller does not have to remember.
 */
export function byTeam(
  fixtures: FixturesFile | null,
  season: string | null,
): Map<string, TeamFixture[]> {
  const out = new Map<string, TeamFixture[]>();
  if (!fixtures || season === null || season !== fixtures.season) return out;

  const push = (team: string, entry: TeamFixture) => {
    const list = out.get(team);
    if (list) list.push(entry);
    else out.set(team, [entry]);
  };

  for (const row of fixtures.fixtures) {
    if (row.gw === null) continue;
    push(row.team_h, {
      gw: row.gw,
      opponent: row.team_a,
      home: true,
      difficulty: row.team_h_difficulty,
      custom: row.custom_difficulty_home,
      played: row.played,
      kickoff: row.kickoff_time,
    });
    push(row.team_a, {
      gw: row.gw,
      opponent: row.team_h,
      home: false,
      difficulty: row.team_a_difficulty,
      custom: row.custom_difficulty_away,
      played: row.played,
      kickoff: row.kickoff_time,
    });
  }

  for (const list of out.values()) list.sort((left, right) => left.gw - right.gw);
  return out;
}

/**
 * A club's next `count` unplayed fixtures.
 *
 * A double gameweek contributes two entries and a blank contributes none,
 * which is the honest shape: "the next five fixtures" is a count of
 * matches, not of gameweeks, and a club with a blank in the middle of the
 * run reaches five matches a week later than one without.
 */
export function nextFixtures(
  schedule: Map<string, TeamFixture[]>,
  team: string,
  count: number,
): TeamFixture[] {
  return (schedule.get(team) ?? []).filter((entry) => !entry.played).slice(0, count);
}

/** Which of the two difficulty ratings a caller wants. */
export type DifficultyBasis = "fpl" | "custom";

/**
 * Mean difficulty over a club's next `count` fixtures, or null when it
 * has none left.
 *
 * The basis is a parameter rather than a fixed choice, and it has to be:
 * a surface that recolours its cells by Elo while ranking its rows by
 * FPL is showing two different opinions in one table and labelling the
 * whole thing with one of them. The two ratings genuinely disagree —
 * that disagreement is worth being able to see, which is the only reason
 * both are exported.
 *
 * `panel.py:add_model_columns` explains why the *model* uses FPL's: it
 * measured better and it is committed data, where the Elo derives from a
 * gitignored cache. That argument is about what to build a projection
 * on, not about what a reader may look at.
 */
export function runDifficulty(
  schedule: Map<string, TeamFixture[]>,
  team: string,
  count: number,
  basis: DifficultyBasis = "fpl",
): number | null {
  const pick = (entry: TeamFixture) => (basis === "fpl" ? entry.difficulty : entry.custom);
  const run = nextFixtures(schedule, team, count).filter((entry) => pick(entry) !== null);
  if (run.length === 0) return null;
  return run.reduce((total, entry) => total + pick(entry)!, 0) / run.length;
}

/**
 * Which clubs have no fixture in a gameweek, and which have more than
 * one. Both are §5.4.3's markers, and neither is knowable from the panel:
 * a blank leaves no row at all, so the Form Matrix cannot tell "the club
 * did not play" from "this player was not in the squad" without asking
 * the schedule.
 */
export function gameweekShape(
  schedule: Map<string, TeamFixture[]>,
  gw: number,
): { blank: Set<string>; double: Set<string> } {
  const blank = new Set<string>();
  const double = new Set<string>();
  for (const [team, list] of schedule) {
    const inWeek = list.filter((entry) => entry.gw === gw).length;
    if (inWeek === 0) blank.add(team);
    else if (inWeek > 1) double.add(team);
  }
  return { blank, double };
}

/** Every gameweek the schedule covers, in order. */
export function gameweeks(fixtures: FixturesFile | null): number[] {
  if (!fixtures) return [];
  const seen = new Set<number>();
  for (const row of fixtures.fixtures) if (row.gw !== null) seen.add(row.gw);
  return [...seen].sort((left, right) => left - right);
}

export type { FixtureRow };
