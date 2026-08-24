import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  byTeam,
  gameweekShape,
  gameweeks,
  nextFixtures,
  nextGameweek,
  nextKickoff,
  runDifficulty,
} from "./fixtures";
import { FixturesFile } from "./schema";

/**
 * The forward-looking helpers, over the real committed schedule.
 *
 * The property worth guarding hardest is the season check. Every other
 * surface in this app reads a file whose season is whatever the panel
 * last recorded; this one reads a file that is always about the season
 * being *played*. Crossing the two would invent blanks and doubles that
 * never happened in the archive, and it would do so silently — the grid
 * would render, the cells would be plausible, and every one of them would
 * be describing a different year's calendar.
 */
const PATH = resolve(__dirname, "../../../../data/web/v1/fixtures.json");
const file = FixturesFile.parse(JSON.parse(readFileSync(PATH, "utf-8")));

describe("the season guard", () => {
  it("returns a schedule for the season the file describes", () => {
    expect(byTeam(file, file.season).size).toBeGreaterThan(0);
  });

  it("returns nothing for any other season", () => {
    // The archive seasons must never be annotated with this calendar.
    for (const season of ["2023-24", "2024-25", "2025-26"]) {
      if (season === file.season) continue;
      expect(byTeam(file, season).size, season).toBe(0);
    }
  });

  it("returns nothing when the season is unknown", () => {
    expect(byTeam(file, null).size).toBe(0);
  });

  it("returns nothing when the file is missing", () => {
    // A failed fetch costs the fixture columns, never the surface.
    expect(byTeam(null, file.season).size).toBe(0);
    expect(nextGameweek(null)).toBe(null);
    expect(nextKickoff(null)).toBe(null);
    expect(gameweeks(null)).toEqual([]);
  });
});

describe("the schedule", () => {
  const schedule = byTeam(file, file.season);

  it("gives every club a full set of fixtures", () => {
    // Twenty clubs, and a full season is 38 each unless the file is
    // partial. Both sides of every fixture are recorded.
    expect(schedule.size).toBe(20);
    const total = [...schedule.values()].reduce((sum, list) => sum + list.length, 0);
    expect(total).toBe(file.fixtures.filter((row) => row.gw !== null).length * 2);
  });

  it("orders each club's fixtures by gameweek", () => {
    for (const [team, list] of schedule) {
      const weeks = list.map((entry) => entry.gw);
      expect(weeks, team).toEqual([...weeks].sort((a, b) => a - b));
    }
  });

  it("records each fixture from both clubs' points of view", () => {
    const row = file.fixtures.find((entry) => entry.gw !== null)!;
    const home = schedule.get(row.team_h)!.find((entry) => entry.gw === row.gw)!;
    const away = schedule.get(row.team_a)!.find((entry) => entry.gw === row.gw)!;

    expect(home.home).toBe(true);
    expect(home.opponent).toBe(row.team_a);
    expect(away.home).toBe(false);
    expect(away.opponent).toBe(row.team_h);
    // Each side carries its *own* difficulty, not the fixture's.
    expect(home.difficulty).toBe(row.team_h_difficulty);
    expect(away.difficulty).toBe(row.team_a_difficulty);
  });
});

describe("what comes next", () => {
  const schedule = byTeam(file, file.season);

  it("finds the earliest gameweek with anything left to play", () => {
    const next = nextGameweek(file);
    expect(next).not.toBe(null);
    // Nothing before it can still be unplayed, or it would be next.
    const earlierUnplayed = file.fixtures.filter(
      (row) => !row.played && row.gw !== null && row.gw < next!,
    );
    expect(earlierUnplayed).toEqual([]);
  });

  it("finds the soonest unplayed kickoff", () => {
    const kickoff = nextKickoff(file);
    if (kickoff === null) return;
    const earlier = file.fixtures.filter(
      (row) => !row.played && row.kickoff_time !== null && row.kickoff_time < kickoff,
    );
    expect(earlier).toEqual([]);
  });

  it("counts matches rather than gameweeks", () => {
    /*
     * "The next five fixtures" is five *matches*. A club with a blank in
     * the middle of the run reaches five a week later than one without,
     * and a double gets there sooner — collapsing that to five gameweeks
     * would describe a schedule nobody has.
     */
    for (const team of schedule.keys()) {
      const run = nextFixtures(schedule, team, 5);
      expect(run.length).toBeLessThanOrEqual(5);
      expect(run.every((entry) => !entry.played)).toBe(true);
    }
  });

  it("never includes a played fixture in a run", () => {
    // A run computed from matches already in the books would describe the
    // past while looking like a projection.
    const played = file.fixtures.filter((row) => row.played);
    if (played.length === 0) return;
    const team = played[0]!.team_h;
    expect(nextFixtures(schedule, team, 10).some((entry) => entry.played)).toBe(false);
  });

  it("averages difficulty only over fixtures that have one", () => {
    for (const team of schedule.keys()) {
      const run = runDifficulty(schedule, team, 5);
      if (run === null) continue;
      // FPL's scale is 1 to 5, so any mean of it must sit inside that.
      expect(run).toBeGreaterThanOrEqual(1);
      expect(run).toBeLessThanOrEqual(5);
    }
  });
});

describe("blanks and doubles", () => {
  const schedule = byTeam(file, file.season);

  it("splits every club into exactly one state per gameweek", () => {
    for (const gw of gameweeks(file)) {
      const { blank, double } = gameweekShape(schedule, gw);
      // A club cannot both have no fixture and have two.
      for (const team of blank) expect(double.has(team)).toBe(false);
      // And the counts have to add up against the fixtures themselves.
      const inWeek = file.fixtures.filter((row) => row.gw === gw);
      const playing = new Set<string>();
      for (const row of inWeek) {
        playing.add(row.team_h);
        playing.add(row.team_a);
      }
      expect(blank.size).toBe(schedule.size - playing.size);
    }
  });

  it("finds no blank in a full gameweek", () => {
    // A round with ten fixtures involves all twenty clubs.
    const full = gameweeks(file).find(
      (gw) => file.fixtures.filter((row) => row.gw === gw).length === 10,
    );
    if (full === undefined) return;
    expect(gameweekShape(schedule, full).blank.size).toBe(0);
  });
});
