import { describe, expect, it } from "vitest";
import type { TeamFixture } from "./fixtures";
import {
  BLANK_VALUE,
  gameweekBadges,
  gameweekValue,
  rankClubs,
  rotationPairs,
  windowGameweeks,
  windowScore,
} from "./fixtureWindow";

const fx = (gw: number, difficulty: number | null, custom: number | null = difficulty): TeamFixture => ({
  gw,
  opponent: "X",
  home: true,
  difficulty,
  custom,
  played: false,
  kickoff: null,
});

/** Easy: 2,2,2. Hard: 4,4,4. Blanky: 1 then blank then 1. Dgw: 3, then 1+5, then 3. */
const schedule = new Map<string, TeamFixture[]>([
  ["Easy", [fx(1, 2), fx(2, 2), fx(3, 2)]],
  ["Hard", [fx(1, 4), fx(2, 4), fx(3, 4)]],
  ["Blanky", [fx(1, 1), fx(3, 1)]],
  ["Dgw", [fx(1, 3), fx(2, 1), fx(2, 5), fx(3, 3)]],
]);
const gws = [1, 2, 3];

describe("windowGameweeks", () => {
  it("starts at the given week and clamps the length to 1-8", () => {
    const all = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    expect(windowGameweeks(all, 3, 4)).toEqual([3, 4, 5, 6]);
    expect(windowGameweeks(all, 1, 99)).toHaveLength(8);
    expect(windowGameweeks(all, 1, 0)).toEqual([1]);
    expect(windowGameweeks(all, 9, 5)).toEqual([9, 10]);
  });
});

describe("windowScore", () => {
  it("is the mean or sum of ratings", () => {
    expect(windowScore(schedule, "Easy", gws, "fpl")).toBe(2);
    expect(windowScore(schedule, "Easy", gws, "fpl", "sum")).toBe(6);
  });

  it("counts both fixtures of a double", () => {
    // 3, 1, 5, 3 -> mean 3
    expect(windowScore(schedule, "Dgw", gws, "fpl")).toBe(3);
  });

  it("scores a blank as worse than any fixture, never as easy", () => {
    const blanky = windowScore(schedule, "Blanky", gws, "fpl")!;
    expect(blanky).toBeCloseTo((2 + BLANK_VALUE) / 3);
    expect(blanky).toBeGreaterThan(windowScore(schedule, "Easy", gws, "fpl")!);
    expect(BLANK_VALUE).toBeGreaterThan(5);
  });

  it("honours the basis", () => {
    const s = new Map([["A", [fx(1, 2, 4.5)]]]);
    expect(windowScore(s, "A", [1], "fpl")).toBe(2);
    expect(windowScore(s, "A", [1], "custom")).toBe(4.5);
  });

  it("is null when nothing is rated, and unrated is not blank", () => {
    const s = new Map([["A", [fx(1, null)]]]);
    expect(windowScore(s, "A", [1], "fpl")).toBeNull();
    expect(windowScore(new Map([["B", []]]), "B", [1], "fpl")).toBeNull();
    const mixed = new Map([["A", [fx(1, null), fx(2, 3)]]]);
    expect(windowScore(mixed, "A", [1, 2], "fpl")).toBe(3);
  });
});

describe("rankClubs", () => {
  it("orders kindest first with unscorable clubs last", () => {
    const s = new Map(schedule);
    s.set("Nothing", []);
    const order = rankClubs(s, gws, "fpl").map((row) => row.team);
    expect(order).toEqual(["Easy", "Blanky", "Dgw", "Hard", "Nothing"]);
  });
});

describe("gameweekValue", () => {
  it("averages a double and marks a blank", () => {
    expect(gameweekValue(schedule, "Dgw", 2, "fpl")).toBe(3);
    expect(gameweekValue(schedule, "Blanky", 2, "fpl")).toBe(BLANK_VALUE);
  });
});

describe("rotationPairs", () => {
  it("takes the easier club each week and highlights the pick", () => {
    const [best] = rotationPairs(schedule, gws, "fpl", 10);
    // Blanky + Easy: gw1 Blanky 1, gw2 Easy 2, gw3 Blanky 1 -> 4/3.
    expect(best!.a).toBe("Blanky");
    expect(best!.b).toBe("Easy");
    expect(best!.picks.map((p) => p.team)).toEqual(["Blanky", "Easy", "Blanky"]);
    expect(best!.score).toBeCloseTo(4 / 3);
  });

  it("is sorted ascending, limited to top-n, and deterministic", () => {
    const pairs = rotationPairs(schedule, gws, "fpl", 3);
    expect(pairs).toHaveLength(3);
    for (let i = 1; i < pairs.length; i++) {
      expect(pairs[i]!.score).toBeGreaterThanOrEqual(pairs[i - 1]!.score);
    }
    expect(rotationPairs(schedule, gws, "fpl", 3)).toEqual(pairs);
  });

  it("costs a week where one club is blank nothing when the other plays", () => {
    const t = new Map<string, TeamFixture[]>([["A", [fx(2, 2)]], ["B", [fx(2, 3)]]]);
    const [pair] = rotationPairs(t, [1, 2], "fpl");
    // Week 1 both blank: BLANK_VALUE. Week 2: A's 2.
    expect(pair!.picks[0]).toEqual({ gw: 1, team: "A", value: BLANK_VALUE });
    expect(pair!.score).toBeCloseTo((BLANK_VALUE + 2) / 2);
  });

  it("skips pairs with nothing rated", () => {
    const s = new Map<string, TeamFixture[]>([["A", [fx(1, null)]], ["B", [fx(1, null)]]]);
    expect(rotationPairs(s, [1], "fpl")).toEqual([]);
  });
});

describe("gameweekBadges", () => {
  it("lists blank and double clubs per gameweek", () => {
    const [one, two, three] = gameweekBadges(schedule, gws);
    expect(one).toEqual({ gw: 1, blank: [], double: [] });
    expect(two).toEqual({ gw: 2, blank: ["Blanky"], double: ["Dgw"] });
    expect(three!.blank).toEqual([]);
  });
});
