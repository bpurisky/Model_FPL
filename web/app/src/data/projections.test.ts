import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  NO_FILTERS,
  clampWindow,
  fixturesFor,
  matches,
  selectRows,
  windowTotal,
} from "./projections";
import { ProjectionsFile, type ProjectionPlayer } from "./schema";

function player(over: Partial<ProjectionPlayer>): ProjectionPlayer {
  return {
    element_id: 1,
    name: "A",
    team: "Team A",
    position: "MID",
    price: 60,
    xp: [1, 2, 3, 4, 5, 6],
    p_blank: 0,
    p_short: 0,
    p_full: 1,
    ...over,
  };
}

describe("windowTotal", () => {
  it("sums the first N gameweeks only", () => {
    const xp = [1, 2, 3, 4, 5, 6];
    expect(windowTotal(xp, 1)).toBe(1);
    expect(windowTotal(xp, 3)).toBe(6);
    expect(windowTotal(xp, 6)).toBe(21);
  });

  it("counts a blank gameweek as the 0 it is and still totals the rest", () => {
    expect(windowTotal([2, 0, 3], 3)).toBe(5);
  });

  it("skips a missing number rather than reading it as zero, and is null when all are missing", () => {
    expect(windowTotal([2, null, 3], 3)).toBe(5);
    expect(windowTotal([null, null], 2)).toBeNull();
  });

  it("does not run past a short horizon", () => {
    expect(windowTotal([1, 2], 6)).toBe(3);
  });
});

describe("clampWindow", () => {
  it("stays inside what the file carries", () => {
    expect(clampWindow(6, 2)).toBe(2);
    expect(clampWindow(0, 6)).toBe(1);
    expect(clampWindow(4, 6)).toBe(4);
    expect(clampWindow(3, 0)).toBe(0);
  });
});

describe("selectRows", () => {
  const rows = [
    player({ element_id: 1, name: "Low", xp: [1, 1, 1], price: 45, position: "DEF" }),
    player({ element_id: 2, name: "High", xp: [5, 0, 5], price: 130, position: "FWD" }),
    player({ element_id: 3, name: "Mid", xp: [3, 4, 0], price: 70, position: "MID" }),
    player({ element_id: 4, name: "None", xp: [null, null, null], price: null, position: "MID" }),
  ];
  const names = (sorted: ProjectionPlayer[]) => sorted.map((p) => p.name);

  it("sorts by the window total, and the window changes the order", () => {
    const total = { key: { kind: "total" as const }, desc: true };
    expect(names(selectRows(rows, NO_FILTERS, total, 3))).toEqual(["High", "Mid", "Low", "None"]);
    // Over the first gameweek alone Mid (3) is ahead of Low (1) but behind High (5).
    expect(names(selectRows(rows, NO_FILTERS, total, 1))).toEqual(["High", "Mid", "Low", "None"]);
    // Over two gameweeks Mid (7) overtakes High (5).
    expect(names(selectRows(rows, NO_FILTERS, total, 2))).toEqual(["Mid", "High", "Low", "None"]);
  });

  it("sorts a single gameweek column", () => {
    const gw2 = { key: { kind: "gw" as const, index: 1 }, desc: true };
    expect(names(selectRows(rows, NO_FILTERS, gw2, 3))).toEqual(["Mid", "Low", "High", "None"]);
  });

  it("puts a player with no number last in both directions", () => {
    const up = { key: { kind: "total" as const }, desc: false };
    expect(names(selectRows(rows, NO_FILTERS, up, 3)).at(-1)).toBe("None");
    const price = { key: { kind: "price" as const }, desc: true };
    expect(names(selectRows(rows, NO_FILTERS, price, 3)).at(-1)).toBe("None");
  });

  it("sorts text ascending and positions in pitch order", () => {
    expect(names(selectRows(rows, NO_FILTERS, { key: { kind: "name" }, desc: false }, 3))).toEqual([
      "High",
      "Low",
      "Mid",
      "None",
    ]);
    expect(
      names(selectRows(rows, NO_FILTERS, { key: { kind: "position" }, desc: false }, 3))[0],
    ).toBe("Low");
  });

  it("does not mutate the input", () => {
    const before = names(rows);
    selectRows(rows, NO_FILTERS, { key: { kind: "name" }, desc: true }, 3);
    expect(names(rows)).toEqual(before);
  });

  it("filters by position, ceiling and search without touching a number", () => {
    const sort = { key: { kind: "total" as const }, desc: true };
    expect(names(selectRows(rows, { ...NO_FILTERS, position: "MID" }, sort, 3))).toEqual(["Mid", "None"]);
    expect(names(selectRows(rows, { ...NO_FILTERS, maxPrice: 70 }, sort, 3))).toEqual(["Mid", "Low"]);
    expect(names(selectRows(rows, { ...NO_FILTERS, search: "  hIgH " }, sort, 3))).toEqual(["High"]);
  });
});

describe("matches", () => {
  it("excludes an unpriced player once a ceiling is set", () => {
    expect(matches(player({ price: null }), { ...NO_FILTERS, maxPrice: 100 })).toBe(false);
    expect(matches(player({ price: null }), NO_FILTERS)).toBe(true);
  });

  it("searches the club as well as the name", () => {
    expect(matches(player({ team: "Man City" }), { ...NO_FILTERS, search: "city" })).toBe(true);
  });
});

describe("the committed file", () => {
  const path = resolve(__dirname, "../../../../data/web/v1/projections.json");
  const file = ProjectionsFile.parse(JSON.parse(readFileSync(path, "utf-8")));

  it("has one number per player per gameweek, and a fixture entry per club per gameweek", () => {
    for (const p of file.players) expect(p.xp).toHaveLength(file.gameweeks.length);
    for (const t of file.teams) expect(t.gameweeks.map((g) => g.gw)).toEqual(file.gameweeks);
  });

  it("looks a club's fixtures up by name and returns none for an unknown club", () => {
    const fixtures = fixturesFor(file);
    const team = file.teams[0]!;
    expect(fixtures(team.team, file.gameweeks[0]!)).toEqual(team.gameweeks[0]!.fixtures);
    expect(fixtures("No Such FC", file.gameweeks[0]!)).toEqual([]);
  });
});
