import { describe, expect, it } from "vitest";
import type { OwnershipPlayer } from "../data/schema";
import {
  DEFAULT_SORT,
  filterOwnership,
  fmtDelta,
  fmtPct,
  NO_OWNERSHIP_FILTERS,
  nextSort,
  sortOwnership,
} from "./EliteOwnership";

const player = (over: Partial<OwnershipPlayer>): OwnershipPlayer => ({
  element_id: 1,
  name: "Salah",
  team: "Liverpool",
  position: "MID",
  now_cost: 130,
  overall_pct: 40,
  top_pct: 80,
  top_start_pct: 75,
  captain_pct: 30,
  eo_pct: 110,
  delta_pct: 40,
  ...over,
});

const players = [
  player({ element_id: 1, name: "Salah", eo_pct: 110, delta_pct: 40 }),
  player({ element_id: 2, name: "Haaland", team: "Man City", position: "FWD", eo_pct: 150, delta_pct: -5 }),
  player({ element_id: 3, name: "Gabriel", team: "Arsenal", position: "DEF", eo_pct: 20, overall_pct: null, delta_pct: null }),
];

describe("filterOwnership", () => {
  it("filters by position and by name or club, case-insensitively", () => {
    expect(filterOwnership(players, { position: "FWD", query: "" }).map((p) => p.name)).toEqual(["Haaland"]);
    expect(filterOwnership(players, { position: "all", query: " ARSEN " }).map((p) => p.name)).toEqual(["Gabriel"]);
    expect(filterOwnership(players, NO_OWNERSHIP_FILTERS)).toHaveLength(3);
  });
});

describe("sortOwnership", () => {
  it("defaults to EO descending", () => {
    expect(sortOwnership(players, DEFAULT_SORT).map((p) => p.name)).toEqual(["Haaland", "Salah", "Gabriel"]);
  });

  it("sorts text ascending and does not mutate the input", () => {
    const before = players.map((p) => p.name);
    expect(sortOwnership(players, { key: "name", dir: "asc" }).map((p) => p.name)).toEqual(["Gabriel", "Haaland", "Salah"]);
    expect(players.map((p) => p.name)).toEqual(before);
  });

  it("puts null values last in either direction", () => {
    expect(sortOwnership(players, { key: "delta_pct", dir: "desc" }).map((p) => p.name)).toEqual(["Salah", "Haaland", "Gabriel"]);
    expect(sortOwnership(players, { key: "delta_pct", dir: "asc" }).map((p) => p.name)).toEqual(["Haaland", "Salah", "Gabriel"]);
  });
});

describe("nextSort", () => {
  it("flips the active column and starts text ascending, numbers descending", () => {
    expect(nextSort({ key: "eo_pct", dir: "desc" }, "eo_pct")).toEqual({ key: "eo_pct", dir: "asc" });
    expect(nextSort(DEFAULT_SORT, "name")).toEqual({ key: "name", dir: "asc" });
    expect(nextSort(DEFAULT_SORT, "top_pct")).toEqual({ key: "top_pct", dir: "desc" });
  });
});

describe("formatting", () => {
  it("renders null as n/a and signs the delta", () => {
    expect(fmtPct(null)).toBe("n/a");
    expect(fmtPct(12.345)).toBe("12.3%");
    expect(fmtDelta(null)).toBe("n/a");
    expect(fmtDelta(4.04)).toBe("+4.0");
    expect(fmtDelta(-4.04)).toBe("−4.0");
  });
});
