/**
 * `projections.json`: the pure logic behind the Projections table.
 *
 * Kept out of the view so the three things a reader could be misled by are
 * testable without rendering anything:
 *
 * **A window total is a sum over the first N gameweeks, and nothing else.**
 * `xp[i]` belongs to `gameweeks[i]`, a blank gameweek is already 0 and a
 * double already sums both fixtures (the exporter does that, see
 * `web/export/projections.py`), so the total here adds numbers and does not
 * re-derive any of them.
 *
 * **A player with no number sorts last, in both directions.** A null is
 * "the model did not say", not "the lowest value there is".
 *
 * **Filters narrow the table and never change a number.**
 */

import type { ProjectionPlayer, ProjectionsFile } from "./schema";

/** The widest window the file can carry, and the narrowest the table offers. */
export const MIN_WINDOW = 1;

export type SortKey =
  | { kind: "name" }
  | { kind: "team" }
  | { kind: "position" }
  | { kind: "price" }
  | { kind: "total" }
  | { kind: "gw"; index: number };

export interface Sort {
  key: SortKey;
  desc: boolean;
}

export interface Filters {
  /** "all", or one of GK / DEF / MID / FWD. */
  position: string;
  /** Tenths of a million, as `price` is carried; null for no ceiling. */
  maxPrice: number | null;
  /** Case-insensitive substring of the name or the club. */
  search: string;
}

export const NO_FILTERS: Filters = { position: "all", maxPrice: null, search: "" };

/** Clamp a requested window to what the file actually has. */
export function clampWindow(requested: number, available: number): number {
  return Math.max(Math.min(MIN_WINDOW, available), Math.min(requested, available));
}

/** Sum of the first `window` gameweeks; null when none of them has a number. */
export function windowTotal(xp: (number | null)[], window: number): number | null {
  const slice = xp.slice(0, Math.max(window, 0));
  let total = 0;
  let seen = false;
  for (const value of slice) {
    if (value === null) continue;
    total += value;
    seen = true;
  }
  return seen ? total : null;
}

export function matches(player: ProjectionPlayer, filters: Filters): boolean {
  if (filters.position !== "all" && player.position !== filters.position) return false;
  if (filters.maxPrice !== null && (player.price === null || player.price > filters.maxPrice)) {
    return false;
  }
  const needle = filters.search.trim().toLowerCase();
  if (needle === "") return true;
  return (
    player.name.toLowerCase().includes(needle) || player.team.toLowerCase().includes(needle)
  );
}

const POSITION_ORDER: Record<string, number> = { GK: 0, DEF: 1, MID: 2, FWD: 3 };

function valueOf(player: ProjectionPlayer, key: SortKey, window: number): number | string | null {
  switch (key.kind) {
    case "name":
      return player.name;
    case "team":
      return player.team;
    case "position":
      return POSITION_ORDER[player.position] ?? Number.MAX_SAFE_INTEGER;
    case "price":
      return player.price;
    case "total":
      return windowTotal(player.xp, window);
    case "gw":
      return player.xp[key.index] ?? null;
  }
}

/** Filtered then sorted, without mutating the file's own array. */
export function selectRows(
  players: ProjectionPlayer[],
  filters: Filters,
  sort: Sort,
  window: number,
): ProjectionPlayer[] {
  return players
    .filter((player) => matches(player, filters))
    .map((player) => ({ player, value: valueOf(player, sort.key, window) }))
    .sort((left, right) => {
      if (left.value === null && right.value === null) return left.player.name.localeCompare(right.player.name);
      if (left.value === null) return 1;
      if (right.value === null) return -1;
      const order =
        typeof left.value === "string" || typeof right.value === "string"
          ? String(left.value).localeCompare(String(right.value))
          : left.value - right.value;
      if (order !== 0) return sort.desc ? -order : order;
      // Ties fall back to name so a re-sort never shuffles equal rows.
      return left.player.name.localeCompare(right.player.name);
    })
    .map((row) => row.player);
}

/** One club's fixtures in one gameweek, by the club's own name. */
export function fixturesFor(file: ProjectionsFile) {
  const lookup = new Map<string, Map<number, ProjectionsFile["teams"][number]["gameweeks"][number]["fixtures"]>>();
  for (const team of file.teams) {
    lookup.set(team.team, new Map(team.gameweeks.map((g) => [g.gw, g.fixtures])));
  }
  return (team: string, gw: number) => lookup.get(team)?.get(gw) ?? [];
}
