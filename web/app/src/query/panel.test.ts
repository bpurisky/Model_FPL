import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { ColumnsFile, type ColumnSpec } from "../data/schema";
import {
  facets,
  filterColumns,
  grouped,
  NO_FILTERS,
  QueryError,
  resolveColumn,
  rowMask,
  type PanelFilters,
} from "./panel";
import type { PanelColumn, Session } from "./session";

/**
 * The query layer, over the real registry.
 *
 * Using the committed `columns.json` rather than a fixture is the point:
 * the allowlist *is* the registry, so a test against an invented one
 * would pass while the real thing let something through.
 */
const COLUMNS = resolve(__dirname, "../../../../data/web/v1/columns.json");
const registry = ColumnsFile.parse(JSON.parse(readFileSync(COLUMNS, "utf-8")));
const columns: ReadonlyMap<string, ColumnSpec> = new Map(
  registry.columns.map((column) => [column.key, column]),
);

/**
 * A panel in memory, and a record of which columns were decoded — the
 * cheap proxy for "did this query read more of the file than it needed".
 */
function panel(data: Record<string, PanelColumn>): Session & { read: string[] } {
  const rows = Object.values(data)[0]?.length ?? 0;
  const read: string[] = [];
  return {
    rows,
    names: Object.keys(data),
    read,
    async column(key: string) {
      read.push(key);
      const values = data[key];
      if (!values) throw new Error(`${key} is not a column in panel.parquet`);
      return values;
    },
    async close() {},
  };
}

const SAMPLE = {
  season: ["2024-25", "2024-25", "2025-26", "2025-26", "2025-26"],
  gw: [1, 2, 1, 2, 3],
  element_id: [1, 1, 1, 2, 2],
  name: ["Saka", "Saka", "Saka", "Salah", "Salah"],
  team: ["Arsenal", "Arsenal", "Arsenal", "Liverpool", "Liverpool"],
  position: ["MID", "MID", "MID", "MID", "MID"],
  value: [95, 95, 100, 130, 130],
  cum_minutes: [90, 180, 90, 90, 180],
  total_points: [2, 8, 5, 12, 1],
  xg_per90: [0.4, null, 0.2, 0.8, null],
  xg_per90_z_pos: [1.1, null, 0.5, 2.0, null],
} satisfies Record<string, PanelColumn>;

describe("the allowlist", () => {
  it("refuses a column the registry does not name", async () => {
    const session = panel(SAMPLE);
    await expect(
      grouped(
        session,
        { groupBy: ["team"], collect: ["nonesuch"], filters: NO_FILTERS, normalized: false },
        columns,
      ),
    ).rejects.toThrow(QueryError);
  });

  it("validates every name before decoding anything", async () => {
    const session = panel(SAMPLE);
    await expect(
      grouped(
        session,
        { groupBy: ["team"], collect: ["nonesuch"], filters: NO_FILTERS, normalized: false },
        columns,
      ),
    ).rejects.toThrow(QueryError);
    // A bad key must cost nothing: decoding 85,000 rows and then failing
    // is the same bug with a worse latency profile.
    expect(session.read).toEqual([]);
  });

  it("refuses a query with no group key", async () => {
    await expect(
      grouped(
        panel(SAMPLE),
        { groupBy: [], collect: ["minutes"], filters: NO_FILTERS, normalized: false },
        columns,
      ),
    ).rejects.toThrow(/at least one group key/);
  });
});

describe("filters", () => {
  const filters = (overrides: Partial<PanelFilters>): PanelFilters => ({
    ...NO_FILTERS,
    ...overrides,
  });

  const loaded = new Map<string, PanelColumn>(Object.entries(SAMPLE));

  it("needs no columns when nothing is filtered", () => {
    expect(filterColumns(NO_FILTERS)).toEqual([]);
    expect([...rowMask(NO_FILTERS, loaded, 5)]).toEqual([1, 1, 1, 1, 1]);
  });

  it("names only the columns its predicates actually read", () => {
    expect(filterColumns(filters({ seasons: ["2025-26"], gwMin: 2 }))).toEqual(["season", "gw"]);
  });

  it("combines predicates as AND", () => {
    const mask = rowMask(filters({ seasons: ["2025-26"], gwMin: 2 }), loaded, 5);
    // rows 3 and 4 are 2025-26 with gw >= 2
    expect([...mask]).toEqual([0, 0, 0, 1, 1]);
  });

  it("truncates a fractional bound rather than half-applying it", () => {
    // Price is tenths of a million and the column is an integer.
    const mask = rowMask(filters({ priceMin: 95.7 }), loaded, 5);
    expect([...mask]).toEqual([1, 1, 1, 1, 1]);
  });

  it("rejects a non-finite bound", () => {
    expect(() => rowMask(filters({ gwMin: Number.NaN }), loaded, 5)).toThrow(QueryError);
  });

  it("filters minutes on the season-to-date column, not the gameweek one", () => {
    /*
     * The distinction is the whole meaning of the control. `minutes`
     * would read as "gameweeks in which he played this much" and would
     * drop every rotation week from a regular starter's series, leaving
     * a chart of nothing but his best days.
     */
    expect(filterColumns(filters({ minutesFloor: 100 }))).toEqual(["cum_minutes"]);
    expect([...rowMask(filters({ minutesFloor: 100 }), loaded, 5)]).toEqual([0, 1, 0, 0, 1]);
  });

  it("keeps a row out when the filtered column is null", () => {
    const withNull = new Map(loaded);
    withNull.set("team", ["Arsenal", null, "Arsenal", "Liverpool", "Liverpool"]);
    const mask = rowMask(filters({ teams: ["Arsenal"] }), withNull, 5);
    // A null team is not Arsenal, and it is not "unknown, so include it".
    expect([...mask]).toEqual([1, 0, 1, 0, 0]);
  });
});

describe("grouping", () => {
  it("collects the values behind each group, nulls included", async () => {
    const result = await grouped(
      panel(SAMPLE),
      {
        groupBy: ["element_id", "name"],
        collect: ["xg_per90", "total_points"],
        filters: NO_FILTERS,
        normalized: false,
      },
      columns,
    );

    expect(result.rows).toHaveLength(2);
    const saka = result.rows.find((row) => row.key["name"] === "Saka")!;
    // Nulls survive: `reduce.ts` distinguishes "no rows" from "rows with
    // no value" and both are real answers (§5.3.3).
    expect(saka.values["xg_per90"]).toEqual([0.4, null, 0.2]);
    expect(saka.values["total_points"]).toEqual([2, 8, 5]);
  });

  it("groups on the composite key, not on one column", async () => {
    const result = await grouped(
      panel(SAMPLE),
      {
        groupBy: ["season", "team"],
        collect: ["total_points"],
        filters: NO_FILTERS,
        normalized: false,
      },
      columns,
    );
    expect(result.rows.map((row) => [row.key["season"], row.key["team"]])).toEqual([
      ["2024-25", "Arsenal"],
      ["2025-26", "Arsenal"],
      ["2025-26", "Liverpool"],
    ]);
  });

  it("does not let two key parts run together into one signature", async () => {
    /*
     * The pair ("AB", "C") must not collide with ("A", "BC"). The keys
     * are joined on a unit separator precisely so a concatenation cannot
     * merge two groups into one and silently average them together.
     */
    const session = panel({
      ...SAMPLE,
      team: ["AB", "A", "AB", "A", "A"],
      position: ["C", "BC", "C", "BC", "BC"],
    });
    const result = await grouped(
      session,
      {
        groupBy: ["team", "position"],
        collect: ["total_points"],
        filters: NO_FILTERS,
        normalized: false,
      },
      columns,
    );
    expect(result.rows).toHaveLength(2);
  });

  it("applies the filters before grouping", async () => {
    const result = await grouped(
      panel(SAMPLE),
      {
        groupBy: ["team"],
        collect: ["total_points"],
        filters: { ...NO_FILTERS, seasons: ["2025-26"], gwMin: 2 },
        normalized: false,
      },
      columns,
    );
    expect(result.rows).toHaveLength(1);
    expect(result.rows[0]!.key["team"]).toBe("Liverpool");
    expect(result.rows[0]!.values["total_points"]).toEqual([12, 1]);
  });

  it("returns groups in a stable order regardless of row order", async () => {
    const forward = await grouped(
      panel(SAMPLE),
      { groupBy: ["team"], collect: ["total_points"], filters: NO_FILTERS, normalized: false },
      columns,
    );
    const reversedSample = Object.fromEntries(
      Object.entries(SAMPLE).map(([key, values]) => [key, [...values].reverse()]),
    ) as Record<string, PanelColumn>;
    const backward = await grouped(
      panel(reversedSample),
      { groupBy: ["team"], collect: ["total_points"], filters: NO_FILTERS, normalized: false },
      columns,
    );
    // A chart whose categories reshuffle when a filter changes is a chart
    // nobody trusts.
    expect(forward.rows.map((row) => row.key["team"])).toEqual(
      backward.rows.map((row) => row.key["team"]),
    );
  });

  it("reads each needed column once and no others", async () => {
    const session = panel(SAMPLE);
    await grouped(
      session,
      {
        groupBy: ["team"],
        collect: ["total_points"],
        filters: { ...NO_FILTERS, seasons: ["2025-26"] },
        normalized: false,
      },
      columns,
    );
    expect([...session.read].sort()).toEqual(["season", "team", "total_points"]);
  });
});

describe("the normalization toggle (§5.7.3)", () => {
  it("reads the companion column when one exists", () => {
    expect(resolveColumn("xg_per90", columns, true)).toEqual({
      key: "xg_per90_z_pos",
      normalized: true,
    });
  });

  it("reads raw when the registry offers no companion", () => {
    /*
     * §5.6 forbids the browser standardizing anything, so a column with
     * no exported z-score stays raw however the toggle is set. Inventing
     * one here is precisely the erosion the rule exists to prevent.
     */
    expect(resolveColumn("minutes", columns, true)).toEqual({
      key: "minutes",
      normalized: false,
    });
  });

  it("reads raw for every column when the toggle is off", () => {
    for (const column of registry.columns) {
      expect(resolveColumn(column.key, columns, false).normalized).toBe(false);
    }
  });

  it("reads the companion but reports it under the requested key", async () => {
    const session = panel(SAMPLE);
    const result = await grouped(
      session,
      {
        groupBy: ["team"],
        collect: ["xg_per90"],
        filters: NO_FILTERS,
        normalized: true,
      },
      columns,
    );
    // §5.7.4: a normalized number renders its basis, which means the
    // surface has to know which numbers moved.
    expect(result.normalizedKeys).toEqual(["xg_per90"]);
    expect(session.read).toContain("xg_per90_z_pos");
    // The caller reads its own vocabulary and does not have to know
    // whether the toggle was on.
    expect(result.rows[0]!.values["xg_per90"]).toEqual([1.1, null, 0.5]);
  });
});

describe("the group cap", () => {
  it("reports truncation and returns only the cap", async () => {
    const result = await grouped(
      panel(SAMPLE),
      {
        groupBy: ["element_id"],
        collect: ["total_points"],
        filters: NO_FILTERS,
        normalized: false,
        limit: 1,
      },
      columns,
    );
    expect(result.rows).toHaveLength(1);
    expect(result.truncated).toBe(true);
  });

  it("does not report truncation when everything fits", async () => {
    const result = await grouped(
      panel(SAMPLE),
      {
        groupBy: ["element_id"],
        collect: ["total_points"],
        filters: NO_FILTERS,
        normalized: false,
        limit: 2,
      },
      columns,
    );
    expect(result.rows).toHaveLength(2);
    expect(result.truncated).toBe(false);
  });
});

describe("facets", () => {
  it("reads the filter bar's options from the data", async () => {
    const result = await facets(panel(SAMPLE));
    expect(result.seasons).toEqual(["2024-25", "2025-26"]);
    expect(result.teams).toEqual(["Arsenal", "Liverpool"]);
    expect(result.gwMin).toBe(1);
    expect(result.gwMax).toBe(3);
    expect(result.priceMin).toBe(95);
    expect(result.priceMax).toBe(130);
    expect(result.rows).toBe(5);
  });

  it("orders positions down the pitch rather than alphabetically", async () => {
    const result = await facets(
      panel({ ...SAMPLE, position: ["FWD", "GK", "MID", "DEF", "GK"] }),
    );
    expect(result.positions).toEqual(["GK", "DEF", "MID", "FWD"]);
  });

  it("sorts an unfamiliar position last rather than first", async () => {
    // A new position appearing upstream should be visible, not leading.
    const result = await facets(
      panel({ ...SAMPLE, position: ["AM", "GK", "MID", "DEF", "FWD"] }),
    );
    expect(result.positions).toEqual(["GK", "DEF", "MID", "FWD", "AM"]);
  });
});
