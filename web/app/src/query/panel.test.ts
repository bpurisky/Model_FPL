import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { ColumnsFile, type ColumnSpec } from "../data/schema";
import {
  grouped,
  NO_FILTERS,
  QueryError,
  resolveColumn,
  wherePredicates,
  type PanelFilters,
} from "./panel";
import type { Session } from "./session";

/**
 * The SQL builder, over the real registry.
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

/** Records the SQL it was asked to run and returns nothing. */
function spy(): Session & { sql: string[] } {
  const sql: string[] = [];
  return {
    sql,
    async run(query: string) {
      sql.push(query);
      return [];
    },
    async close() {},
  };
}

describe("the allowlist", () => {
  it("refuses a column the registry does not name", async () => {
    const session = spy();
    await expect(
      grouped(
        session,
        { groupBy: ["team"], collect: ["nonesuch"], filters: NO_FILTERS, normalized: false },
        columns,
      ),
    ).rejects.toThrow(QueryError);
    expect(session.sql).toEqual([]);
  });

  it("refuses an injected identifier before it reaches a query", async () => {
    const session = spy();
    await expect(
      grouped(
        session,
        {
          groupBy: [`team" ; DROP VIEW panel; --`],
          collect: ["minutes"],
          filters: NO_FILTERS,
          normalized: false,
        },
        columns,
      ),
    ).rejects.toThrow(QueryError);
    // Nothing reached the engine, which is the property that matters.
    expect(session.sql).toEqual([]);
  });

  it("refuses a query with no group key", async () => {
    await expect(
      grouped(
        spy(),
        { groupBy: [], collect: ["minutes"], filters: NO_FILTERS, normalized: false },
        columns,
      ),
    ).rejects.toThrow(/at least one group key/);
  });
});

describe("filter predicates", () => {
  const filters = (overrides: Partial<PanelFilters>): PanelFilters => ({
    ...NO_FILTERS,
    ...overrides,
  });

  it("produces nothing when nothing is filtered", () => {
    expect(wherePredicates(NO_FILTERS)).toEqual([]);
  });

  it("escapes a quote in a value rather than closing the literal", () => {
    const [clause] = wherePredicates(filters({ teams: ["Nott'm Forest"] }));
    expect(clause).toBe(`"team" IN ('Nott''m Forest')`);
  });

  it("truncates a fractional bound rather than emitting it", () => {
    // Price is tenths of a million and the column is an integer; a
    // fractional bound would be a type error inside the engine rather
    // than here, where it can be explained.
    const clauses = wherePredicates(filters({ priceMin: 45.7 }));
    expect(clauses).toEqual([`"value" >= 45`]);
  });

  it("rejects a non-finite bound", () => {
    expect(() => wherePredicates(filters({ gwMin: Number.NaN }))).toThrow(QueryError);
  });

  it("filters minutes on the season-to-date column, not the gameweek one", () => {
    /*
     * The distinction is the whole meaning of the control. `minutes`
     * would read as "gameweeks in which he played this much" and would
     * drop every rotation week from a regular starter's series, leaving
     * a chart of nothing but his best days.
     */
    expect(wherePredicates(filters({ minutesFloor: 450 }))).toEqual([`"cum_minutes" >= 450`]);
  });

  it("combines every filter with AND", async () => {
    const session = spy();
    await grouped(
      session,
      {
        groupBy: ["team"],
        collect: ["total_points"],
        filters: filters({
          seasons: ["2025-26"],
          positions: ["MID"],
          priceMin: 40,
          gwMax: 10,
        }),
        normalized: false,
      },
      columns,
    );
    const [sql] = session.sql;
    expect(sql).toContain(`"season" IN ('2025-26')`);
    expect(sql).toContain(`"position" IN ('MID')`);
    expect(sql).toContain(`"value" >= 40`);
    expect(sql).toContain(`"gw" <= 10`);
    expect(sql).toMatch(/WHERE .* AND .* AND .* AND /);
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

  it("reports which columns were normalized so the surface can say so", async () => {
    const session = spy();
    const result = await grouped(
      session,
      {
        groupBy: ["team"],
        collect: ["xg_per90", "minutes"],
        filters: NO_FILTERS,
        normalized: true,
      },
      columns,
    );
    // §5.7.4: a normalized number renders its basis, which means the
    // surface has to know which numbers moved.
    expect(result.normalizedKeys).toEqual(["xg_per90"]);
    expect(session.sql[0]).toContain(`list("xg_per90_z_pos") AS "xg_per90"`);
    expect(session.sql[0]).toContain(`list("minutes") AS "minutes"`);
  });
});

describe("the group cap", () => {
  it("asks for one more than it will draw, so it can tell full from over", async () => {
    const session = spy();
    await grouped(
      session,
      {
        groupBy: ["name"],
        collect: ["total_points"],
        filters: NO_FILTERS,
        normalized: false,
        limit: 10,
      },
      columns,
    );
    expect(session.sql[0]).toContain("LIMIT 11");
  });

  it("reports truncation and returns only the cap", async () => {
    const many: Session = {
      async run() {
        return Array.from({ length: 11 }, (_, index) => ({
          name: `p${index}`,
          total_points: [1, 2],
        }));
      },
      async close() {},
    };
    const result = await grouped(
      many,
      {
        groupBy: ["name"],
        collect: ["total_points"],
        filters: NO_FILTERS,
        normalized: false,
        limit: 10,
      },
      columns,
    );
    expect(result.rows).toHaveLength(10);
    expect(result.truncated).toBe(true);
  });
});

describe("what comes back from Arrow", () => {
  it("preserves nulls in a collected list", async () => {
    // The failure this guards against was real and silent: reading an
    // Arrow vector with `toArray()` ignores the validity bitmap and
    // returns uninitialised memory in the null slots — measured at 309
    // of 616 values on one gameweek of `xg_per90`.
    const session: Session = {
      async run() {
        return [{ team: "ARS", xg_per90: [0.4, null, 0.1, null] }];
      },
      async close() {},
    };
    const result = await grouped(
      session,
      { groupBy: ["team"], collect: ["xg_per90"], filters: NO_FILTERS, normalized: false },
      columns,
    );
    expect(result.rows[0]!.values["xg_per90"]).toEqual([0.4, null, 0.1, null]);
  });

  it("turns BigInt keys and values into plain numbers", async () => {
    const session: Session = {
      async run() {
        return [{ element_id: 233n, total_points: [3n, null, 8n] }];
      },
      async close() {},
    };
    const result = await grouped(
      session,
      {
        groupBy: ["element_id"],
        collect: ["total_points"],
        filters: NO_FILTERS,
        normalized: false,
      },
      columns,
    );
    expect(result.rows[0]!.key["element_id"]).toBe(233);
    expect(result.rows[0]!.values["total_points"]).toEqual([3, null, 8]);
  });

  it("reads an Arrow-style iterable without calling toArray", async () => {
    const vector = {
      *[Symbol.iterator]() {
        yield 1;
        yield null;
        yield 3;
      },
      toArray() {
        // If the implementation ever reaches for this, the test fails
        // loudly rather than silently returning plausible garbage.
        return [1, 999, 3];
      },
    };
    const session: Session = {
      async run() {
        return [{ team: "ARS", minutes: vector }];
      },
      async close() {},
    };
    const result = await grouped(
      session,
      { groupBy: ["team"], collect: ["minutes"], filters: NO_FILTERS, normalized: false },
      columns,
    );
    expect(result.rows[0]!.values["minutes"]).toEqual([1, null, 3]);
  });
});
