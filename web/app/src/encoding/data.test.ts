import { describe, expect, it } from "vitest";
import type { GroupedRow } from "../query/panel";
import { bin, buildPlot, collectKeysFor, groupKeysFor } from "./data";
import { inferMark } from "./mark";
import type { Encoding, Roles } from "./spec";

/**
 * The shaping between the query and the marks.
 *
 * The properties worth testing here are the quiet ones: that a group
 * which reduced to nothing is dropped rather than drawn at zero
 * (§5.3.3, §5.14.9), that series and facets come back in a stable order,
 * and that a scatter is one point per player rather than one per
 * player-gameweek.
 */

const encoding = (over: Partial<Encoding> = {}): Encoding => ({
  x: null,
  y: null,
  color: null,
  wrap: null,
  aggregate: "mean",
  ...over,
});

const roles = (over: Partial<Roles> = {}): Roles => ({
  x: null,
  y: null,
  color: null,
  wrap: null,
  ...over,
});

function planFor(r: Roles) {
  const result = inferMark(r);
  if (!result.ok) throw new Error(result.reason);
  return result.plan;
}

function row(
  key: GroupedRow["key"],
  values: GroupedRow["values"] = {},
): GroupedRow {
  return { key, values };
}

describe("group and collect keys", () => {
  it("expands the player grain into the four columns that identify one", () => {
    const plan = planFor(roles({ x: "quantitative", y: "quantitative" }));
    const keys = groupKeysFor(plan, encoding({ x: "xg_per90", y: "total_points" }));
    // Grouping on element_id alone would hand back an integer with no
    // name to put in a tooltip and no position to colour by.
    expect(keys).toEqual(["element_id", "name", "team", "position"]);
  });

  it("adds the wrap column to the group key", () => {
    const plan = planFor(roles({ x: "categorical", y: "quantitative", wrap: "categorical" }));
    const keys = groupKeysFor(plan, encoding({ x: "team", y: "total_points", wrap: "position" }));
    expect(keys).toEqual(["team", "position"]);
  });

  it("never repeats a column that is both grouped and wrapped", () => {
    // SQL will not accept the same column twice in a GROUP BY list.
    const plan = planFor(roles({ x: "categorical", y: "quantitative", wrap: "categorical" }));
    const keys = groupKeysFor(plan, encoding({ x: "team", y: "total_points", wrap: "team" }));
    expect(keys).toEqual(["team"]);
  });

  it("collects only the channels the plan reduces", () => {
    const plan = planFor(roles({ x: "ordinal", y: "quantitative", color: "categorical" }));
    const keys = collectKeysFor(plan, encoding({ x: "gw", y: "total_points", color: "team" }));
    // The colour is a group key here, not something to reduce.
    expect(keys).toEqual(["total_points"]);
  });
});

describe("nulls are dropped, never drawn at zero", () => {
  const plan = planFor(roles({ x: "quantitative", y: "quantitative" }));
  const enc = encoding({ x: "xg_per90", y: "total_points" });

  it("omits a point whose reduction came back null, and counts it", () => {
    const rows = [
      row(
        { element_id: 1, name: "Real", team: "ARS", position: "MID" },
        { xg_per90: [0.4, 0.2], total_points: [4, 6] },
      ),
      row(
        // Every value below the minutes floor: the export wrote nulls,
        // and "unknown" is not "zero".
        { element_id: 2, name: "Absent", team: "ARS", position: "MID" },
        { xg_per90: [null, null], total_points: [1, 2] },
      ),
    ];

    const plot = buildPlot(rows, plan, enc);
    expect(plot.count).toBe(1);
    expect(plot.dropped).toBe(1);
    expect(plot.facets[0]!.series[0]!.points[0]!.label).toBe("Real (ARS)");
  });

  it("keeps a genuine zero, which is a different fact", () => {
    const rows = [
      row(
        { element_id: 1, name: "Blank", team: "ARS", position: "MID" },
        { xg_per90: [0, 0], total_points: [0, 0] },
      ),
    ];
    const plot = buildPlot(rows, plan, enc);
    expect(plot.dropped).toBe(0);
    expect(plot.facets[0]!.series[0]!.points[0]!.x).toBe(0);
  });

  it("reports n as the rows behind the reduction, not the group size", () => {
    const rows = [
      row(
        { element_id: 1, name: "Rotated", team: "ARS", position: "MID" },
        { xg_per90: [0.4, null, 0.2, null], total_points: [4, 0, 6, 0] },
      ),
    ];
    const plot = buildPlot(rows, plan, enc);
    // Four gameweeks in the group; two carry an xG.
    expect(plot.facets[0]!.series[0]!.points[0]!.n).toBe(4);
  });
});

describe("series and facets", () => {
  it("splits a line into one series per colour category", () => {
    const plan = planFor(roles({ x: "ordinal", y: "quantitative", color: "categorical" }));
    const enc = encoding({ x: "gw", y: "total_points", color: "position" });
    const rows = [
      row({ gw: 1, position: "MID" }, { total_points: [4] }),
      row({ gw: 2, position: "MID" }, { total_points: [6] }),
      row({ gw: 1, position: "DEF" }, { total_points: [2] }),
    ];
    const plot = buildPlot(rows, plan, enc);
    expect(plot.facets).toHaveLength(1);
    expect(plot.facets[0]!.series.map((series) => series.name)).toEqual(["DEF", "MID"]);
  });

  it("orders facets and series deterministically", () => {
    const plan = planFor(roles({ x: "categorical", y: "quantitative", wrap: "categorical" }));
    const enc = encoding({ x: "team", y: "total_points", wrap: "position" });
    const rows = [
      row({ team: "LIV", position: "MID" }, { total_points: [5] }),
      row({ team: "ARS", position: "DEF" }, { total_points: [3] }),
      row({ team: "ARS", position: "MID" }, { total_points: [4] }),
    ];
    // A legend that reshuffles when a filter changes is a legend nobody
    // trusts, so the order is imposed rather than inherited from the
    // engine's row order.
    const first = buildPlot(rows, plan, enc);
    const second = buildPlot([...rows].reverse(), plan, enc);
    expect(first.facets.map((facet) => facet.name)).toEqual(["DEF", "MID"]);
    expect(second.facets.map((facet) => facet.name)).toEqual(["DEF", "MID"]);
  });

  it("sorts a numeric x domain numerically, so gameweek 10 follows 9", () => {
    const plan = planFor(roles({ x: "ordinal", y: "quantitative" }));
    const enc = encoding({ x: "gw", y: "total_points" });
    const rows = [2, 10, 1, 9].map((gw) => row({ gw }, { total_points: [gw] }));
    const plot = buildPlot(rows, plan, enc);
    expect(plot.xDomain).toEqual([1, 2, 9, 10]);
  });
});

describe("the aggregate applies to every reduced channel", () => {
  it("uses the chosen reduction on both axes of a scatter", () => {
    const plan = planFor(roles({ x: "quantitative", y: "quantitative" }));
    const rows = [
      row(
        { element_id: 1, name: "P", team: "ARS", position: "MID" },
        { xg_per90: [0, 10], total_points: [0, 10] },
      ),
    ];

    const mean = buildPlot(rows, plan, encoding({ x: "xg_per90", y: "total_points" }));
    expect(mean.facets[0]!.series[0]!.points[0]).toMatchObject({ x: 5, y: 5 });

    const max = buildPlot(
      rows,
      plan,
      encoding({ x: "xg_per90", y: "total_points", aggregate: "max" }),
    );
    expect(max.facets[0]!.series[0]!.points[0]).toMatchObject({ x: 10, y: 10 });
  });
});

describe("histogram bins", () => {
  it("returns nothing for no values", () => {
    expect(bin([])).toEqual([]);
  });

  it("puts every value in exactly one bin", () => {
    const values = Array.from({ length: 200 }, (_, index) => index / 7);
    const bins = bin(values);
    expect(bins.reduce((total, entry) => total + entry.count, 0)).toBe(values.length);
  });

  it("puts the top edge in the last bin rather than in one that does not exist", () => {
    const bins = bin([0, 1, 2, 3, 4], 4);
    expect(bins).toHaveLength(4);
    expect(bins[3]!.count).toBeGreaterThan(0);
    expect(bins.reduce((total, entry) => total + entry.count, 0)).toBe(5);
  });

  it("collapses to one bin when every value is identical", () => {
    // A zero-width extent would otherwise divide by zero and produce
    // NaN edges that render as an empty chart.
    expect(bin([3, 3, 3])).toEqual([{ lo: 3, hi: 3, count: 3 }]);
  });
});
