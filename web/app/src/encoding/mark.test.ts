/**
 * §5.11.3: "The mark-inference table (§5.4.2) is exhaustively unit-tested
 * — every role combination, including the ones that produce no valid
 * mark."
 *
 * So this walks the full cross product rather than sampling it: five
 * states per channel (four roles plus empty) across x, y, colour and
 * wrap is 625 combinations, every one of which is asserted to resolve
 * either to a named mark or to a reason. The point of the exhaustive
 * form is not the eight rows the table names — those are checked
 * individually below — it is the 600-odd it does not, where a future
 * edit could quietly start guessing.
 */

import { describe, expect, it } from "vitest";
import { inferMark, type Inference } from "./mark";
import type { ChannelRole, Roles } from "./spec";

const STATES: ChannelRole[] = [null, "quantitative", "categorical", "ordinal", "temporal"];

function roles(
  x: ChannelRole,
  y: ChannelRole = null,
  color: ChannelRole = null,
  wrap: ChannelRole = null,
): Roles {
  return { x, y, color, wrap };
}

function expectMark(result: Inference, mark: string) {
  if (!result.ok) throw new Error(`expected ${mark}, got refusal: ${result.reason}`);
  expect(result.plan.mark).toBe(mark);
  return result.plan;
}

describe("the §5.4.2 table, row by row", () => {
  it("quantitative x quantitative -> point, for every colour", () => {
    for (const color of STATES) {
      const plan = expectMark(inferMark(roles("quantitative", "quantitative", color)), "point");
      // The panel is player-gameweek grain; a scatter of two rates is
      // only meaningful once it is one row per player.
      expect(plan.groupBy).toEqual(["player"]);
      expect(plan.series).toBe(color === "categorical");
    }
  });

  it("quantitative alone -> histogram", () => {
    const plan = expectMark(inferMark(roles("quantitative")), "histogram");
    expect(plan.reduced).toEqual(["x"]);
  });

  it("categorical x quantitative -> aggregated bar", () => {
    const plan = expectMark(inferMark(roles("categorical", "quantitative")), "bar");
    expect(plan.groupBy).toEqual(["x"]);
    expect(plan.reduced).toEqual(["y"]);
  });

  it("ordinal x quantitative -> line", () => {
    const plan = expectMark(inferMark(roles("ordinal", "quantitative")), "line");
    expect(plan.series).toBe(false);
  });

  it("ordinal x quantitative x categorical colour -> one line per series", () => {
    const plan = expectMark(
      inferMark(roles("ordinal", "quantitative", "categorical")),
      "line",
    );
    expect(plan.series).toBe(true);
    expect(plan.groupBy).toEqual(["x", "color"]);
  });

  it("categorical x categorical x quantitative colour -> rect", () => {
    const plan = expectMark(
      inferMark(roles("categorical", "categorical", "quantitative")),
      "rect",
    );
    expect(plan.groupBy).toEqual(["x", "y"]);
    expect(plan.reduced).toEqual(["color"]);
  });

  it("ordinal x categorical x quantitative colour -> rect", () => {
    expectMark(inferMark(roles("ordinal", "categorical", "quantitative")), "rect");
  });

  it("temporal x quantitative -> line", () => {
    expectMark(inferMark(roles("temporal", "quantitative")), "line");
  });
});

describe("combinations that produce no mark", () => {
  it("refuses an empty X before anything else", () => {
    for (const y of STATES) {
      for (const color of STATES) {
        const result = inferMark(roles(null, y, color));
        expect(result.ok).toBe(false);
        if (!result.ok) expect(result.reason).toMatch(/Assign a column to X/);
      }
    }
  });

  it("refuses a quantitative wrap, whatever else is assigned", () => {
    for (const x of STATES) {
      for (const y of STATES) {
        const result = inferMark(roles(x, y, null, "quantitative"));
        expect(result.ok).toBe(false);
        if (!result.ok) expect(result.reason).toMatch(/finite set of values/);
      }
    }
  });

  it("refuses a coloured histogram rather than ignoring the colour", () => {
    // D7: a drop zone that accepts a column and does nothing with it is
    // worse than one that says why it cannot.
    for (const color of STATES.filter((role) => role !== null)) {
      const result = inferMark(roles("quantitative", null, color));
      expect(result.ok).toBe(false);
    }
  });

  it("refuses a coloured bar rather than inventing a stack", () => {
    for (const color of STATES.filter((role) => role !== null)) {
      const result = inferMark(roles("categorical", "quantitative", color));
      expect(result.ok).toBe(false);
      if (!result.ok) expect(result.reason).toMatch(/bar already encodes its value/);
    }
  });

  it("refuses a quantitative colour on a line (D6)", () => {
    for (const x of ["ordinal", "temporal"] as const) {
      const result = inferMark(roles(x, "quantitative", "quantitative"));
      expect(result.ok).toBe(false);
      if (!result.ok) expect(result.reason).toMatch(/one series per category/);
    }
  });

  it("refuses a two-category grid with nothing to fill the cells", () => {
    for (const color of [null, "categorical", "ordinal", "temporal"] as const) {
      const result = inferMark(roles("categorical", "categorical", color));
      expect(result.ok).toBe(false);
      if (!result.ok) expect(result.reason).toMatch(/needs a number in Colour/);
    }
  });

  it("tells the user to swap axes when an ordinal lands on Y", () => {
    const result = inferMark(roles("quantitative", "ordinal"));
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toMatch(/belong on X/);
  });
});

describe("the whole cross product", () => {
  it("resolves every one of the 625 combinations to a mark or a reason", () => {
    let drawn = 0;
    let refused = 0;

    for (const x of STATES) {
      for (const y of STATES) {
        for (const color of STATES) {
          for (const wrap of STATES) {
            const result = inferMark(roles(x, y, color, wrap));
            if (result.ok) {
              drawn += 1;
              expect(result.plan.mark).toBeTruthy();
              expect(result.plan.groupBy.length).toBeGreaterThan(0);
            } else {
              refused += 1;
              // Every refusal is actionable prose, not a code.
              expect(result.reason.length).toBeGreaterThan(20);
              expect(result.reason).toMatch(/[.]$/);
            }
          }
        }
      }
    }

    expect(drawn + refused).toBe(STATES.length ** 4);
    /*
     * Recorded rather than asserted loosely: if an edit widens what the
     * builder will draw, this number moves and the diff has to say why.
     *
     * 39 = the thirteen role triples the table names, times the three
     * wrap states that can facet (empty, categorical, ordinal). The
     * thirteen are: quantitative x quantitative under each of the five
     * colour states (5), the lone histogram (1), the bar (1), the two
     * line rows for ordinal (2) and their temporal twins (2), and the
     * two rect rows (2).
     */
    expect(drawn).toBe(39);
    expect(refused).toBe(586);
  });

  it("never depends on wrap for which mark is drawn", () => {
    for (const x of STATES) {
      for (const y of STATES) {
        for (const color of STATES) {
          const base = inferMark(roles(x, y, color, null));
          for (const wrap of ["categorical", "ordinal"] as const) {
            const wrapped = inferMark(roles(x, y, color, wrap));
            expect(wrapped.ok).toBe(base.ok);
            if (base.ok && wrapped.ok) {
              expect(wrapped.plan.mark).toBe(base.plan.mark);
            }
          }
        }
      }
    }
  });
});
