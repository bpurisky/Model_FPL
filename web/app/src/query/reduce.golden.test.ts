import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { GoldenReductionsFile } from "../data/schema";
import { n as countPresent, reduce, type Reduction } from "./reduce";

/**
 * §5.11.2: "Golden-value tests for every §5.6.2 reduction against Python
 * over the panel, exact equality for integer reductions and 1e-12 for
 * floats."
 *
 * §5.6.2 lets the browser reduce on the argument that these seven
 * operations "cannot silently disagree with Python because there is
 * nothing to disagree about". That is an argument, not a guarantee, and
 * this is what turns it into one. The fixture carries its own inputs for
 * the same reason the Spearman one does: this test cannot read
 * `panel.parquet` (§5.3.4 does not commit it) and §5.14.8 requires a
 * fresh clone to work.
 */
const GOLDEN = resolve(__dirname, "../../../../data/web/v1/golden_reductions.json");

const file = GoldenReductionsFile.parse(JSON.parse(readFileSync(GOLDEN, "utf-8")));

const byColumn = new Map(
  file.columns.map((name, index) => [name, file.rows.map((row) => row[index] ?? null)]),
);

describe("the §5.6.2 reductions against Python's goldens", () => {
  it("covers all seven names", () => {
    const seen = new Set(file.cases.map((testCase) => testCase.fn));
    expect([...seen].sort()).toEqual([
      "count",
      "max",
      "mean",
      "median",
      "min",
      "quantile",
      "sum",
    ]);
  });

  it("covers a column that is entirely absent", () => {
    /*
     * The empty-set rule is the one this fixture exists to pin down, and
     * a sample that happened to populate every column would leave it
     * untested while still looking green. `cbi_per90` does not exist
     * before 2025-26 and the sample is drawn from 2023-24 and 2024-25,
     * so the absence is real rather than constructed.
     */
    const empty = file.cases.filter((testCase) => testCase.n === 0);
    expect(empty.length).toBeGreaterThan(0);
    for (const testCase of empty) {
      const expected = testCase.fn === "count" ? 0 : null;
      expect(testCase.value).toBe(expected);
    }
  });

  it("agrees with every golden value within the stated tolerance", () => {
    const failures: string[] = [];

    for (const testCase of file.cases) {
      const values = byColumn.get(testCase.column);
      if (!values) {
        failures.push(`${testCase.column}: not in the embedded sample`);
        continue;
      }

      const got = reduce(values, testCase.fn as Reduction, testCase.q ?? 0.5);
      const label = `${testCase.column} ${testCase.fn}${testCase.q === null ? "" : `(${testCase.q})`}`;

      if (countPresent(values) !== testCase.n) {
        failures.push(`${label}: n ${countPresent(values)} vs ${testCase.n}`);
        continue;
      }

      if (testCase.value === null) {
        // Python found nothing to reduce; the port must agree that there
        // is nothing rather than returning a zero (§5.3.3).
        if (got !== null) failures.push(`${label}: got ${got}, expected null`);
        continue;
      }

      if (got === null) {
        failures.push(`${label}: got null, expected ${testCase.value}`);
        continue;
      }

      // `count` is a count and is compared exactly; §5.11.2 asks for
      // exact equality on integer reductions.
      const ok =
        testCase.fn === "count"
          ? got === testCase.value
          : Math.abs(got - testCase.value) <= file.tolerance;

      if (!ok) failures.push(`${label}: ${got} vs ${testCase.value}`);
    }

    expect(failures).toEqual([]);
  });
});

describe("properties the reductions have to hold regardless of the fixture", () => {
  const sample = byColumn.get("total_points")!;

  it("is order-independent", () => {
    // The claim §5.6.2 rests on. Reversing and shuffling the input must
    // not move any answer, including `sum` — which is only true because
    // `present()` imposes an order before adding.
    const reversed = [...sample].reverse();
    const shuffled = [...sample].sort(() => Math.random() - 0.5);
    for (const fn of ["count", "sum", "mean", "median", "min", "max"] as const) {
      expect(reduce(reversed, fn)).toBe(reduce(sample, fn));
      expect(reduce(shuffled, fn)).toBe(reduce(sample, fn));
    }
  });

  it("drops nulls rather than reading them as zero", () => {
    const withNulls = [...sample, null, null, null];
    for (const fn of ["sum", "mean", "median", "min", "max"] as const) {
      expect(reduce(withNulls, fn)).toBe(reduce(sample, fn));
    }
    // count is the one that must notice, and must not count them.
    expect(reduce(withNulls, "count")).toBe(reduce(sample, "count"));
  });

  it("puts median and quantile(0.5) on the same code path", () => {
    for (const [name, values] of byColumn) {
      expect(reduce(values, "median"), name).toBe(reduce(values, "quantile", 0.5));
    }
  });

  it("brackets every quantile between min and max", () => {
    for (const [name, values] of byColumn) {
      const lo = reduce(values, "min");
      const hi = reduce(values, "max");
      if (lo === null || hi === null) continue;
      for (const q of [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]) {
        const value = reduce(values, "quantile", q)!;
        expect(value, `${name} q=${q}`).toBeGreaterThanOrEqual(lo);
        expect(value, `${name} q=${q}`).toBeLessThanOrEqual(hi);
      }
    }
  });

  it("returns the extremes at q=0 and q=1", () => {
    for (const [name, values] of byColumn) {
      expect(reduce(values, "quantile", 0), name).toBe(reduce(values, "min"));
      expect(reduce(values, "quantile", 1), name).toBe(reduce(values, "max"));
    }
  });
});
