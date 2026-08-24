import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { GoldenSpearmanFile } from "./schema";
import { averageRanks, spearman } from "./spearman";

/**
 * §5.6.1, condition 3: "CI fails if the TS implementation disagrees with
 * any golden value by more than 1e-9."
 *
 * The fixture is self-contained by design — it carries the values its
 * answers were computed over, because this test cannot read
 * `panel.parquet` (§5.3.4 does not commit it) and §5.14.8 requires a
 * fresh clone to work. So this runs the port over exactly the numbers
 * Python ran over, which is the only comparison that means anything.
 */
const GOLDEN = resolve(__dirname, "../../../../data/web/v1/golden_spearman.json");

const file = GoldenSpearmanFile.parse(JSON.parse(readFileSync(GOLDEN, "utf-8")));
const samples = new Map(file.samples.map((sample) => [sample.group, sample]));

describe("the Spearman port against Python's goldens", () => {
  it("ships enough pairs to be worth checking", () => {
    // §5.6.1 asks for at least 50 metric pairs across positions.
    const computable = file.pairs.filter((pair) => pair.rho !== null);
    expect(computable.length).toBeGreaterThanOrEqual(50);
  });

  it("agrees with every golden value within the stated tolerance", () => {
    const failures: string[] = [];

    for (const pair of file.pairs) {
      const sample = samples.get(pair.group)!;
      const ia = sample.metrics.indexOf(pair.a);
      const ib = sample.metrics.indexOf(pair.b);
      const xs = sample.rows.map((row) => row[ia] ?? null);
      const ys = sample.rows.map((row) => row[ib] ?? null);

      const result = spearman(xs, ys);

      if (result.n !== pair.n) {
        failures.push(`${pair.group} ${pair.a}x${pair.b}: n ${result.n} vs ${pair.n}`);
        continue;
      }
      if (pair.rho === null) {
        // Python found no correlation defined; the port must agree that
        // none is defined rather than returning a zero.
        if (result.rho !== null) {
          failures.push(`${pair.group} ${pair.a}x${pair.b}: got ${result.rho}, expected null`);
        }
        continue;
      }
      if (result.rho === null || Math.abs(result.rho - pair.rho) > file.tolerance) {
        failures.push(
          `${pair.group} ${pair.a}x${pair.b}: ${result.rho} vs ${pair.rho}`,
        );
      }
    }

    expect(failures).toEqual([]);
  });

  it("reads its tolerance from the fixture rather than hard-coding one", () => {
    // A second copy of the threshold is a second thing that can drift
    // from the number the export actually promised.
    expect(file.tolerance).toBe(1e-9);
  });
});

describe("tie handling, which is why the port exists", () => {
  it("averages tied ranks", () => {
    // polars' `rank()` default, and what `report.spearman` ranks with. An
    // implementation choosing "min" or "ordinal" would be wrong only
    // where values repeat — which in FPL data is everywhere.
    expect(averageRanks([10, 20, 20, 30])).toEqual([1, 2.5, 2.5, 4]);
    expect(averageRanks([5, 5, 5])).toEqual([2, 2, 2]);
  });

  it("ranks ascending, so a rank of 1 is the smallest value", () => {
    expect(averageRanks([3, 1, 2])).toEqual([3, 1, 2]);
  });
});

describe("the null-pair hazard", () => {
  it("drops incomplete pairs instead of ranking a column of nulls", () => {
    // The failure this project already had in Python: a perfect rank
    // correlation over five complete pairs reported as rho 0.23 over
    // n=10, because the nulls were ignored by the ranking and counted by
    // the n.
    const xs = [1, 2, 3, 4, 5, null, null, null, null, null];
    const ys = [1, 2, 3, 4, 5, 9, 8, 7, 6, 5.5];

    const result = spearman(xs, ys);

    expect(result.n).toBe(5);
    expect(result.rho).toBeCloseTo(1, 12);
  });

  it("returns a null rho, never a zero, when nothing is defined", () => {
    expect(spearman([1, 1, 1], [1, 2, 3]).rho).toBeNull();
    expect(spearman([1], [2]).rho).toBeNull();
    expect(spearman([], []).n).toBe(0);
  });
});
