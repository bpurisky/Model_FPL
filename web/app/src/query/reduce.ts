/**
 * §5.6.2's exact descriptive reductions — the second and last exception
 * to "the browser does not infer".
 *
 * > `count`, `sum`, `mean`, `median`, `min`, `max`, `quantile`
 * >
 * > "These are exact, deterministic, order-independent reductions with no
 * > estimator choice, no tie-handling convention, and no distributional
 * > assumption. They cannot silently disagree with Python because there
 * > is nothing to disagree about."
 *
 * That last sentence is only true if there is **one** implementation and
 * a golden test holds it against Python. Hence the shape of this layer:
 * DuckDB reads the parquet, pushes down the filters and does the
 * grouping — the work an engine is for — and hands back one array per
 * group. The reduction itself happens here, in plain TypeScript, so that
 * `reduce.golden.test.ts` can check it against
 * `data/web/v1/golden_reductions.json` with no engine in the loop, the
 * same way `spearman.golden.test.ts` checks the §5.6.1 port.
 *
 * Writing the reductions as SQL instead would have been shorter and would
 * have left the numbers on screen covered by no test that runs in CI.
 *
 * Nulls are dropped, never coerced (§5.3.3). A null is "this player is
 * below the minutes floor", and averaging it in as zero would drag every
 * position mean toward a value no player has.
 */

/** The closed set. Nothing may be added without amending §5.6.2. */
export type Reduction = "count" | "sum" | "mean" | "median" | "min" | "max" | "quantile";

/**
 * Non-null values, ascending. Every reduction below works from this, so
 * they all agree on what "present" means and none of them can be made
 * order-dependent by a caller handing over rows in a different sequence.
 */
function present(values: readonly (number | null)[]): number[] {
  const kept: number[] = [];
  for (const value of values) {
    if (value !== null && Number.isFinite(value)) kept.push(value);
  }
  kept.sort((left, right) => left - right);
  return kept;
}

/**
 * The linear-interpolation quantile — numpy's default method, which is
 * what `numpy.quantile` and `numpy.median` use and therefore what the
 * Python fixture computes.
 *
 * Stated explicitly because "quantile" is the one name in §5.6.2's list
 * that has competing conventions in the wild: R alone ships nine. Naming
 * the method here, and pinning it with a golden test, is what keeps it
 * inside "no estimator choice".
 */
function linearQuantile(sorted: readonly number[], q: number): number | null {
  if (sorted.length === 0) return null;
  if (sorted.length === 1) return sorted[0]!;

  const index = q * (sorted.length - 1);
  const lo = Math.floor(index);
  const hi = Math.ceil(index);
  if (lo === hi) return sorted[lo]!;
  return sorted[lo]! + (index - lo) * (sorted[hi]! - sorted[lo]!);
}

/**
 * Apply one reduction to one column of values.
 *
 * Returns `null` for an empty set rather than zero — including for `sum`,
 * where Python's own `sum([])` is 0. That is a deliberate divergence from
 * the language and an agreement with §5.3.3: a bar with no underlying
 * rows has no height, and drawing it at zero would put it on the axis
 * beside genuine zeroes. The Python fixture encodes the same rule so the
 * golden test covers it rather than papering over it.
 *
 * `count` is the exception and returns 0, because "how many rows are
 * there" always has an answer and that answer is sometimes none.
 */
export function reduce(
  values: readonly (number | null)[],
  fn: Reduction,
  q = 0.5,
): number | null {
  const sorted = present(values);

  if (fn === "count") return sorted.length;
  if (sorted.length === 0) return null;

  switch (fn) {
    case "sum": {
      // Summed in ascending order, which is what `present` guarantees:
      // floating-point addition is not associative, so a reduction that
      // claims to be order-independent has to impose an order.
      let total = 0;
      for (const value of sorted) total += value;
      return total;
    }
    case "mean": {
      let total = 0;
      for (const value of sorted) total += value;
      return total / sorted.length;
    }
    case "median":
      return linearQuantile(sorted, 0.5);
    case "min":
      return sorted[0]!;
    case "max":
      return sorted[sorted.length - 1]!;
    case "quantile":
      return linearQuantile(sorted, q);
  }
}

/**
 * Number of rows carrying a value, which is what every surface means by
 * `n` when it prints one beside a reduced number (§5.6.3). Distinct from
 * the size of the group: a group of 40 player-gameweeks where 12 sit
 * below the minutes floor supports an `n` of 28, not 40.
 */
export function n(values: readonly (number | null)[]): number {
  return present(values).length;
}
