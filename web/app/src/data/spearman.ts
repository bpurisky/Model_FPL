/**
 * §5.6.1's permitted exception — the Spearman port.
 *
 * **Python source: `backtest/report.py`, functions `_pearson` and
 * `spearman`.** This is a deliberate line-by-line port of that method,
 * not an independent implementation, because §5.6 names the exact risk:
 * "a second Spearman implementation in JavaScript — with its own
 * tie-handling — creates a number that no test covers and that can
 * silently disagree with the paper result."
 *
 * The three conditions §5.6.1 attaches are all met:
 *   1. this module is the single deliberate port, and names its source;
 *   2. `data/web/v1/golden_spearman.json` ships 600 pairs with
 *      Python-computed rho and the values they were computed over;
 *   3. `spearman.golden.test.ts` fails on any disagreement beyond 1e-9.
 *
 * **No p-value.** §5.6 forbids significance testing in the browser
 * without exception, and the exception here covers correlation only. A
 * client-computed cell therefore carries rho and n and nothing else, and
 * the UI states that rather than showing a p from a different population.
 */

/**
 * Ties-averaged ranks — polars' `rank()` default method, which is what
 * `report.spearman` ranks with.
 *
 * The tie convention is the whole reason §5.6.1 insists on a port rather
 * than any Spearman: a fresh implementation can quietly choose "min" or
 * "ordinal" and produce a number that is wrong only where the data has
 * repeated values, which FPL data has everywhere.
 */
export function averageRanks(values: number[]): number[] {
  const order = values.map((value, index) => ({ value, index }));
  order.sort((left, right) => left.value - right.value);

  const ranks = new Array<number>(values.length);
  let i = 0;
  while (i < order.length) {
    let j = i;
    while (j + 1 < order.length && order[j + 1]!.value === order[i]!.value) j += 1;
    const shared = (i + j) / 2 + 1;
    for (let k = i; k <= j; k += 1) ranks[order[k]!.index] = shared;
    i = j + 1;
  }
  return ranks;
}

/**
 * Sample Pearson, matching `report._pearson`: covariance over n-1 and
 * standard deviations with the same denominator. Returns NaN where the
 * Python returns NaN — fewer than two points, or a series with no spread
 * — because a caller must be able to tell "no correlation is defined"
 * from "the correlation is zero" (§5.3.3).
 */
export function pearson(x: number[], y: number[]): number {
  const n = x.length;
  if (n < 2 || y.length !== n) return Number.NaN;

  const meanX = x.reduce((sum, value) => sum + value, 0) / n;
  const meanY = y.reduce((sum, value) => sum + value, 0) / n;

  let covariance = 0;
  let varianceX = 0;
  let varianceY = 0;
  for (let i = 0; i < n; i += 1) {
    const dx = x[i]! - meanX;
    const dy = y[i]! - meanY;
    covariance += dx * dy;
    varianceX += dx * dx;
    varianceY += dy * dy;
  }
  covariance /= n - 1;
  const sdX = Math.sqrt(varianceX / (n - 1));
  const sdY = Math.sqrt(varianceY / (n - 1));
  if (sdX === 0 || sdY === 0) return Number.NaN;

  return covariance / (sdX * sdY);
}

export interface Correlation {
  /** Null where none is defined, never zero (§5.3.3). */
  rho: number | null;
  /** Rows where *both* metrics were present. */
  n: number;
}

/**
 * Spearman over the pairs where both values are present.
 *
 * The pairwise filter is not tidiness. `report.spearman` ranks each
 * series independently and polars leaves nulls null, so feeding it a
 * column that is half null and reading `n` as the column length reports a
 * *perfect* rank correlation as rho=0.23 over n=10. That happened in this
 * project, in Python, and the same shape is available here — four of the
 * sixteen metrics do not exist before 2025-26, so incomplete pairs are
 * the normal case rather than an edge.
 */
export function spearman(
  xs: readonly (number | null)[],
  ys: readonly (number | null)[],
): Correlation {
  const x: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < xs.length; i += 1) {
    const a = xs[i];
    const b = ys[i];
    if (a === null || a === undefined || b === null || b === undefined) continue;
    x.push(a);
    y.push(b);
  }

  if (x.length < 2) return { rho: null, n: x.length };

  const rho = pearson(averageRanks(x), averageRanks(y));
  return { rho: Number.isNaN(rho) ? null : rho, n: x.length };
}
