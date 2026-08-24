/**
 * Grouped rows -> plottable series.
 *
 * The join between `query/` and the marks: DuckDB grouped the panel and
 * collected the values, `reduce.ts` turns each collection into one
 * number, and this decides what shape those numbers make. It is pure and
 * has no React and no SVG in it, so `data.test.ts` can check the parts
 * that are easy to get quietly wrong — an empty group drawn at zero, a
 * facet ordered by hash iteration, a series that loses its name.
 *
 * Nulls survive the whole way through (§5.3.3). A group whose values were
 * all below the minutes floor reduces to null, and a null point is
 * **omitted from the mark rather than drawn at zero** — the distinction
 * §5.14.9 requires every view to preserve. A bar of height zero and no
 * bar at all are different claims.
 */

import { reduce, n as countPresent, type Reduction } from "../query/reduce";
import type { GroupedRow } from "../query/panel";
import type { MarkPlan } from "./mark";
import type { Encoding } from "./spec";

export interface PlotPoint {
  /** Stable identity for React keys and for hit-testing. */
  id: string;
  /** What the tooltip names this point. */
  label: string;
  x: number | string | null;
  y: number | string | null;
  color: number | string | null;
  /** Rows behind the reduction (§5.6.3 — every derived number shows n). */
  n: number;
}

export interface PlotSeries {
  /** The colour category, or null when the mark has one series. */
  name: string | null;
  points: PlotPoint[];
}

export interface PlotFacet {
  /** The wrap value, or null when the mark is not wrapped. */
  name: string | null;
  series: PlotSeries[];
}

export interface Plot {
  facets: PlotFacet[];
  /** Distinct x values in order, for a categorical or ordinal axis. */
  xDomain: (string | number)[];
  /** Distinct y values in order, for a rect mark's categorical y. */
  yDomain: (string | number)[];
  /** Total points drawn, across every facet and series. */
  count: number;
  /** Points dropped because their reduction was null. */
  dropped: number;
}

/** The panel columns that identify a player, in the order they read. */
export const PLAYER_KEYS = ["element_id", "name", "team", "position"] as const;

/**
 * The group keys a plan needs in SQL. `"player"` expands to the four
 * columns that identify one, because a scatter needs a name to put in a
 * tooltip and a position to colour by — grouping on `element_id` alone
 * would hand back an integer nobody can read.
 */
export function groupKeysFor(plan: MarkPlan, encoding: Encoding): string[] {
  const keys: string[] = [];
  for (const entry of plan.groupBy) {
    if (entry === "player") {
      keys.push(...PLAYER_KEYS);
    } else {
      const key = encoding[entry];
      if (key) keys.push(key);
    }
  }
  if (encoding.wrap) keys.push(encoding.wrap);
  // A channel can appear in both groupBy and wrap; SQL will not accept
  // the same column twice in a GROUP BY list.
  return [...new Set(keys)];
}

/** The columns to `list(...)` — everything the plan reduces. */
export function collectKeysFor(plan: MarkPlan, encoding: Encoding): string[] {
  const keys = plan.reduced
    .map((channel) => encoding[channel])
    .filter((key): key is string => key !== null);
  return [...new Set(keys)];
}

function scalar(value: string | number | null): string {
  return value === null ? "—" : String(value);
}

/**
 * Build the plot.
 *
 * `aggregate` is applied to every reduced channel. §5.4.2 offers one
 * aggregation control rather than one per channel, and that is the right
 * call for a scatter: `mean` on x and `median` on y would produce a point
 * that describes no player and no gameweek.
 */
export function buildPlot(
  rows: GroupedRow[],
  plan: MarkPlan,
  encoding: Encoding,
): Plot {
  const aggregate = encoding.aggregate as Reduction;
  const reduced = new Set(plan.reduced);

  /** A channel's value for one group: reduced if collected, else the key. */
  const channelValue = (
    row: GroupedRow,
    channel: "x" | "y" | "color",
  ): { value: number | string | null; n: number } => {
    const key = encoding[channel];
    if (!key) return { value: null, n: 0 };
    if (reduced.has(channel)) {
      const values = row.values[key] ?? [];
      return { value: reduce(values, aggregate), n: countPresent(values) };
    }
    const raw = row.key[key];
    return { value: raw ?? null, n: 0 };
  };

  const facetMap = new Map<string | null, Map<string | null, PlotPoint[]>>();
  const xSeen = new Map<string, string | number>();
  const ySeen = new Map<string, string | number>();
  let dropped = 0;
  let count = 0;

  for (const row of rows) {
    const x = channelValue(row, "x");
    const y = channelValue(row, "y");
    const color = channelValue(row, "color");

    /*
     * A mark needs its own coordinates. A point whose reduction came back
     * null has no position on the axis and is dropped rather than placed
     * at zero — the count is reported so the surface can say how many.
     */
    const needsY = plan.mark !== "histogram";
    const needsColor = plan.mark === "rect";
    if (x.value === null || (needsY && y.value === null) || (needsColor && color.value === null)) {
      dropped += 1;
      continue;
    }

    const facetName = encoding.wrap ? scalar(row.key[encoding.wrap] ?? null) : null;
    const seriesName = plan.series && encoding.color ? scalar(row.key[encoding.color] ?? null) : null;

    const label = PLAYER_KEYS.every((key) => key in row.key)
      ? `${row.key["name"]} (${row.key["team"]})`
      : [x.value, y.value].filter((entry) => entry !== null).join(" · ");

    const point: PlotPoint = {
      /*
       * The whole group key, not `element_id` alone.
       *
       * A player who moves club inside the filtered range has two rows
       * here — the panel carries `team` per gameweek — so an id taken
       * from the element alone collides, and React quietly drops one of
       * the two marks. Same for any grouping where one channel repeats.
       * The key is unique by construction; the id should be the key.
       */
      id: Object.values(row.key).map(scalar).join(""),
      label,
      x: x.value,
      y: y.value,
      color: color.value,
      n: Math.max(x.n, y.n, color.n),
    };

    if (typeof x.value === "string" || plan.mark === "bar" || plan.mark === "line" || plan.mark === "rect") {
      xSeen.set(scalar(x.value), x.value as string | number);
    }
    if (plan.mark === "rect" && y.value !== null) {
      ySeen.set(scalar(y.value), y.value as string | number);
    }

    let facet = facetMap.get(facetName);
    if (!facet) {
      facet = new Map();
      facetMap.set(facetName, facet);
    }
    let series = facet.get(seriesName);
    if (!series) {
      series = [];
      facet.set(seriesName, series);
    }
    series.push(point);
    count += 1;
  }

  /*
   * Ordered explicitly rather than left to insertion order. Insertion
   * order comes from the SQL ORDER BY today, which is the right order —
   * but it is the right order by coincidence, and a chart whose series
   * legend reshuffles when a filter changes is a chart nobody trusts.
   */
  const facets: PlotFacet[] = [...facetMap.entries()]
    .sort(([left], [right]) => compareNames(left, right))
    .map(([name, seriesMap]) => ({
      name,
      series: [...seriesMap.entries()]
        .sort(([left], [right]) => compareNames(left, right))
        .map(([seriesName, points]) => ({ name: seriesName, points })),
    }));

  return {
    facets,
    xDomain: sortDomain([...xSeen.values()]),
    yDomain: sortDomain([...ySeen.values()]),
    count,
    dropped,
  };
}

function compareNames(left: string | null, right: string | null): number {
  if (left === right) return 0;
  if (left === null) return -1;
  if (right === null) return 1;
  return left.localeCompare(right, undefined, { numeric: true });
}

/** Numeric domains sort numerically; gameweek 10 follows gameweek 9. */
function sortDomain(values: (string | number)[]): (string | number)[] {
  return values.sort((left, right) => {
    if (typeof left === "number" && typeof right === "number") return left - right;
    return String(left).localeCompare(String(right), undefined, { numeric: true });
  });
}

/**
 * Equal-width bins over the observed extent, for the histogram mark.
 *
 * A fixed count rather than Freedman-Diaconis or Sturges. Those are
 * *rules* that read a distribution's spread and choose for you, which is
 * an estimator by another name and the sort of thing §5.6 exists to keep
 * out of the browser. A fixed width is a display choice the user can see
 * and the axis states.
 */
export const HISTOGRAM_BINS = 24;

export interface Bin {
  lo: number;
  hi: number;
  count: number;
}

export function bin(values: number[], bins = HISTOGRAM_BINS): Bin[] {
  if (values.length === 0) return [];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  if (lo === hi) return [{ lo, hi, count: values.length }];

  const width = (hi - lo) / bins;
  const out: Bin[] = Array.from({ length: bins }, (_, index) => ({
    lo: lo + index * width,
    hi: lo + (index + 1) * width,
    count: 0,
  }));

  for (const value of values) {
    // The top edge belongs to the last bin rather than to a bin that
    // does not exist.
    const index = Math.min(Math.floor((value - lo) / width), bins - 1);
    out[index]!.count += 1;
  }
  return out;
}
