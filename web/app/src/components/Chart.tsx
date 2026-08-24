/**
 * The five marks (§5.4.2), drawn.
 *
 * Hand-rolled SVG rather than Visx, continuing what `RankScatter` and
 * `Matrix` already do. §5.2 permits either — "Visx (or D3 primitives
 * directly)" — and the argument that settled it there settles it here:
 * these are five simple marks over scales this file already owns, and a
 * chart library's value is in the scales and the theme, one of which we
 * have and the other of which §5.8 forbids importing.
 *
 * **Series colour samples the diverging ramp.** §5.8.2 defines no
 * categorical palette, and inventing one would spend the visual budget
 * §5.8.5 explicitly assigns elsewhere ("Graph Builder, Explorer, and
 * Trend are utilitarian"). Sampling the brand scale keeps the app to one
 * set of colours. Recorded as a §5.16 deviation (D8) because it is a real
 * if small distortion: an ordered ramp on unordered categories implies a
 * sequence that is not there. The mitigation is that colour is never the
 * only encoding — every series also carries a dash pattern and a legend
 * entry, so the chart survives greyscale, which §5.10 requires of the
 * heat maps and which costs nothing to extend to the lines.
 */

import { useId } from "react";
import type { ColumnSpec } from "../data/schema";
import { divergingColor } from "../design/scale";
import { bin, type Plot, type PlotSeries } from "../encoding/data";
import type { MarkPlan } from "../encoding/mark";
import styles from "./Chart.module.css";

/** Above this many series a legend stops being readable and so does the chart. */
export const MAX_SERIES = 8;

const W = 520;
const H = 340;
const PAD = { top: 16, right: 16, bottom: 46, left: 62 };

/** Dash patterns, so a series is identifiable without colour. */
const DASHES = ["", "5 3", "2 3", "8 3 2 3", "1 3", "12 4", "6 2 2 2", "3 6"];

export interface ChartProps {
  plot: Plot;
  plan: MarkPlan;
  xLabel: string;
  yLabel: string;
  colorLabel: string | null;
  /** For number formatting and the tooltip's own vocabulary. */
  columns: ReadonlyMap<string, ColumnSpec>;
  xKey: string | null;
  yKey: string | null;
  colorKey: string | null;
}

/**
 * Positions along the diverging ramp for `count` series. Evenly spaced
 * across the full width so adjacent series are maximally separated, and
 * never landing exactly on zero, where the scale is `--panel` by design
 * and a mark would vanish into its own background.
 */
function seriesColor(index: number, count: number): string {
  if (count <= 1) return "var(--paper)";
  const t = count === 1 ? 0 : index / (count - 1);
  const value = -0.95 + t * 1.9;
  return divergingColor(Math.abs(value) < 0.25 ? Math.sign(value) * 0.25 || 0.25 : value);
}

export function Chart({
  plot,
  plan,
  xLabel,
  yLabel,
  colorLabel,
  columns,
  xKey,
  yKey,
  colorKey,
}: ChartProps) {
  const seriesNames = [
    ...new Set(plot.facets.flatMap((facet) => facet.series.map((series) => series.name))),
  ].filter((name): name is string => name !== null);

  if (seriesNames.length > MAX_SERIES) {
    return (
      <p className={styles.refusal} role="note">
        <span className="data">{colorLabel}</span> has {seriesNames.length} distinct values,
        which is {seriesNames.length} overlapping series. Move it to Wrap to get one panel
        each, or filter down to at most {MAX_SERIES}.
      </p>
    );
  }

  return (
    <div className={styles.chart}>
      {seriesNames.length > 0 && (
        <ul className={styles.legend} aria-label={`${colorLabel} series`}>
          {seriesNames.map((name, index) => (
            <li key={name} className={styles.legendItem}>
              <svg className={styles.swatch} viewBox="0 0 24 8" aria-hidden="true">
                <line
                  x1="0"
                  y1="4"
                  x2="24"
                  y2="4"
                  stroke={seriesColor(index, seriesNames.length)}
                  strokeWidth="2"
                  strokeDasharray={DASHES[index % DASHES.length]}
                />
              </svg>
              <span className="data">{name}</span>
            </li>
          ))}
        </ul>
      )}

      <div className={styles.facets} data-many={plot.facets.length > 1 || undefined}>
        {plot.facets.map((facet) => (
          <figure key={facet.name ?? "__all"} className={styles.facet}>
            {facet.name !== null && (
              <figcaption className={styles.facetName}>{facet.name}</figcaption>
            )}
            <Panel
              series={facet.series}
              plan={plan}
              plot={plot}
              xLabel={xLabel}
              yLabel={yLabel}
              seriesNames={seriesNames}
              columns={columns}
              xKey={xKey}
              yKey={yKey}
              colorKey={colorKey}
            />
          </figure>
        ))}
      </div>
    </div>
  );
}

interface PanelProps {
  series: PlotSeries[];
  plan: MarkPlan;
  plot: Plot;
  xLabel: string;
  yLabel: string;
  seriesNames: string[];
  columns: ReadonlyMap<string, ColumnSpec>;
  xKey: string | null;
  yKey: string | null;
  colorKey: string | null;
}

function Panel(props: PanelProps) {
  const { series, plan, plot, xLabel, yLabel, seriesNames, columns } = props;
  // The registry entries behind the three value channels, so every
  // number below renders in the unit the pipeline declared for it.
  const xSpec = props.xKey ? columns.get(props.xKey) : undefined;
  const ySpec = props.yKey ? columns.get(props.yKey) : undefined;
  const colorSpec = props.colorKey ? columns.get(props.colorKey) : undefined;
  const clip = useId();

  const points = series.flatMap((entry) => entry.points);
  if (points.length === 0) {
    return <p className={styles.empty}>No rows survive the current filters.</p>;
  }

  /*
   * Whether the x labels have to be rotated, decided here rather than in
   * `Axes` because it changes how much vertical room the plot area has.
   * Long category names on a crowded axis collide, and thinning them is
   * the wrong fix when every one is a team the reader is looking for.
   */
  const xLabels =
    plan.mark === "bar" || plan.mark === "rect" || plan.mark === "line"
      ? plot.xDomain.map(String)
      : [];
  const longest = xLabels.reduce((max, label) => Math.max(max, label.length), 0);
  const perLabel = (W - PAD.left - PAD.right) / Math.max(xLabels.length, 1);
  const rotate = longest > 3 && perLabel < longest * 6.5;
  // 0.64 = sin(40 degrees), the vertical reach of a rotated label, plus
  // room for the axis title underneath it.
  const padBottom = rotate ? Math.min(PAD.bottom + longest * 6 * 0.64, 130) : PAD.bottom;

  const inner = { w: W - PAD.left - PAD.right, h: H - PAD.top - padBottom };

  // --- scales --------------------------------------------------------
  const categorical = plan.mark === "bar" || plan.mark === "rect" || plot.xDomain.length > 0;
  const xNumeric = points
    .map((point) => point.x)
    .filter((value): value is number => typeof value === "number");
  const yNumeric = points
    .map((point) => point.y)
    .filter((value): value is number => typeof value === "number");

  const useBandX = plan.mark === "bar" || plan.mark === "rect";
  const band = plot.xDomain.length > 0 ? inner.w / plot.xDomain.length : inner.w;

  const xIndex = new Map(plot.xDomain.map((value, index) => [String(value), index]));
  const yIndex = new Map(plot.yDomain.map((value, index) => [String(value), index]));

  const xExtent = extent(xNumeric);
  /*
   * A bar encodes its value by *length*, so its scale has to include
   * zero. Without this the axis starts at the smallest value and the
   * bars become a picture of the differences between them rather than of
   * the values themselves — measured on real data, a mean xGI of 1.20
   * drew about ten times the bar of 0.39, for a true ratio of three.
   * That is the classic truncated-axis lie, and it is not one this repo
   * gets to tell while §5.0.1 calls the tool a diagnostic instrument.
   *
   * Point and line marks keep a fitted scale: they encode by *position*,
   * where the reader takes the axis labels as the reference, and forcing
   * zero on a rate that lives between 0.4 and 1.2 would flatten every
   * real difference into the top fifth of the plot.
   */
  const yExtent =
    plan.mark === "bar" || plan.mark === "histogram"
      ? withZero(extent(yNumeric))
      : extent(yNumeric);

  const px = (value: number | string | null): number => {
    if (value === null) return PAD.left;
    if (useBandX || (categorical && typeof value === "string")) {
      const index = xIndex.get(String(value)) ?? 0;
      return PAD.left + index * band + band / 2;
    }
    if (plan.mark === "line" && xIndex.size > 0) {
      const index = xIndex.get(String(value)) ?? 0;
      return PAD.left + (xIndex.size === 1 ? inner.w / 2 : (index / (xIndex.size - 1)) * inner.w);
    }
    return PAD.left + scale(Number(value), xExtent) * inner.w;
  };

  const py = (value: number | string | null): number => {
    if (value === null) return PAD.top + inner.h;
    if (plan.mark === "rect") {
      const index = yIndex.get(String(value)) ?? 0;
      const rowH = inner.h / Math.max(yIndex.size, 1);
      return PAD.top + index * rowH + rowH / 2;
    }
    return PAD.top + inner.h - scale(Number(value), yExtent) * inner.h;
  };

  const colorExtent = extent(
    points.map((point) => point.color).filter((value): value is number => typeof value === "number"),
  );

  return (
    <svg
      className={styles.plot}
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`${plan.mark} of ${yLabel} against ${xLabel}, ${points.length} marks`}
    >
      <defs>
        <clipPath id={clip}>
          <rect x={PAD.left} y={PAD.top} width={inner.w} height={inner.h} />
        </clipPath>
      </defs>

      <Axes
        plan={plan}
        plot={plot}
        inner={inner}
        xExtent={xExtent}
        yExtent={yExtent}
        xLabel={xLabel}
        yLabel={yLabel}
        band={band}
        rotate={rotate}
        padBottom={padBottom}
      />

      <g clipPath={`url(#${clip})`}>
        {plan.mark === "histogram" && <Histogram values={xNumeric} inner={inner} />}

        {plan.mark === "point" &&
          series.map((entry) => {
            const index = entry.name ? seriesNames.indexOf(entry.name) : 0;
            return entry.points.map((point) => (
              <circle
                key={`${entry.name ?? ""}-${point.id}`}
                cx={px(point.x)}
                cy={py(point.y)}
                r={2.8}
                className={styles.point}
                fill={
                  typeof point.color === "number"
                    ? divergingColor(-1 + 2 * scale(point.color, colorExtent))
                    : entry.name
                      ? seriesColor(index, seriesNames.length)
                      : "var(--paper)"
                }
              >
                <title>
                  {point.label} — {xLabel} {format(point.x, xSpec)}, {yLabel} {format(point.y, ySpec)}
                  {point.n > 0 ? ` (n=${point.n})` : ""}
                </title>
              </circle>
            ));
          })}

        {plan.mark === "bar" &&
          series.flatMap((entry) =>
            entry.points.map((point) => {
              const zero = py(Math.max(0, Math.min(yExtent[0], yExtent[1])));
              const top = py(point.y);
              return (
                <rect
                  key={point.id}
                  x={px(point.x) - band * 0.35}
                  y={Math.min(top, zero)}
                  width={band * 0.7}
                  height={Math.max(Math.abs(zero - top), 1)}
                  className={styles.bar}
                >
                  <title>
                    {String(point.x)} — {yLabel} {format(point.y, ySpec)}
                    {point.n > 0 ? ` over n=${point.n}` : ""}
                  </title>
                </rect>
              );
            }),
          )}

        {plan.mark === "line" &&
          series.map((entry) => {
            const index = entry.name ? seriesNames.indexOf(entry.name) : 0;
            const ordered = [...entry.points].sort(
              (left, right) => px(left.x) - px(right.x),
            );
            const d = ordered
              .map((point, i) => `${i === 0 ? "M" : "L"}${px(point.x)},${py(point.y)}`)
              .join(" ");
            return (
              <g key={entry.name ?? "__one"}>
                <path
                  d={d}
                  className={styles.line}
                  stroke={entry.name ? seriesColor(index, seriesNames.length) : "var(--paper)"}
                  strokeDasharray={entry.name ? DASHES[index % DASHES.length] : ""}
                />
                {ordered.map((point) => (
                  <circle
                    key={point.id}
                    cx={px(point.x)}
                    cy={py(point.y)}
                    r={2}
                    fill={entry.name ? seriesColor(index, seriesNames.length) : "var(--paper)"}
                  >
                    <title>
                      {entry.name ? `${entry.name} — ` : ""}
                      {xLabel} {String(point.x)}, {yLabel} {format(point.y, ySpec)}
                      {point.n > 0 ? ` over n=${point.n}` : ""}
                    </title>
                  </circle>
                ))}
              </g>
            );
          })}

        {plan.mark === "rect" &&
          series.flatMap((entry) =>
            entry.points.map((point) => {
              const rowH = inner.h / Math.max(yIndex.size, 1);
              const value = typeof point.color === "number" ? point.color : null;
              return (
                <rect
                  key={point.id}
                  x={px(point.x) - band / 2}
                  y={py(point.y) - rowH / 2}
                  width={band}
                  height={rowH}
                  /*
                   * §5.10: a heat map cell never encodes by colour alone.
                   * These carry their value as text below, and a cell
                   * whose reduction was null is left unpainted rather
                   * than painted at the midpoint.
                   */
                  fill={
                    value === null
                      ? "transparent"
                      : divergingColor(-1 + 2 * scale(value, colorExtent))
                  }
                  stroke="var(--ground)"
                  strokeWidth={1}
                >
                  <title>
                    {String(point.x)} × {String(point.y)} — {format(point.color, colorSpec)}
                    {point.n > 0 ? ` over n=${point.n}` : ""}
                  </title>
                </rect>
              );
            }),
          )}

        {plan.mark === "rect" &&
          series.flatMap((entry) =>
            entry.points.map((point) => (
              <text
                key={`t-${point.id}`}
                x={px(point.x)}
                y={py(point.y)}
                className={styles.cellValue}
                textAnchor="middle"
                dominantBaseline="central"
              >
                {format(point.color, colorSpec)}
              </text>
            )),
          )}
      </g>
    </svg>
  );
}

function Histogram({ values, inner }: { values: number[]; inner: { w: number; h: number } }) {
  const bins = bin(values);
  if (bins.length === 0) return null;
  const tallest = Math.max(...bins.map((entry) => entry.count));
  const width = inner.w / bins.length;

  return (
    <>
      {bins.map((entry, index) => {
        const height = (entry.count / tallest) * inner.h;
        return (
          <rect
            key={`${entry.lo}-${index}`}
            x={PAD.left + index * width}
            y={PAD.top + inner.h - height}
            width={Math.max(width - 1, 1)}
            height={height}
            className={styles.bar}
          >
            <title>
              {entry.lo.toFixed(2)} to {entry.hi.toFixed(2)} — {entry.count} players
            </title>
          </rect>
        );
      })}
    </>
  );
}

interface AxesProps {
  plan: MarkPlan;
  plot: Plot;
  inner: { w: number; h: number };
  xExtent: [number, number];
  yExtent: [number, number];
  xLabel: string;
  yLabel: string;
  band: number;
  rotate: boolean;
  padBottom: number;
}

function Axes({
  plan,
  plot,
  inner,
  xExtent,
  yExtent,
  xLabel,
  yLabel,
  band,
  rotate,
  padBottom,
}: AxesProps) {
  const bandX = plan.mark === "bar" || plan.mark === "rect";
  const categoricalX = bandX || plan.mark === "line";

  const xTicks: { at: number; label: string }[] = categoricalX
    ? plot.xDomain.map((value, index) => ({
        at: bandX
          ? PAD.left + index * band + band / 2
          : PAD.left +
            (plot.xDomain.length === 1
              ? inner.w / 2
              : (index / (plot.xDomain.length - 1)) * inner.w),
        label: String(value),
      }))
    : ticks(xExtent).map((value) => ({
        at: PAD.left + scale(value, xExtent) * inner.w,
        label: tickLabel(value),
      }));

  const yTicks =
    plan.mark === "rect"
      ? plot.yDomain.map((value, index) => {
          const rowH = inner.h / Math.max(plot.yDomain.length, 1);
          return { at: PAD.top + index * rowH + rowH / 2, label: String(value) };
        })
      : ticks(yExtent).map((value) => ({
          at: PAD.top + inner.h - scale(value, yExtent) * inner.h,
          label: tickLabel(value),
        }));

  /*
   * Thin only when the labels are short enough that dropping some loses
   * nothing -- gameweek numbers, where the axis is self-evidently
   * sequential. Rotated word labels are all kept.
   */
  const step = !rotate && xTicks.length > 14 ? Math.ceil(xTicks.length / 14) : 1;

  return (
    <g aria-hidden="true">
      {yTicks.map((tick) => (
        <line
          key={`g${tick.at}`}
          x1={PAD.left}
          y1={tick.at}
          x2={PAD.left + inner.w}
          y2={tick.at}
          className={styles.grid}
        />
      ))}

      <line
        x1={PAD.left}
        y1={PAD.top + inner.h}
        x2={PAD.left + inner.w}
        y2={PAD.top + inner.h}
        className={styles.axis}
      />
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + inner.h} className={styles.axis} />

      {xTicks.map((tick, index) =>
        index % step === 0 ? (
          <text
            key={`x${tick.at}-${tick.label}`}
            x={tick.at}
            y={PAD.top + inner.h + (rotate ? 10 : 14)}
            className={styles.tick}
            textAnchor={rotate ? "end" : "middle"}
            transform={
              rotate ? `rotate(-40 ${tick.at} ${PAD.top + inner.h + 10})` : undefined
            }
          >
            {tick.label}
          </text>
        ) : null,
      )}

      {yTicks.map((tick) => (
        <text
          key={`y${tick.at}-${tick.label}`}
          x={PAD.left - 8}
          y={tick.at}
          className={styles.tick}
          textAnchor="end"
          dominantBaseline="central"
        >
          {tick.label}
        </text>
      ))}

      <text
        x={PAD.left + inner.w / 2}
        y={PAD.top + inner.h + padBottom - 6}
        className={styles.axisLabel}
        textAnchor="middle"
      >
        {xLabel}
      </text>
      <text
        x={12}
        y={PAD.top + inner.h / 2}
        className={styles.axisLabel}
        textAnchor="middle"
        transform={`rotate(-90 12 ${PAD.top + inner.h / 2})`}
      >
        {yLabel}
      </text>
    </g>
  );
}

// --- small numeric helpers ------------------------------------------

function extent(values: number[]): [number, number] {
  if (values.length === 0) return [0, 1];
  let lo = Infinity;
  let hi = -Infinity;
  for (const value of values) {
    if (value < lo) lo = value;
    if (value > hi) hi = value;
  }
  if (lo === hi) return [lo - 0.5, hi + 0.5];
  return [lo, hi];
}

/** An extent widened to include zero, for marks that encode by length. */
function withZero([lo, hi]: [number, number]): [number, number] {
  return [Math.min(0, lo), Math.max(0, hi)];
}

function scale(value: number, [lo, hi]: [number, number]): number {
  if (hi === lo) return 0.5;
  return (value - lo) / (hi - lo);
}

/** Five evenly spaced ticks. A "nice number" algorithm is not worth it here. */
function ticks([lo, hi]: [number, number], count = 5): number[] {
  return Array.from({ length: count }, (_, index) => lo + ((hi - lo) * index) / (count - 1));
}

function tickLabel(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude >= 1000) return `${Math.round(value / 100) / 10}k`;
  if (magnitude >= 10) return String(Math.round(value));
  if (magnitude >= 1) return value.toFixed(1);
  return value.toFixed(2);
}

/**
 * §5.3.3: a null is an em dash, never a zero.
 *
 * `spec` is the registry entry, and its `format` is honoured when there
 * is one. That is not cosmetic: `clean_sheet_prob` and
 * `minutes_reliability` are declared `.0%`, and rendering a 0.23 clean
 * sheet probability as "0.23" beside a `.2f` rate invites reading it as
 * the same kind of number. The registry is where the pipeline said what
 * kind it is (§5.3.5).
 */
function format(value: number | string | null, spec?: ColumnSpec): string {
  if (value === null) return "—";
  if (typeof value === "string") return value;

  switch (spec?.format) {
    case "d":
      return value.toFixed(0);
    case ".0%":
      return `${(value * 100).toFixed(0)}%`;
    case ".1%":
      return `${(value * 100).toFixed(1)}%`;
    case ".2f":
      return value.toFixed(2);
    case ".1f":
      return value.toFixed(1);
    default:
      break;
  }

  const magnitude = Math.abs(value);
  if (magnitude >= 1000) return value.toFixed(0);
  if (magnitude >= 10) return value.toFixed(1);
  return value.toFixed(2);
}
