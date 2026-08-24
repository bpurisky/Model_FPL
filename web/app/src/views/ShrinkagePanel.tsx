/**
 * §5.4.7's annotated shrinkage panel.
 *
 * > "**The defender/goals-conceded finding gets a permanent annotated
 * > panel.** The `GOALS_CONCEDED_SHRINKAGE = 0.7` plateau is the most
 * > interesting result the project has produced; a shrinkage-vs-metrics
 * > plot showing the 0.6–0.85 plateau belongs on screen, not only in a
 * > docstring."
 *
 * Two curves against one axis, because the whole finding is that they
 * move in opposite directions: raise the goals-conceded weight and MAE
 * improves while defenders rank worse. Each carries its own baseline as a
 * dashed line — "beats the baseline" is what either curve is measured
 * against, and a curve without its bar is a shape with no argument
 * attached.
 *
 * **The panel renders two verdicts, and they disagree.** §4.4's criterion
 * as written is "within-position rank correlation", which the scorecard
 * reports as the unweighted mean across positions; on that reading both
 * bars clear from 0.5 to 0.9 and the shipped 0.7 sits comfortably inside.
 * The ablation comment in `analytics/projections.py` frames the same
 * trade on DEF alone, where it says the damage is "entirely concentrated";
 * on that reading the range is 0.5 to 0.6 and 0.7 is just past it.
 *
 * Showing only the flattering one would be choosing a measurement to suit
 * a constant, which is the failure this whole repo is arranged against.
 */

import type { ShrinkageFile, ShrinkagePoint } from "../data/schema";
import styles from "./Scorecard.module.css";

const W = 420;
const H = 250;
const PAD = { top: 16, right: 16, bottom: 34, left: 56 };
const INNER = { w: W - PAD.left - PAD.right, h: H - PAD.top - PAD.bottom };

const px = (shrinkage: number) => PAD.left + shrinkage * INNER.w;

interface Curve {
  path: string;
  py: (value: number) => number;
  lo: number;
  hi: number;
}

function buildCurve(points: ShrinkagePoint[], values: (number | null)[], baseline: number | null): Curve {
  const present = values.filter((value): value is number => value !== null);
  const all = baseline === null ? present : [...present, baseline];
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const py = (value: number) => PAD.top + INNER.h - ((value - lo) / (hi - lo || 1)) * INNER.h;

  let started = false;
  const path = points
    .map((point, index) => {
      const value = values[index];
      if (value === null || value === undefined) return "";
      const command = started ? "L" : "M";
      started = true;
      return `${command}${px(point.shrinkage).toFixed(1)},${py(value).toFixed(1)}`;
    })
    .join(" ");

  return { path, py, lo, hi };
}

function formatRange(points: ShrinkagePoint[]): string {
  if (points.length === 0) return "nothing";
  return `${points[0]!.shrinkage.toFixed(2)}–${points[points.length - 1]!.shrinkage.toFixed(2)}`;
}

interface PlotProps {
  title: string;
  better: string;
  points: ShrinkagePoint[];
  values: (number | null)[];
  baseline: number | null;
  baselineLabel: string;
  cleared: ShrinkagePoint[];
  fallback: number;
  digits: number;
}

function Plot({
  title,
  better,
  points,
  values,
  baseline,
  baselineLabel,
  cleared,
  fallback,
  digits,
}: PlotProps) {
  const curve = buildCurve(points, values, baseline);

  return (
    <figure className={styles.sweep}>
      <figcaption className={styles.sweepCaption}>
        {title} <span className={styles.muted}>({better} is better)</span>
      </figcaption>
      <svg
        className={styles.plot}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`${title} against goals-conceded shrinkage`}
      >
        {/* The range clearing both §4.4 bars, drawn behind everything. */}
        {cleared.length > 0 && (
          <rect
            x={px(cleared[0]!.shrinkage)}
            y={PAD.top}
            width={Math.max(
              px(cleared[cleared.length - 1]!.shrinkage) - px(cleared[0]!.shrinkage),
              2,
            )}
            height={INNER.h}
            className={styles.plateau}
          />
        )}

        {baseline !== null && (
          <>
            <line
              x1={PAD.left}
              y1={curve.py(baseline)}
              x2={PAD.left + INNER.w}
              y2={curve.py(baseline)}
              className={styles.diagonal}
            />
            <text
              x={PAD.left + INNER.w}
              y={curve.py(baseline) - 5}
              className={styles.tick}
              textAnchor="end"
            >
              {baselineLabel}
            </text>
          </>
        )}

        {/* Where the shipped constant sits, so a reader can see at a
            glance whether it is inside the shaded range or beside it. */}
        <line
          x1={px(fallback)}
          y1={PAD.top}
          x2={px(fallback)}
          y2={PAD.top + INNER.h}
          className={styles.marker}
        />
        <text x={px(fallback)} y={PAD.top + 9} className={styles.tick} textAnchor="middle">
          {fallback}
        </text>

        <line
          x1={PAD.left}
          y1={PAD.top + INNER.h}
          x2={PAD.left + INNER.w}
          y2={PAD.top + INNER.h}
          className={styles.axis}
        />

        <path d={curve.path} className={styles.curve} />

        {points.map((point, index) => {
          const value = values[index];
          if (value === null || value === undefined) return null;
          return (
            <circle
              key={point.shrinkage}
              cx={px(point.shrinkage)}
              cy={curve.py(value)}
              r={point.shrinkage === fallback ? 4 : 2.5}
              className={styles.point}
            >
              <title>
                shrinkage {point.shrinkage.toFixed(2)}: {value.toFixed(5)} over n=
                {point.n.toLocaleString()}
              </title>
            </circle>
          );
        })}

        {[0, 0.5, 1].map((value) => (
          <text
            key={value}
            x={px(value)}
            y={PAD.top + INNER.h + 14}
            className={styles.tick}
            textAnchor="middle"
          >
            {value}
          </text>
        ))}
        <text x={PAD.left - 8} y={PAD.top + 6} className={styles.tick} textAnchor="end">
          {curve.hi.toFixed(digits)}
        </text>
        <text x={PAD.left - 8} y={PAD.top + INNER.h} className={styles.tick} textAnchor="end">
          {curve.lo.toFixed(digits)}
        </text>
      </svg>
    </figure>
  );
}

export function ShrinkagePanel({ file }: { file: ShrinkageFile }) {
  const stated = file.points.filter((point) => point.beats_mae_bar && point.beats_spearman_bar);
  const focus = file.points.filter((point) => point.beats_mae_bar && point.beats_focus_bar);
  const atDefault = file.points.find((point) => point.shrinkage === file.default);

  return (
    <section className={styles.panel}>
      <h2 className={styles.panelTitle}>The goals-conceded shrinkage</h2>
      <p className={styles.panelSub}>
        Goals conceded is shared across a whole back line and swamped by single-match variance,
        so the model down-weights it. This is the sweep behind that constant, re-run across the
        full range — shaded where both §4.4 bars clear, with the shipped{" "}
        <span className="data">{file.default}</span> marked.
      </p>

      <div className={styles.sweeps}>
        <Plot
          title="MAE, all positions"
          better="lower"
          points={file.points}
          values={file.points.map((point) => point.mae)}
          baseline={file.baseline_mae}
          baselineLabel={file.baseline_mae_model}
          cleared={stated}
          fallback={file.default}
          digits={4}
        />
        <Plot
          title={`Spearman, ${file.focus_position} only`}
          better="higher"
          points={file.points}
          values={file.points.map((point) => point.spearman_focus)}
          baseline={file.baseline_spearman_focus}
          baselineLabel={file.baseline_spearman_model}
          cleared={focus}
          fallback={file.default}
          digits={4}
        />
      </div>

      <p className={styles.finding}>
        The two metrics move in opposite directions across the whole range, so this is a trade
        rather than an optimum. On §4.4&rsquo;s criterion as written — within-position rank
        correlation, meaning the mean across positions — both bars clear from{" "}
        <span className="data">{formatRange(stated)}</span>, and the shipped{" "}
        <span className="data">{file.default}</span> sits inside it. Measured on{" "}
        <span className="data">{file.focus_position}</span> alone, where the model&rsquo;s own
        ablation comment says the damage is concentrated, the range is only{" "}
        <span className="data">{formatRange(focus)}</span> and{" "}
        <span className="data">{file.default}</span> is just past it
        {atDefault?.spearman_focus != null && file.baseline_spearman_focus != null ? (
          <>
            {" "}
            — <span className="data">{atDefault.spearman_focus.toFixed(4)}</span> against the
            baseline&rsquo;s{" "}
            <span className="data">{file.baseline_spearman_focus.toFixed(4)}</span>
          </>
        ) : null}
        . Both are true of different measurements; the panel shows both rather than the one
        that flatters the constant.
      </p>
    </section>
  );
}
