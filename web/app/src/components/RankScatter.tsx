import { useMemo, useState } from "react";
import type { ColumnSpec, PlayerRow } from "../data/schema";
import { formatRho } from "../design/scale";
import styles from "./RankScatter.module.css";

export interface RankScatterProps {
  a: string;
  b: string;
  players: PlayerRow[];
  columns: Map<string, ColumnSpec>;
  position: string;
  season: string;
  /** rho for the selected pair, from the matrix, for the caption. */
  rho: number | null;
  n: number;
}

interface Point {
  id: number;
  name: string;
  team: string;
  x: number;
  y: number;
  rawX: number;
  rawY: number;
}

/**
 * Ties-averaged ranks over the plotted set.
 *
 * §5.6 forbids the browser computing "percentile ranks against a
 * population" — that is a normalization, and it ships from
 * `normalize.py`. This is the other side of §5.6.2's own distinction: it
 * orders the points the user is currently looking at, which is a
 * reduction over a filtered set rather than a position against a
 * reference group the model defines. The numbers never leave this chart
 * and are never compared against an exported z-score.
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

const PAD = 34;
const SIZE = 320;

/**
 * §5.4.1: "Click a cell → rank scatter below the matrix: the two metrics
 * on *rank* axes, because that is what Spearman measures and plotting raw
 * values beside a rank statistic misleads. Raw-axis toggle offered,
 * defaulting to rank."
 */
export function RankScatter({
  a,
  b,
  players,
  columns,
  position,
  season,
  rho,
  n,
}: RankScatterProps) {
  const [useRank, setUseRank] = useState(true);

  const points = useMemo<Point[]>(() => {
    const eligible = players.filter(
      (player) =>
        (position === "all" || player.position === position) &&
        player.metrics[a]?.value !== null &&
        player.metrics[a]?.value !== undefined &&
        player.metrics[b]?.value !== null &&
        player.metrics[b]?.value !== undefined,
    );
    const xs = eligible.map((player) => player.metrics[a]!.value!);
    const ys = eligible.map((player) => player.metrics[b]!.value!);
    const rx = averageRanks(xs);
    const ry = averageRanks(ys);
    return eligible.map((player, index) => ({
      id: player.element_id,
      name: player.name,
      team: player.team,
      x: rx[index]!,
      y: ry[index]!,
      rawX: xs[index]!,
      rawY: ys[index]!,
    }));
  }, [players, position, a, b]);

  const [xKey, yKey] = useRank ? (["x", "y"] as const) : (["rawX", "rawY"] as const);
  const xs = points.map((point) => point[xKey]);
  const ys = points.map((point) => point[yKey]);
  const xMin = Math.min(...xs, 0);
  const xMax = Math.max(...xs, 1);
  const yMin = Math.min(...ys, 0);
  const yMax = Math.max(...ys, 1);

  const px = (value: number) => PAD + ((value - xMin) / (xMax - xMin || 1)) * (SIZE - PAD * 1.5);
  const py = (value: number) => SIZE - PAD - ((value - yMin) / (yMax - yMin || 1)) * (SIZE - PAD * 1.5);

  const labelA = columns.get(a)?.label ?? a;
  const labelB = columns.get(b)?.label ?? b;

  return (
    <section className={styles.panel} aria-label={`${labelA} against ${labelB}`}>
      <header className={styles.head}>
        <div>
          <h2 className={styles.title}>
            {labelA} <span className={styles.against}>against</span> {labelB}
          </h2>
          <p className={styles.caption}>
            <span className="data">{formatRho(rho)}</span> over{" "}
            <span className="data">{n}</span> player-seasons
            {position === "all" ? ", all positions pooled" : `, ${position} only`}.
          </p>
        </div>
        <label className={styles.toggle}>
          <input
            type="checkbox"
            checked={useRank}
            onChange={(event) => setUseRank(event.target.checked)}
          />
          Rank axes
        </label>
      </header>

      {points.length === 0 ? (
        <p className={styles.empty}>
          No {season} player carries both metrics. The matrix pools every archive
          season; this scatter shows {season} only, so a pair that exists in the
          matrix can be empty here.
        </p>
      ) : (
        <svg
          className={styles.plot}
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          role="img"
          aria-label={`Scatter of ${points.length} players, ${labelA} against ${labelB}`}
        >
          <line x1={PAD} y1={SIZE - PAD} x2={SIZE} y2={SIZE - PAD} className={styles.axis} />
          <line x1={PAD} y1={0} x2={PAD} y2={SIZE - PAD} className={styles.axis} />
          {points.map((point) => (
            <circle
              key={point.id}
              cx={px(point[xKey])}
              cy={py(point[yKey])}
              r={2.5}
              className={styles.point}
            >
              <title>
                {point.name} ({point.team}) — {labelA} {point.rawX.toFixed(2)}, {labelB}{" "}
                {point.rawY.toFixed(2)}
              </title>
            </circle>
          ))}
          <text x={SIZE / 2} y={SIZE - 6} className={styles.axisLabel} textAnchor="middle">
            {labelA} {useRank ? "rank" : "per 90"}
          </text>
          <text
            x={10}
            y={SIZE / 2}
            className={styles.axisLabel}
            textAnchor="middle"
            transform={`rotate(-90 10 ${SIZE / 2})`}
          >
            {labelB} {useRank ? "rank" : "per 90"}
          </text>
        </svg>
      )}
    </section>
  );
}
