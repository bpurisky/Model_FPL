import { legendStops } from "../design/scale";
import styles from "./Legend.module.css";

export interface LegendProps {
  minN: number;
  hatchedCount: number;
}

/**
 * §5.8.2's v2 extension requires the legend to state the scale's
 * direction explicitly, and §5.10 requires the low-n encoding to be
 * legible as pattern rather than hue. Both are stated here once, so the
 * matrix itself does not have to repeat them 240 times.
 */
export function Legend({ minN, hatchedCount }: LegendProps) {
  return (
    <div className={styles.legend}>
      <span className={styles.ends}>
        <span className="data">−1</span>
        <span className={styles.ramp} aria-hidden="true">
          {legendStops().map((stop) => (
            <span
              key={stop.value}
              className={styles.stop}
              style={{ background: stop.color }}
            />
          ))}
        </span>
        <span className="data">+1</span>
      </span>
      <span>Spearman rho. Poles swap where low is better.</span>
      {hatchedCount > 0 && (
        <span className={styles.hatchKey}>
          <span className={styles.hatchSwatch} aria-hidden="true" />
          {hatchedCount} cells below n={minN} — hatched, not coloured
        </span>
      )}
    </div>
  );
}
