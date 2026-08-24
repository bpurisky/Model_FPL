import type { SeasonSummary } from "../data/schema";
import styles from "./SeasonFilter.module.css";

export interface SeasonFilterProps {
  seasons: SeasonSummary[];
  selected: ReadonlySet<string>;
  onChange: (next: Set<string>) => void;
  busy: boolean;
}

/**
 * Which seasons the correlation runs over.
 *
 * This is §5.6.1's "arbitrary user-defined filter" made concrete, and it
 * cannot be served by precomputation: Spearman does not compose, so rho
 * over a subset is not assembled from per-season matrices, and shipping
 * one matrix per subset is 2^n.
 *
 * A season still being played is marked. A rho over two gameweeks and one
 * over thirty-eight are indistinguishable once they are both a rho, and
 * the current season joins this list the moment it has a recorded
 * gameweek — so the flag is what stops it reading as a peer of a finished
 * season.
 */
export function SeasonFilter({ seasons, selected, onChange, busy }: SeasonFilterProps) {
  const toggle = (season: string) => {
    const next = new Set(selected);
    if (next.has(season)) {
      // Never allow an empty selection: a correlation over no seasons is
      // not a view anyone wants, and an empty matrix would read as broken.
      if (next.size === 1) return;
      next.delete(season);
    } else {
      next.add(season);
    }
    onChange(next);
  };

  return (
    <fieldset className={styles.set} disabled={busy}>
      <legend className={styles.legend}>Seasons</legend>
      {seasons.map((season) => {
        const on = selected.has(season.season);
        return (
          <label
            key={season.season}
            className={`${styles.option} ${on ? styles.on : ""}`}
            title={
              season.partial
                ? `${season.gameweeks} of 38 gameweeks recorded — rates over a part-season are volatile`
                : `${season.gameweeks} gameweeks, ${season.players} eligible players`
            }
          >
            <input
              type="checkbox"
              checked={on}
              onChange={() => toggle(season.season)}
              className={styles.box}
            />
            <span className="data">{season.season}</span>
            {season.partial ? (
              <span className={styles.partial}>gw{season.gameweeks}</span>
            ) : (
              <span className={styles.count}>{season.players}</span>
            )}
          </label>
        );
      })}
    </fieldset>
  );
}
