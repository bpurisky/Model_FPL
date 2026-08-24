import type { SeasonSummary } from "../data/schema";
import { count } from "../design/text";
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
                ? season.gameweeks > 0
                  ? `${count(season.gameweeks, "gameweek")} of 38 recorded — rates over a part-season are volatile`
                  : "Part-season. How much of it has been recorded is not known until the values behind the matrix load."
                : season.players > 0
                  ? `${count(season.gameweeks, "gameweek")}, ${count(season.players, "eligible player")}`
                  : `${count(season.gameweeks, "gameweek")}. The eligible count loads with the values behind the matrix.`
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
              // A zero here means "not loaded yet", not "no gameweeks",
              // and printing `gw0` would be the wrong claim either way.
              <span className={styles.partial}>
                {season.gameweeks > 0 ? `gw${season.gameweeks}` : "partial"}
              </span>
            ) : (
              // A zero is "not loaded yet" in the placeholder state, and
              // printing it would claim no player qualified.
              <span className={styles.count}>{season.players > 0 ? season.players : "—"}</span>
            )}
          </label>
        );
      })}
    </fieldset>
  );
}
