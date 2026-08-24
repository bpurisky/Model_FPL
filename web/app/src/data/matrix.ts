import type { CorrelationCell, ObservationsFile } from "./schema";
import { spearman } from "./spearman";

/**
 * A correlation matrix over a chosen subset of seasons (§5.6.1).
 *
 * This is the exception §5.6 permits and nothing more: it computes rho
 * and n, and it does **not** compute a p-value, because significance
 * testing in the browser is forbidden without exception. A cell from here
 * carries `p_value: null`, and the surface says so — showing the
 * precomputed p beside a rho from a different population would be worse
 * than showing none.
 *
 * The population matches `correlations.py`'s exactly: the same eligible
 * player-seasons, the same minutes-weighted rates. That is what lets the
 * app hand back to the precomputed matrix when the selection is every
 * season, and get the same numbers.
 */
export function correlateSelection(
  observations: ObservationsFile,
  seasons: ReadonlySet<string>,
  group: string,
): CorrelationCell[] {
  const rows = observations.rows.filter(
    (row) => seasons.has(row.season) && (group === "all" || row.position === group),
  );

  const columns = observations.metrics.map((_, index) =>
    rows.map((row) => row.values[index] ?? null),
  );

  const cells: CorrelationCell[] = [];
  for (let a = 0; a < observations.metrics.length; a += 1) {
    for (let b = a + 1; b < observations.metrics.length; b += 1) {
      const { rho, n } = spearman(columns[a]!, columns[b]!);
      cells.push({
        group,
        a: observations.metrics[a]!,
        b: observations.metrics[b]!,
        rho,
        n,
        // §5.6: no significance testing client-side.
        p_value: null,
      });
    }
  }
  return cells;
}

/** Whether a selection covers every season the export knows about. */
export function isEverySeason(
  observationSeasons: readonly { season: string }[],
  selected: ReadonlySet<string>,
): boolean {
  return (
    observationSeasons.length === selected.size &&
    observationSeasons.every((entry) => selected.has(entry.season))
  );
}

/**
 * Population per position group for a season selection.
 *
 * The position filter shows each group's size, and that number has to
 * follow the seasons actually selected. Left reading the precomputed
 * pooled count it would say 284 midfielders beside a matrix computed over
 * 194 — the kind of small disagreement that quietly teaches a reader the
 * numbers on this surface are decorative.
 */
export function groupSizes(
  observations: ObservationsFile,
  seasons: ReadonlySet<string>,
): Map<string, number> {
  const sizes = new Map<string, number>([["all", 0]]);
  for (const row of observations.rows) {
    if (!seasons.has(row.season)) continue;
    sizes.set("all", (sizes.get("all") ?? 0) + 1);
    sizes.set(row.position, (sizes.get(row.position) ?? 0) + 1);
  }
  return sizes;
}
