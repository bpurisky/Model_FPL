/**
 * The public data files, in one place (the Data & API page reads this).
 *
 * Every file the app fetches is an ordinary static file under
 * `${DATA_BASE}/` on the Pages site, so the app's own data layer *is* the
 * public API: there is no separate endpoint to keep in step, and nothing
 * here describes a file that is not also what the app reads.
 *
 * Kept as data rather than prose in the view so `catalog.test.ts` can
 * check it against what `load.ts` fetches and against `data/web/v1/` on
 * disk. A new export that nobody added here fails that test, which is the
 * only way a documentation list stays true.
 *
 * Cadences are read from `.github/workflows/`: `web.yml` (every six
 * hours, `web.export all`), `papertrade.yml` (every two hours) and the
 * scorecard, which `web.yml` rebuilds only on a push or manual run.
 * "Rewritten only when its numbers change" is `web.export`'s
 * `write_json`: a file's `generated_at` is when it last changed, not when
 * it last ran.
 */

import {
  loadBoard,
  loadColumns,
  loadCorrelations,
  loadFixtures,
  loadObservations,
  loadPaperTrade,
  loadPlayers,
  loadOwnership,
  loadPrices,
  loadProjections,
  loadScorecard,
  loadShrinkage,
  type LoadProgress,
} from "./load";
import type { Header } from "./schema";

export type Cadence = "every_6h" | "every_2h" | "on_code_change" | "manual";

export const CADENCE_TEXT: Record<Cadence, string> = {
  every_6h: "Every 6 hours (web.yml), rewritten only when its numbers change",
  every_2h: "Every 2 hours (papertrade.yml), rewritten only when its numbers change",
  on_code_change:
    "Only when the model, backtest or archive changes (web.yml push or manual run), not on the 6-hourly schedule",
  manual: "Rebuilt by hand when the model changes; not on any schedule",
};

export interface DataFile {
  /** File name under the data directory. */
  name: string;
  format: "json" | "parquet";
  /** One line: what is in it. */
  description: string;
  cadence: Cadence;
  /**
   * Reads the file through the app's own loader (shared cache, zod
   * validation) so its header can be shown. JSON only: the parquet files
   * are read by the query engine, and their header sits in the file
   * footer.
   */
  loadHeader?: (onProgress?: (p: LoadProgress) => void) => Promise<{ header: Header }>;
}

export const DATA_FILES: readonly DataFile[] = [
  {
    name: "players.json",
    format: "json",
    description:
      "One row per player: identity, price, ownership, season-to-date actuals, rate metrics with within-position z-scores and percentiles, set-piece takers, and the next-gameweek projection split by component.",
    cadence: "every_6h",
    loadHeader: loadPlayers,
  },
  {
    name: "board.json",
    format: "json",
    description: "The model's ranking within position, with the bucket each player falls in.",
    cadence: "every_6h",
    loadHeader: loadBoard,
  },
  {
    name: "projections.json",
    format: "json",
    description:
      "Expected points for every player over the next six gameweeks, each club's fixtures for those gameweeks, and the minutes split (blank / under 60 / 60+) for the first of them.",
    cadence: "every_6h",
    loadHeader: loadProjections,
  },
  {
    name: "prices.json",
    format: "json",
    description:
      "The current price cycle: each player's 24h net transfers against the fitted rise threshold, eligibility and the rise-only call, with the rule's out-of-sample track record.",
    cadence: "every_6h",
    loadHeader: loadPrices,
  },
  {
    name: "ownership.json",
    format: "json",
    description:
      "Elite ownership: for the latest gameweek with collected picks, each player's ownership among the top managers, starting and captain share, effective ownership, and the gap to the whole field.",
    cadence: "every_6h",
    loadHeader: loadOwnership,
  },
  {
    name: "fixtures.json",
    format: "json",
    description: "The season's schedule per club, with FPL's difficulty and our Elo-based difficulty.",
    cadence: "every_6h",
    loadHeader: loadFixtures,
  },
  {
    name: "columns.json",
    format: "json",
    description:
      "The column registry: label, definition, unit, direction and position relevance for every exported column.",
    cadence: "every_6h",
    loadHeader: loadColumns,
  },
  {
    name: "correlations.json",
    format: "json",
    description: "Within-position Spearman correlations between every pair of exported metrics, per season.",
    cadence: "every_6h",
    loadHeader: loadCorrelations,
  },
  {
    name: "observations.json",
    format: "json",
    description: "The eligible player-seasons behind the correlations: one row per player, metric values positional.",
    cadence: "every_6h",
    loadHeader: loadObservations,
  },
  {
    name: "scorecard.json",
    format: "json",
    description: "The walk-forward backtest: accuracy by component and bucket, against the baselines.",
    cadence: "on_code_change",
    loadHeader: loadScorecard,
  },
  {
    name: "shrinkage.json",
    format: "json",
    description: "The goals-conceded plateau that sets the model's shrinkage.",
    cadence: "manual",
    loadHeader: loadShrinkage,
  },
  {
    name: "papertrade.json",
    format: "json",
    description:
      "The frozen shadow team's results gameweek by gameweek, and the status of the launch-gate criteria.",
    cadence: "every_2h",
    loadHeader: loadPaperTrade,
  },
  {
    name: "panel.parquet",
    format: "parquet",
    description:
      "The player-gameweek panel behind the Graph Builder and Form Matrix: every exported column, one row per player per gameweek.",
    cadence: "every_6h",
  },
  {
    name: "timeseries.parquet",
    format: "parquet",
    description: "Price and ownership over the collector's hourly snapshot history, behind the Trend Explorer.",
    cadence: "every_6h",
  },
];
