/**
 * The navigation, and what each entry honestly is right now.
 *
 * §5.1.3 is explicit about the two unbuilt *phases*: "Phase 3 and 4
 * surfaces appear in navigation as **disabled entries with a one-line
 * explanation of what will live there**, not as hidden routes and not as
 * fake data. An empty state that explains the phase boundary is honest;
 * a mocked squad is a lie that will survive into screenshots."
 *
 * The same argument applies with equal force to the Phase 5 surfaces
 * that later milestones own. A nav that hides them misrepresents the
 * shape of the tool; a nav that links them to a blank page misrepresents
 * its progress. So every entry carries a `status`, and the shell renders
 * from that rather than from a list of routes it happens to have.
 *
 * `milestone` is not decoration either — it is what lets the empty state
 * say "5D" instead of "coming soon", which is the difference between a
 * roadmap and a marketing promise.
 */

import type { View } from "./url";

export type SurfaceStatus =
  /** Built and rendering real data. */
  | "live"
  /** A Phase 5 surface a later milestone owns (§5.13). */
  | "planned"
  /** A Phase 3/4 surface, outside this phase entirely (§5.1.3). */
  | "out_of_phase";

export interface Surface {
  view: View;
  label: string;
  /** §5.8.7 register: what the surface is, in one line, no promises. */
  blurb: string;
  status: SurfaceStatus;
  milestone: string;
}

export const SURFACES: Surface[] = [
  {
    view: "correlations",
    label: "Correlation Lab",
    blurb: "Within-position Spearman across every exported metric.",
    status: "live",
    milestone: "5B",
  },
  {
    view: "graph",
    label: "Graph Builder",
    blurb: "Your own question: four channels over the player-gameweek panel.",
    status: "live",
    milestone: "5C",
  },
  {
    view: "form",
    label: "Form Matrix",
    blurb: "Player by gameweek, as a heat map. Where a slump becomes visible.",
    status: "planned",
    milestone: "5D",
  },
  {
    view: "compare",
    label: "Comparison",
    blurb: "Two or more players, decomposed into the components behind the total.",
    status: "planned",
    milestone: "5D",
  },
  {
    view: "board",
    label: "Model Board",
    blurb: "The model's own ranking within position, and what each bucket was worth.",
    status: "planned",
    milestone: "5E",
  },
  {
    view: "explorer",
    label: "Explorer",
    blurb: "Every player, every exported column, sortable.",
    status: "planned",
    milestone: "5F",
  },
  {
    view: "scorecard",
    label: "Scorecard",
    blurb: "How the model scored in the walk-forward backtest, and where it was wrong.",
    status: "planned",
    milestone: "5F",
  },
  {
    view: "trend",
    label: "Trend Explorer",
    blurb: "Price and ownership over the collector's snapshot history.",
    status: "planned",
    milestone: "5F",
  },
  {
    view: "optimizer",
    label: "Squad Optimizer",
    blurb: "Phase 3 solves for a 15-man squad under budget and formation constraints.",
    status: "out_of_phase",
    milestone: "Phase 3",
  },
  {
    view: "papertrade",
    label: "Paper Trade",
    blurb: "Phase 4 runs a frozen squad forward each gameweek and records what it scored.",
    status: "out_of_phase",
    milestone: "Phase 4",
  },
];

export const SURFACE_BY_VIEW = new Map(SURFACES.map((surface) => [surface.view, surface]));
