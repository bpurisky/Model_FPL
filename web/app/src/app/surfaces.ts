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

/**
 * The five places a reader goes, in the order they go to them. The eleven
 * surfaces are too many for one row of tabs, and on a phone they wrapped
 * into four rows that filled half the screen. Grouped, the header carries
 * five sections and the section's own pages sit in a strip below it; on a
 * phone the five become the bottom tab bar.
 */
export type SectionId = "model" | "players" | "fixtures" | "team" | "lab";

export interface Section {
  id: SectionId;
  label: string;
}

export const SECTIONS: Section[] = [
  { id: "model", label: "Model" },
  { id: "players", label: "Players" },
  { id: "fixtures", label: "Fixtures" },
  { id: "team", label: "My Team" },
  { id: "lab", label: "Lab" },
];

export interface Surface {
  view: View;
  section: SectionId;
  label: string;
  /** §5.8.7 register: what the surface is, in one line, no promises. */
  blurb: string;
  status: SurfaceStatus;
  milestone: string;
}

export const SURFACES: Surface[] = [
  {
    view: "board",
    section: "model",
    label: "Model Board",
    blurb: "The model's own ranking within position, and what each bucket was worth.",
    status: "live",
    milestone: "5E",
  },
  {
    view: "scorecard",
    section: "model",
    label: "Scorecard",
    blurb: "How the model scored in the walk-forward backtest, and where it was wrong.",
    status: "live",
    milestone: "5F",
  },
  {
    view: "papertrade",
    section: "model",
    label: "Paper Trade",
    blurb: "What the frozen shadow team actually scored, gameweek by gameweek, against the five criteria that gate a real launch.",
    status: "live",
    milestone: "5G",
  },
  {
    view: "explorer",
    section: "players",
    label: "Explorer",
    blurb: "Every player, every exported column, sortable.",
    status: "live",
    milestone: "5F",
  },
  {
    view: "compare",
    section: "players",
    label: "Comparison",
    blurb: "Two or more players, decomposed into the components behind the total.",
    status: "live",
    milestone: "5D",
  },
  {
    view: "form",
    section: "players",
    label: "Form Matrix",
    blurb: "Player by gameweek, as a heat map. Where a slump becomes visible.",
    status: "live",
    milestone: "5D",
  },
  {
    view: "trend",
    section: "players",
    label: "Trend Explorer",
    blurb: "Price and ownership over the collector's snapshot history.",
    status: "live",
    milestone: "5F",
  },
  {
    view: "fixtures",
    section: "fixtures",
    label: "Fixtures",
    blurb: "The season ahead: every club by gameweek, coloured by difficulty.",
    status: "live",
    milestone: "5F",
  },
  {
    view: "optimizer",
    section: "team",
    label: "Squad Optimizer",
    blurb: "Given a team ID, solves for the best legal transfer under budget and formation constraints — the only surface that calls a live backend.",
    status: "live",
    milestone: "5H",
  },
  {
    view: "correlations",
    section: "lab",
    label: "Correlation Lab",
    blurb: "Within-position Spearman across every exported metric.",
    status: "live",
    milestone: "5B",
  },
  {
    view: "graph",
    section: "lab",
    label: "Graph Builder",
    blurb: "Your own question: four channels over the player-gameweek panel.",
    status: "live",
    milestone: "5C",
  },
];

export const SURFACE_BY_VIEW = new Map(SURFACES.map((surface) => [surface.view, surface]));

export function surfacesIn(section: SectionId): Surface[] {
  return SURFACES.filter((surface) => surface.section === section);
}
