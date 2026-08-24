/**
 * Mark inference (§5.4.2) — the single decision that lets the Graph
 * Builder work without a chart-type menu.
 *
 * > "Mark inference is automatic and not user-selectable. The user
 * > chooses data, not chart type."
 *
 * The §5.4.2 table is transcribed here row for row and nowhere else. A
 * combination the table does not name produces **no mark and a reason**,
 * rather than a nearest-guess chart: guessing is how a builder ends up
 * drawing a bar of averaged gameweek numbers that the user reads as a
 * total. §5.11.3 requires every role combination to be unit-tested,
 * including the ones that produce nothing, and `mark.test.ts` walks the
 * whole cross product.
 *
 * Two places where this file is narrower than the table, both recorded
 * in the README under Phase 5 deviations (§5.16):
 *
 *   D6. The table's `temporal | quantitative | any` row is implemented as
 *       "any *categorical* colour". A quantitative colour cannot split a
 *       line into series, and the alternatives — binning it, or running a
 *       gradient along the path — are an estimator choice and a
 *       decoration respectively. The reason string names the fix.
 *
 *   D7. Rows the table writes with `—` in the Colour column are treated
 *       as requiring that zone to be *empty*. The alternative is to
 *       silently ignore an assignment the user made, and a drop zone that
 *       accepts a column and then does nothing with it is worse than one
 *       that says why it cannot.
 */

import type { Channel, Roles } from "./spec";

export type Mark = "point" | "histogram" | "bar" | "line" | "rect";

export interface MarkPlan {
  mark: Mark;
  /**
   * The group key for the query. `"player"` means one row per element —
   * the panel is at player-gameweek grain (§5.3.2), so a scatter of two
   * rate metrics is only meaningful once the gameweeks inside the filter
   * window are reduced to one row per player.
   */
  groupBy: ("player" | Channel)[];
  /** Channels reduced by the encoding's aggregate. */
  reduced: Channel[];
  /** Whether `color` splits the mark into one series per category. */
  series: boolean;
}

export type Inference =
  | { readonly ok: true; readonly plan: MarkPlan }
  | { readonly ok: false; readonly reason: string };

const no = (reason: string): Inference => ({ ok: false, reason });
const yes = (plan: MarkPlan): Inference => ({ ok: true, plan });

/**
 * `wrap` facets; it never changes which mark is drawn. It has to be
 * something with a finite set of panels, which rules out a raw
 * quantitative column — one panel per distinct xG value is not a chart.
 */
function wrapIsUsable(role: Roles["wrap"]): boolean {
  return role === null || role === "categorical" || role === "ordinal";
}

export function inferMark(roles: Roles): Inference {
  if (!wrapIsUsable(roles.wrap)) {
    return no(
      "Wrap needs a column with a finite set of values — a category or the gameweek. " +
        "A number would give one panel per distinct value.",
    );
  }

  const { x, y, color } = roles;

  if (x === null) {
    return no("Assign a column to X. Every mark needs one.");
  }

  // quantitative | quantitative | any -> point
  if (x === "quantitative" && y === "quantitative") {
    return yes({
      mark: "point",
      groupBy: ["player"],
      reduced: color === "quantitative" ? ["x", "y", "color"] : ["x", "y"],
      series: color === "categorical",
    });
  }

  // quantitative | - | - -> histogram
  if (x === "quantitative" && y === null) {
    if (color !== null) {
      return no(
        "A histogram counts one column; it has nothing for Colour to encode. " +
          "Add a column to Y for a scatter, or clear Colour.",
      );
    }
    return yes({ mark: "histogram", groupBy: ["player"], reduced: ["x"], series: false });
  }

  // categorical | quantitative | - -> bar (aggregated)
  if (x === "categorical" && y === "quantitative") {
    if (color !== null) {
      return no(
        "A bar already encodes its value by length. Move the Colour column to Wrap " +
          "to compare across panels, or clear it.",
      );
    }
    return yes({ mark: "bar", groupBy: ["x"], reduced: ["y"], series: false });
  }

  // ordinal | quantitative | (- | categorical) -> line
  if (x === "ordinal" && y === "quantitative") {
    if (color === null) {
      return yes({ mark: "line", groupBy: ["x"], reduced: ["y"], series: false });
    }
    if (color === "categorical") {
      return yes({ mark: "line", groupBy: ["x", "color"], reduced: ["y"], series: true });
    }
    return no(
      "A line splits into one series per category. Colour is carrying a number — " +
        "move it to Y, or drop it.",
    );
  }

  // temporal | quantitative | any -> line   (see D6 above)
  if (x === "temporal" && y === "quantitative") {
    if (color === null) {
      return yes({ mark: "line", groupBy: ["x"], reduced: ["y"], series: false });
    }
    if (color === "categorical") {
      return yes({ mark: "line", groupBy: ["x", "color"], reduced: ["y"], series: true });
    }
    return no(
      "A line splits into one series per category. Colour is carrying a number — " +
        "move it to Y, or drop it.",
    );
  }

  // (categorical | ordinal) | categorical | quantitative -> rect (heat map)
  if ((x === "categorical" || x === "ordinal") && y === "categorical") {
    if (color === "quantitative") {
      return yes({ mark: "rect", groupBy: ["x", "y"], reduced: ["color"], series: false });
    }
    return no(
      "A heat map of two categories needs a number in Colour to fill the cells.",
    );
  }

  return no(unmatchedReason(roles));
}

/**
 * The message for a combination the table does not name. Written to say
 * what to change rather than what is wrong — §5.8.7: "errors state what
 * failed and what to do".
 */
function unmatchedReason(roles: Roles): string {
  const { x, y } = roles;

  if (y === null) {
    if (x === "categorical") {
      return "A category on X needs a number on Y to have a height. Add one.";
    }
    return "Add a number to Y. On its own, the gameweek is an axis with nothing on it.";
  }

  if (y === "quantitative") {
    // x is categorical/ordinal/temporal are all handled above, so this is
    // the quantitative-y-with-an-unusable-x remainder.
    return "Move the number on X to Y, or put a category or the gameweek on X.";
  }

  if (y === "ordinal" || y === "temporal") {
    return "Gameweek and kickoff time belong on X, not Y. Swap them.";
  }

  // y categorical, x quantitative or temporal
  return "A category on Y needs a category or the gameweek on X to form a heat map grid.";
}

/**
 * Whether the aggregate control is shown (§5.4.2: "appears only when the
 * mark requires it"). Every mark this builder draws reduces the panel
 * from player-gameweek grain, so the honest answer is "always, once
 * there is a mark" — but the control names which channel it applies to,
 * which is the part that stops it being a mystery dropdown.
 */
export function reducedLabel(plan: MarkPlan): string {
  if (plan.mark === "histogram") return "count of players by";
  return plan.reduced.join(" and ");
}
