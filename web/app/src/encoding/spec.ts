/**
 * The Graph Builder's encoding state (§5.4.2).
 *
 * Four channels, no more. "Four zones cover the overwhelming majority of
 * real questions and each additional zone multiplies the mark-inference
 * matrix" — so Overlay, Size, Shape, Group X/Y and Page are absent by
 * decision rather than by omission, and adding one is a spec change.
 *
 * This module holds only the *state* and its URL form. The state is
 * deliberately dumb: four column keys, a reduction, and a normalization
 * flag. Everything about what to draw is derived in `mark.ts`, so there
 * is exactly one place where the §5.4.2 table lives.
 */

import type { ColumnSpec } from "../data/schema";

/** The four drop zones. Ordered as they read on screen. */
export const CHANNELS = ["x", "y", "color", "wrap"] as const;
export type Channel = (typeof CHANNELS)[number];

/**
 * §5.6.2's closed set, and nothing else may be added here. Each is exact,
 * order-independent, and carries no estimator choice — which is the whole
 * reason the browser is allowed to compute them at all.
 *
 * `quantile` is absent from this list on purpose: it needs a parameter,
 * and a parameterised reduction in a drop-zone UI is a control the spec
 * does not describe. It ships in `query/reduce.ts` for callers that name
 * the quantile, and the golden tests cover it there.
 */
export const AGGREGATES = ["mean", "median", "sum", "count", "min", "max"] as const;
export type Aggregate = (typeof AGGREGATES)[number];

export interface Encoding {
  x: string | null;
  y: string | null;
  color: string | null;
  wrap: string | null;
  aggregate: Aggregate;
}

export const EMPTY_ENCODING: Encoding = {
  x: null,
  y: null,
  color: null,
  wrap: null,
  aggregate: "mean",
};

/**
 * The role a channel is carrying, or null when the zone is empty.
 *
 * Roles come from `columns.json` and never from inspecting values. A
 * column's role is a property of the data the pipeline exported, not
 * something the browser should guess from what happens to be in it —
 * `gw` holding only the integer 1 in an opening week does not make it
 * quantitative.
 */
export type ChannelRole = ColumnSpec["role"] | null;

export interface Roles {
  x: ChannelRole;
  y: ChannelRole;
  color: ChannelRole;
  wrap: ChannelRole;
}

export function rolesOf(
  encoding: Encoding,
  columns: ReadonlyMap<string, ColumnSpec>,
): Roles {
  const role = (key: string | null): ChannelRole =>
    key === null ? null : (columns.get(key)?.role ?? null);
  return {
    x: role(encoding.x),
    y: role(encoding.y),
    color: role(encoding.color),
    wrap: role(encoding.wrap),
  };
}

export function isAggregate(value: string): value is Aggregate {
  return (AGGREGATES as readonly string[]).includes(value);
}
