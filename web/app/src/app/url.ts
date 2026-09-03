/**
 * URL as state (§5.5).
 *
 * > "Every view's selection encodes into the query string, including the
 * > full Graph Builder encoding spec, so a finding can be linked. Cheapest
 * > possible collaboration feature; costs nothing if done from the start."
 *
 * Which is why it is here at 5C rather than retrofitted at 5F. The
 * retrofit is what costs: every piece of state added in between acquires
 * a second home, and the two drift.
 *
 * The encoding is flat, short, and readable — `?view=graph&x=xg_per90`
 * rather than a base64 blob. A linked finding should be legible in the
 * link, and a reader should be able to edit `pos=MID` to `pos=FWD` in the
 * address bar and get what they expect. That rules out compressing it,
 * which would buy characters nobody is short of.
 *
 * Everything here round-trips, and `url.test.ts` asserts it does over
 * generated states. A parse that silently drops an unknown key is
 * correct: a URL from a newer build must degrade to a usable view rather
 * than to an error, and a URL from an older one must not resurrect a
 * channel that no longer exists.
 */

import { EMPTY_ENCODING, isAggregate, type Encoding } from "../encoding/spec";
import { NO_FILTERS, type PanelFilters } from "../query/panel";

/** Every surface, including the ones §5.1.3 stubs. */
export const VIEWS = [
  "correlations",
  "fixtures",
  "graph",
  "form",
  "compare",
  "explorer",
  "board",
  "scorecard",
  "trend",
  "optimizer",
  "papertrade",
] as const;
export type View = (typeof VIEWS)[number];

export const DEFAULT_VIEW: View = "correlations";

export interface AppUrlState {
  view: View;
  /** App-level per §5.5.3, carried across every surface. */
  position: string;
  normalized: boolean;
  filters: PanelFilters;
  /** Element ids, for cross-view player selection (§5.5.3). */
  selection: number[];
  /** View-level per §5.5.3, but linkable, so it travels in the URL too. */
  encoding: Encoding;
  /**
   * The Squad Optimizer's team ID (view-level, but §5.5 wants a run
   * linkable the same as any other finding — "?view=optimizer&entry=123"
   * should reproduce the request, not just the empty form).
   */
  entry: number | null;
}

export const DEFAULT_STATE: AppUrlState = {
  view: DEFAULT_VIEW,
  position: "all",
  normalized: false,
  filters: NO_FILTERS,
  selection: [],
  encoding: EMPTY_ENCODING,
  entry: null,
};

function isView(value: string): value is View {
  return (VIEWS as readonly string[]).includes(value);
}

/** `a,b,c` -> `["a","b","c"]`, with blanks dropped. */
function csv(value: string | null): string[] {
  if (!value) return [];
  return value
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
}

function numbers(value: string | null): number[] {
  return csv(value)
    .map(Number)
    .filter((entry) => Number.isFinite(entry));
}

/** `10-38` -> `[10, 38]`. Either side may be blank: `-38`, `10-`. */
function range(value: string | null): [number | null, number | null] {
  if (!value) return [null, null];
  const [lo, hi] = value.split("-", 2);
  const parse = (part: string | undefined) => {
    if (part === undefined || part.trim() === "") return null;
    const parsed = Number(part);
    return Number.isFinite(parsed) ? parsed : null;
  };
  return [parse(lo), parse(hi)];
}

function encodeRange(lo: number | null, hi: number | null): string | null {
  if (lo === null && hi === null) return null;
  return `${lo ?? ""}-${hi ?? ""}`;
}

export function parseUrl(search: string): AppUrlState {
  const params = new URLSearchParams(search);

  const rawView = params.get("view") ?? "";
  const view = isView(rawView) ? rawView : DEFAULT_VIEW;

  const [gwMin, gwMax] = range(params.get("gw"));
  const [priceMin, priceMax] = range(params.get("price"));
  const minutesRaw = params.get("mins");
  const minutesFloor = minutesRaw === null || minutesRaw === "" ? null : Number(minutesRaw);

  const aggregate = params.get("agg") ?? "";

  const entryRaw = params.get("entry");
  const entryParsed = entryRaw === null ? null : Number(entryRaw);
  const entry = entryParsed !== null && Number.isFinite(entryParsed) ? entryParsed : null;

  return {
    view,
    position: params.get("pos") ?? DEFAULT_STATE.position,
    normalized: params.get("norm") === "1",
    filters: {
      seasons: csv(params.get("seasons")),
      positions: csv(params.get("positions")),
      teams: csv(params.get("teams")),
      elements: numbers(params.get("el")),
      priceMin,
      priceMax,
      minutesFloor: minutesFloor !== null && Number.isFinite(minutesFloor) ? minutesFloor : null,
      gwMin,
      gwMax,
    },
    selection: numbers(params.get("sel")),
    encoding: {
      x: params.get("x") || null,
      y: params.get("y") || null,
      color: params.get("color") || null,
      wrap: params.get("wrap") || null,
      aggregate: isAggregate(aggregate) ? aggregate : EMPTY_ENCODING.aggregate,
    },
    entry,
  };
}

/**
 * State back to a query string, omitting everything at its default.
 *
 * The omission is what keeps a link readable: a Correlation Lab URL with
 * nothing selected should be `?view=correlations`, not forty parameters
 * spelling out that nothing is set. It also makes the round-trip test
 * meaningful — two states that render identically must encode
 * identically, or the browser history fills with false steps.
 */
export function toSearch(state: AppUrlState): string {
  const params = new URLSearchParams();
  const set = (key: string, value: string | null | undefined) => {
    if (value !== null && value !== undefined && value !== "") params.set(key, value);
  };

  if (state.view !== DEFAULT_VIEW) set("view", state.view);
  if (state.position !== DEFAULT_STATE.position) set("pos", state.position);
  if (state.normalized) set("norm", "1");

  const { filters } = state;
  if (filters.seasons.length) set("seasons", filters.seasons.join(","));
  if (filters.positions.length) set("positions", filters.positions.join(","));
  if (filters.teams.length) set("teams", filters.teams.join(","));
  if (filters.elements.length) set("el", filters.elements.join(","));
  set("price", encodeRange(filters.priceMin, filters.priceMax));
  set("gw", encodeRange(filters.gwMin, filters.gwMax));
  if (filters.minutesFloor !== null) set("mins", String(filters.minutesFloor));

  if (state.selection.length) set("sel", state.selection.join(","));

  const { encoding } = state;
  set("x", encoding.x);
  set("y", encoding.y);
  set("color", encoding.color);
  set("wrap", encoding.wrap);
  // The aggregate only means something once a channel is filled; writing
  // it into an empty builder's URL is noise.
  const anyChannel = encoding.x ?? encoding.y ?? encoding.color ?? encoding.wrap;
  if (anyChannel && encoding.aggregate !== EMPTY_ENCODING.aggregate) {
    set("agg", encoding.aggregate);
  }

  if (state.entry !== null) set("entry", String(state.entry));

  const search = params.toString();
  return search ? `?${search}` : "";
}

/** The href for a state, for anchors that must be real links. */
export function hrefFor(state: AppUrlState): string {
  return `${window.location.pathname}${toSearch(state)}`;
}
