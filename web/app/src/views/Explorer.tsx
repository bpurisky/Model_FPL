/**
 * §5.4.5 — the Player Explorer.
 *
 * > "Deliberately plain. It is a workbench; the visual budget is spent
 * > elsewhere (§5.8.5)."
 *
 * Sortable, filterable table over every element, with the column groups
 * §5.4.5 names: identity, price and ownership, projection and its
 * components, trailing actuals, and the position-normalized companions.
 *
 * **Virtualized**, because 841 players times two dozen columns is 20,000
 * cells and the browser should not be asked to lay them all out. Only the
 * rows inside the scroll window plus a small overscan are mounted.
 * Written by hand rather than pulled in from TanStack: §5.2 permits the
 * library and what it buys is sorting, filtering and virtualization over
 * a headless table — the first two are a `sort` and a `filter` here, and
 * the third is thirty lines. The dependency would be larger than the code
 * it replaced.
 *
 * **§5.7.5's caution is the point of the surface.** §5.4.5: "Sorting a
 * mixed-position table by raw xG is permitted but shows a persistent
 * inline caution, because that sort is exactly the mistake the tool
 * exists to teach the user out of." Permitted, and never silent.
 *
 * The normalization toggle is at *column-group* level per §5.4.5's v2
 * addition, and it does not disturb sort or filter state — flipping it
 * re-reads the same rows through the companion columns the export
 * already carries. Nothing is standardized here (§5.6).
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useApp } from "../app/state";
import { Provenance } from "../components/Provenance";
import { loadColumns, loadPlayers, type LoadProgress } from "../data/load";
import { byTeam, nextFixtures, runDifficulty, useFixtures, type TeamFixture } from "../data/fixtures";
import type { ColumnsFile, PlayerRow, PlayersFile } from "../data/schema";
import styles from "./Explorer.module.css";

type State =
  | { status: "loading"; progress: LoadProgress | null }
  | { status: "error"; error: Error }
  | { status: "ready"; players: PlayersFile; columns: ColumnsFile };

const ROW_HEIGHT = 26;
const OVERSCAN = 8;

/** What a cell reads from. Kept narrow so `valueOf` stays total. */
type Source =
  | { kind: "identity"; key: "name" | "team" | "position" }
  | { kind: "price" }
  | { kind: "ownership" }
  | { kind: "gameweeks" }
  | { kind: "projection" }
  | { kind: "component"; key: string }
  | { kind: "actual"; key: string }
  | { kind: "metric"; key: string }
  // §5.4.5's "next-N fixture difficulty" group. Read from the schedule
  // rather than from the player row, because a fixture is a property of
  // the club and the panel carries no forward-looking column at all.
  | { kind: "run"; count: number }
  | { kind: "fixture"; index: number };

interface Column {
  id: string;
  label: string;
  group: string;
  source: Source;
  /** Numeric columns sort descending first; text ascending. */
  numeric: boolean;
  format: (value: number) => string;
}

export function Explorer() {
  const { state, dispatch } = useApp();
  const [data, setData] = useState<State>({ status: "loading", progress: null });
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ id: string; desc: boolean }>({
    id: "projection",
    desc: true,
  });
  /** §5.4.5's group-level toggle: read metrics as percentiles, or raw. */
  const [metricsNormalized, setMetricsNormalized] = useState(false);
  const [cautionDismissed, setCautionDismissed] = useState(false);
  const [scrollTop, setScrollTop] = useState(0);
  /*
   * The schedule, for §5.4.5's fixture columns. `byTeam` returns nothing
   * unless the season matches the one the file describes, so an archive
   * season gets no fixture annotations rather than a schedule that
   * belongs to a different year.
   */
  const fixtures = useFixtures();
  const viewport = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [players, columns] = await Promise.all([
          loadPlayers((progress) =>
            cancelled ? undefined : setData({ status: "loading", progress }),
          ),
          loadColumns(),
        ]);
        if (!cancelled) setData({ status: "ready", players, columns });
      } catch (error) {
        if (!cancelled) setData({ status: "error", error: error as Error });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const schedule = useMemo(
    () => byTeam(fixtures, data.status === "ready" ? data.players.season : null),
    [fixtures, data],
  );

  const columns = useMemo<Column[]>(() => {
    if (data.status !== "ready") return [];
    return buildColumns(data.players, data.columns, metricsNormalized, schedule.size > 0);
  }, [data, metricsNormalized, schedule]);

  const rows = useMemo(() => {
    if (data.status !== "ready") return [];
    const needle = query.trim().toLowerCase();
    const positions = new Set(state.filters.positions);

    const filtered = data.players.players.filter((player) => {
      if (positions.size > 0 && !positions.has(player.position)) return false;
      if (needle === "") return true;
      return (
        player.name.toLowerCase().includes(needle) ||
        player.team.toLowerCase().includes(needle)
      );
    });

    const column = columns.find((entry) => entry.id === sort.id);
    if (!column) return filtered;

    return [...filtered].sort((left, right) => {
      const a = valueOf(left, column.source, schedule);
      const b = valueOf(right, column.source, schedule);
      /*
       * §5.3.3: a null sorts to the bottom in either direction. It is
       * "the export declined to say", and putting it above a measured
       * zero when sorting ascending would rank absence as achievement.
       */
      if (a === null && b === null) return 0;
      if (a === null) return 1;
      if (b === null) return -1;
      if (typeof a === "string" || typeof b === "string") {
        const compared = String(a).localeCompare(String(b));
        return sort.desc ? -compared : compared;
      }
      return sort.desc ? b - a : a - b;
    });
  }, [data, columns, query, sort, state.filters.positions]);

  /*
   * §5.4.5 + §5.7.5: sorting a mixed-position table by a `normalizable`
   * metric in raw units is the exact mistake the tool exists to teach the
   * reader out of. Permitted — and never silent.
   */
  const sortedColumn = columns.find((entry) => entry.id === sort.id);
  const mixedPositions = state.filters.positions.length !== 1;
  const sortedMetricKey =
    sortedColumn?.source.kind === "metric" ? sortedColumn.source.key : null;
  const sortedIsNormalizable =
    data.status === "ready" && sortedMetricKey !== null
      ? (data.columns.columns.find((entry) => entry.key === sortedMetricKey)?.normalizable ??
        false)
      : false;
  const rawNormalizableSort = !metricsNormalized && mixedPositions && sortedIsNormalizable;

  if (data.status === "loading") return <Loading progress={data.progress} />;
  if (data.status === "error") return <Failed error={data.error} />;

  const { players } = data;
  const height = viewport.current?.clientHeight ?? 640;
  const first = Math.max(Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN, 0);
  const last = Math.min(first + Math.ceil(height / ROW_HEIGHT) + OVERSCAN * 2, rows.length);
  const visible = rows.slice(first, last);

  return (
    <main className={styles.explorer}>
      <header className={styles.head}>
        <div>
          <h1 className={styles.title}>Explorer</h1>
          <p className={styles.sub}>
            Every element, every exported column, sortable. {players.season} after gameweek{" "}
            <span className="data">{players.gameweek}</span>.
          </p>
        </div>
        <Provenance header={players.header} basis={players.header.normalization_basis} />
      </header>

      <div className={styles.controls}>
        <label className={styles.searchLabel}>
          <span className={styles.searchText}>Find</span>
          <input
            type="search"
            className={styles.search}
            placeholder="name or club"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>

        <fieldset className={styles.group}>
          <legend className={styles.legend}>Position</legend>
          <div className={styles.chips}>
            <button
              type="button"
              className={styles.chip}
              data-on={state.filters.positions.length === 0 || undefined}
              onClick={() => dispatch({ type: "filters", filters: { positions: [] } })}
            >
              All
            </button>
            {["GK", "DEF", "MID", "FWD"].map((entry) => (
              <button
                key={entry}
                type="button"
                className={styles.chip}
                data-on={state.filters.positions.includes(entry) || undefined}
                onClick={() => dispatch({ type: "filters", filters: { positions: [entry] } })}
              >
                {entry}
              </button>
            ))}
          </div>
        </fieldset>

        <fieldset className={styles.group}>
          <legend className={styles.legend}>Trailing rates</legend>
          <label className={styles.toggle}>
            <input
              type="checkbox"
              checked={metricsNormalized}
              onChange={(event) => setMetricsNormalized(event.target.checked)}
            />
            <span>as position percentiles</span>
          </label>
        </fieldset>

        <p className={styles.count}>
          {rows.length.toLocaleString()} of{" "}
          <span className="data">{players.players.length.toLocaleString()}</span> players
        </p>
      </div>

      {rawNormalizableSort && !cautionDismissed && (
        /*
         * §5.7.5, in the register §5.8.7 reserves for it. "This is the
         * highest-value single piece of copy in the application. It is
         * the moment the tool teaches."
         */
        <p className={styles.caution} role="note">
          Sorting <span className="data">{sortedColumn?.label}</span> in raw units across
          positions — forwards will fill the top of this table regardless of how good any of
          them is at his own job. Switch the trailing rates to position percentiles to compare
          each player against his own group.
          <button
            type="button"
            className={styles.dismiss}
            onClick={() => setCautionDismissed(true)}
          >
            Dismiss
          </button>
        </p>
      )}

      <div
        className={styles.viewport}
        ref={viewport}
        onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}
      >
        <table className={styles.table}>
          <thead>
            {/*
             * §5.4.5's column groups, rendered rather than merely modelled.
             * Without them the table shows "minutes" twice and "goals"
             * twice — once as a projection component and once as a season
             * actual — which are different numbers in different units, and
             * a reader has no way to tell which is which.
             */}
            <tr className={styles.groupRow}>
              {groupSpans(columns).map((span, index) => (
                <th
                  key={`${span.group}-${index}`}
                  scope="colgroup"
                  colSpan={span.span}
                  className={styles.groupTh}
                >
                  {span.group}
                </th>
              ))}
            </tr>
            <tr>
              {columns.map((column) => {
                const active = sort.id === column.id;
                return (
                  <th
                    key={column.id}
                    scope="col"
                    className={styles.th}
                    data-numeric={column.numeric || undefined}
                    data-active={active || undefined}
                    aria-sort={active ? (sort.desc ? "descending" : "ascending") : "none"}
                  >
                    <button
                      type="button"
                      className={styles.sortButton}
                      title={`${column.group} · ${column.label}`}
                      onClick={() =>
                        setSort((current) =>
                          current.id === column.id
                            ? { id: column.id, desc: !current.desc }
                            : { id: column.id, desc: column.numeric },
                        )
                      }
                    >
                      {column.label}
                      <span className={styles.sortMark} aria-hidden="true">
                        {active ? (sort.desc ? "▾" : "▴") : ""}
                      </span>
                    </button>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {/* Spacers stand in for the rows outside the window, so the
                scrollbar describes the whole table rather than the slice. */}
            {first > 0 && (
              <tr style={{ height: first * ROW_HEIGHT }} aria-hidden="true">
                <td colSpan={columns.length} />
              </tr>
            )}
            {visible.map((player) => (
              <tr
                key={player.element_id}
                className={styles.row}
                data-selected={state.selection.includes(player.element_id) || undefined}
                onClick={() => dispatch({ type: "toggleSelect", id: player.element_id })}
                title="Add to or remove from the Comparison selection"
              >
                {columns.map((column) => {
                  const value = valueOf(player, column.source, schedule);
                  return (
                    <td
                      key={column.id}
                      className={styles.td}
                      data-numeric={column.numeric || undefined}
                    >
                      {value === null ? (
                        <span className={styles.null}>—</span>
                      ) : typeof value === "string" ? (
                        value
                      ) : (
                        column.format(value)
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
            {last < rows.length && (
              <tr style={{ height: (rows.length - last) * ROW_HEIGHT }} aria-hidden="true">
                <td colSpan={columns.length} />
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <p className={styles.footnote}>
        {metricsNormalized
          ? "Trailing rates show percentile within position, read from the export. A player below the minutes floor renders as an em dash — the export declined to place him, which is not the same as placing him last."
          : "Trailing rates show raw per-90 values over the season to date."}
      </p>
    </main>
  );
}

// --- columns ----------------------------------------------------------

/**
 * §5.4.5's column groups, built from the registry rather than listed
 * here: "identity, price/ownership, projection + components, trailing
 * actuals, position-normalized companions."
 */
function buildColumns(
  players: PlayersFile,
  registry: ColumnsFile,
  metricsNormalized: boolean,
  hasSchedule: boolean,
): Column[] {
  const int = (value: number) => value.toFixed(0);
  const two = (value: number) => value.toFixed(2);
  const money = (value: number) => `£${(value / 10).toFixed(1)}`;
  const pct = (value: number) => `${Math.round(value * 100)}`;

  const columns: Column[] = [
    { id: "name", label: "Player", group: "Identity", source: { kind: "identity", key: "name" }, numeric: false, format: int },
    { id: "team", label: "Club", group: "Identity", source: { kind: "identity", key: "team" }, numeric: false, format: int },
    { id: "position", label: "Pos", group: "Identity", source: { kind: "identity", key: "position" }, numeric: false, format: int },
    // Kept beside the other identity columns so the group header spans a
    // contiguous run — a second "Identity" band further along the row
    // reads as a different group rather than the same one.
    { id: "gameweeks", label: "GWs", group: "Identity", source: { kind: "gameweeks" }, numeric: true, format: int },
    { id: "price", label: "Price", group: "Price and ownership", source: { kind: "price" }, numeric: true, format: money },
    { id: "selected", label: "Owned", group: "Price and ownership", source: { kind: "ownership" }, numeric: true, format: int },
    { id: "projection", label: "Proj", group: "Projection", source: { kind: "projection" }, numeric: true, format: two },
  ];

  // Projection components, in the order `players.json` carries them.
  const sample = players.players.find((player) => player.projection !== null);
  for (const key of Object.keys(sample?.projection?.components ?? {})) {
    columns.push({
      id: `component:${key}`,
      label: key.replace(/_/g, " ").slice(0, 12),
      group: "Projection components",
      source: { kind: "component", key },
      numeric: true,
      format: two,
    });
  }

  // Trailing actuals.
  for (const key of Object.keys(players.players[0]?.actuals ?? {})) {
    columns.push({
      id: `actual:${key}`,
      label: key.replace(/_/g, " ").slice(0, 12),
      group: "Trailing actuals",
      source: { kind: "actual", key },
      numeric: true,
      format: int,
    });
  }

  /*
   * §5.4.5's "next-N fixture difficulty". Only when the schedule covers
   * the season on screen — an archive season gets no columns rather than
   * a column of em dashes implying its fixtures were unknowable.
   *
   * The mean over the next five is first because it is the column anyone
   * actually sorts by: "who has the kindest run" is the question, and
   * sorting ascending answers it.
   */
  if (hasSchedule) {
    columns.push({
      id: "run5",
      label: "Next 5",
      group: "Next fixtures",
      source: { kind: "run", count: 5 },
      numeric: true,
      format: (value: number) => value.toFixed(2),
    });
    for (let index = 0; index < 5; index += 1) {
      columns.push({
        id: `fixture:${index}`,
        label: `+${index + 1}`,
        group: "Next fixtures",
        source: { kind: "fixture", index },
        numeric: true,
        format: int,
      });
    }
  }

  // The rate metrics, raw or as position percentiles (§5.4.5's toggle).
  for (const key of Object.keys(players.players[0]?.metrics ?? {})) {
    const spec = registry.columns.find((entry) => entry.key === key);
    columns.push({
      id: `metric:${key}`,
      label: spec?.label?.replace(" per 90", "/90") ?? key,
      group: metricsNormalized ? "Trailing rates, percentile" : "Trailing rates, raw",
      source: { kind: "metric", key },
      numeric: true,
      format: metricsNormalized ? pct : two,
    });
  }

  return columns;
}

/** Consecutive runs of columns sharing a group, for the header spans. */
function groupSpans(columns: Column[]): { group: string; span: number }[] {
  const spans: { group: string; span: number }[] = [];
  for (const column of columns) {
    const last = spans[spans.length - 1];
    if (last && last.group === column.group) last.span += 1;
    else spans.push({ group: column.group, span: 1 });
  }
  return spans;
}

/** One cell's value. `null` means the export declined to say (§5.3.3). */
function valueOf(
  player: PlayerRow,
  source: Source,
  schedule: Map<string, TeamFixture[]>,
): number | string | null {
  switch (source.kind) {
    case "identity":
      return player[source.key];
    case "price":
      return player.price;
    case "ownership":
      return player.selected;
    case "gameweeks":
      return player.gameweeks;
    case "projection":
      return player.projection?.total ?? null;
    case "component":
      return player.projection?.components[source.key] ?? null;
    case "actual":
      return player.actuals[source.key] ?? null;
    case "metric":
      return player.metrics[source.key]?.value ?? null;
    case "run":
      return runDifficulty(schedule, player.team, source.count);
    case "fixture": {
      const run = nextFixtures(schedule, player.team, source.index + 1);
      const entry = run[source.index];
      // A club with fewer remaining fixtures than the column asks for has
      // no value here, and that is not a difficulty of zero (§5.3.3).
      return entry?.difficulty ?? null;
    }
  }
}

// --- states -----------------------------------------------------------

function Loading({ progress }: { progress: LoadProgress | null }) {
  const kb = (bytes: number) => `${Math.round(bytes / 1024)} KB`;
  const pct =
    progress?.total != null ? Math.round((progress.received / progress.total) * 100) : null;
  return (
    <main className={styles.explorer}>
      <h1 className={styles.title}>Explorer</h1>
      <div className={styles.loading}>
        <div
          className={styles.track}
          role="progressbar"
          aria-valuenow={pct ?? undefined}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Loading players"
        >
          <div className={styles.fill} style={{ inlineSize: pct === null ? "30%" : `${pct}%` }} />
        </div>
        <p className={`${styles.sub} data`}>
          {progress
            ? `players.json — ${kb(progress.received)}${progress.total ? ` of ${kb(progress.total)}` : ""}`
            : "requesting players.json"}
        </p>
      </div>
    </main>
  );
}

function Failed({ error }: { error: Error }) {
  return (
    <main className={styles.explorer}>
      <h1 className={styles.title}>Explorer</h1>
      <div className={styles.failure} role="alert">
        <p className={styles.failureHead}>players.json did not load.</p>
        <p className={`${styles.sub} data`}>{error.message}</p>
        <p className={styles.sub}>
          Run <code className="data">python -m web.export players</code> and reload.
        </p>
      </div>
    </main>
  );
}
