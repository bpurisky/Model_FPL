/**
 * Elite Ownership: how the top-N managers' squads differ from the field,
 * and effective ownership (EO).
 *
 * Descriptive only (§5.0.2): it reports what a sample of the best
 * managers did last deadline and says nothing about whether copying them
 * pays. "Top N" is the top of the overall table at collection time.
 *
 * EO is `sum(multiplier) / n_managers * 100` -- a captain counts twice, a
 * triple captain three times, the bench not at all -- so it can exceed
 * 100. `delta` is top-N ownership minus the field's selected-by %.
 *
 * With no picks collected yet the file says `no_data` and the surface
 * shows an empty state rather than an empty table.
 */

import { useEffect, useMemo, useState } from "react";
import { ErrorState } from "../components/ErrorState";
import { LoadingState } from "../components/LoadingState";
import { Provenance } from "../components/Provenance";
import { loadOwnership, type LoadProgress } from "../data/load";
import type { OwnershipFile, OwnershipPlayer } from "../data/schema";
import { count } from "../design/text";
import styles from "./EliteOwnership.module.css";

type State =
  | { status: "loading"; progress: LoadProgress | null }
  | { status: "error"; error: Error }
  | { status: "ready"; file: OwnershipFile };

const POSITIONS = ["GK", "DEF", "MID", "FWD"] as const;
const INITIAL_ROWS = 60;

// --- pure logic (tested directly) -----------------------------------------

export type SortKey =
  | "name"
  | "team"
  | "position"
  | "now_cost"
  | "overall_pct"
  | "top_pct"
  | "eo_pct"
  | "captain_pct"
  | "delta_pct";

export interface SortState {
  key: SortKey;
  dir: "asc" | "desc";
}

export const DEFAULT_SORT: SortState = { key: "eo_pct", dir: "desc" };

export interface OwnershipFilters {
  position: string;
  query: string;
}

export const NO_OWNERSHIP_FILTERS: OwnershipFilters = { position: "all", query: "" };

export function filterOwnership(players: OwnershipPlayer[], filters: OwnershipFilters): OwnershipPlayer[] {
  const query = filters.query.trim().toLowerCase();
  return players.filter(
    (p) =>
      (filters.position === "all" || p.position === filters.position) &&
      (query === "" || p.name.toLowerCase().includes(query) || p.team.toLowerCase().includes(query)),
  );
}

/** Stable sort; nulls (no overall ownership known) always sort last,
 * whichever the direction, so a missing value never reads as the extreme. */
export function sortOwnership(players: OwnershipPlayer[], sort: SortState): OwnershipPlayer[] {
  const sign = sort.dir === "asc" ? 1 : -1;
  return [...players].sort((a, b) => {
    const x = a[sort.key];
    const y = b[sort.key];
    if (x === null && y === null) return a.element_id - b.element_id;
    if (x === null) return 1;
    if (y === null) return -1;
    const order = typeof x === "string" && typeof y === "string" ? x.localeCompare(y) : Number(x) - Number(y);
    return order !== 0 ? sign * order : a.element_id - b.element_id;
  });
}

/** Clicking the active column flips it; a new column starts descending
 * for numbers and ascending for text. */
export function nextSort(current: SortState, key: SortKey): SortState {
  if (current.key === key) return { key, dir: current.dir === "asc" ? "desc" : "asc" };
  return { key, dir: key === "name" || key === "team" || key === "position" ? "asc" : "desc" };
}

export const fmtPct = (value: number | null): string => (value === null ? "n/a" : `${value.toFixed(1)}%`);

export const fmtDelta = (value: number | null): string =>
  value === null ? "n/a" : `${value > 0 ? "+" : value < 0 ? "−" : ""}${Math.abs(value).toFixed(1)}`;

export function formatUtc(iso: string | null): string {
  if (iso === null) return "n/a";
  return `${new Date(iso).toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

const CHIP_LABEL: Record<string, string> = {
  "3xc": "Triple Captain",
  bboost: "Bench Boost",
  freehit: "Free Hit",
  wildcard: "Wildcard",
  manager: "Assistant Manager",
};

// --- view ------------------------------------------------------------------

export function EliteOwnership() {
  const [data, setData] = useState<State>({ status: "loading", progress: null });

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const file = await loadOwnership((progress) =>
          cancelled ? undefined : setData({ status: "loading", progress }),
        );
        if (!cancelled) setData({ status: "ready", file });
      } catch (error) {
        if (!cancelled) setData({ status: "error", error: error as Error });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (data.status === "loading")
    return (
      <LoadingState
        rootClassName={styles.root}
        title="Elite Ownership"
        file="ownership.json"
        label="Loading elite ownership"
        progress={data.progress}
      />
    );
  if (data.status === "error")
    return (
      <ErrorState
        rootClassName={styles.root}
        title="Elite Ownership"
        heading="ownership.json did not load."
        command="ownership"
        error={data.error}
      />
    );

  const { file } = data;
  return (
    <main className={styles.root}>
      <header className={styles.head}>
        <div>
          <h1 className={styles.title}>Elite Ownership</h1>
          <p className={styles.sub}>
            What the top managers in the overall rankings own, start and captain, against the whole
            field. Effective ownership (EO) counts a captain twice, so it can pass 100%. This
            describes the sample; it is not a recommendation.
          </p>
        </div>
        <Provenance header={file.header} basis={file.header.normalization_basis} />
      </header>
      {file.status === "ok" ? <Table file={file} /> : <NoData />}
    </main>
  );
}

function NoData() {
  return (
    <section className={styles.panel} aria-labelledby="nodata-heading">
      <h2 id="nodata-heading" className={styles.panelTitle}>
        No elite picks collected yet
      </h2>
      <p className={styles.empty}>
        The collector reads the picks of the top managers once after each gameweek deadline. None
        have been collected yet, so there is nothing to compare. This page fills in after the next
        deadline.
      </p>
    </section>
  );
}

const COLUMNS: { key: SortKey; label: string; left?: boolean; title?: string }[] = [
  { key: "name", label: "Player", left: true },
  { key: "team", label: "Club", left: true },
  { key: "position", label: "Pos" },
  { key: "now_cost", label: "Price" },
  { key: "overall_pct", label: "Overall own%", title: "Selected by, whole field" },
  { key: "top_pct", label: "Top-N own%", title: "Share of the top N holding him in their 15" },
  { key: "eo_pct", label: "EO%", title: "Effective ownership: sum of multipliers / managers" },
  { key: "captain_pct", label: "Captain%", title: "Share of the top N captaining him" },
  { key: "delta_pct", label: "Delta", title: "Top-N own% minus overall own%, percentage points" },
];

function Table({ file }: { file: OwnershipFile }) {
  const [filters, setFilters] = useState<OwnershipFilters>(NO_OWNERSHIP_FILTERS);
  const [sort, setSort] = useState<SortState>(DEFAULT_SORT);
  const [showAll, setShowAll] = useState(false);

  const visible = useMemo(
    () => sortOwnership(filterOwnership(file.players, filters), sort),
    [file.players, filters, sort],
  );
  const rows = showAll ? visible : visible.slice(0, INITIAL_ROWS);
  const patch = (next: Partial<OwnershipFilters>) => {
    setFilters({ ...filters, ...next });
    setShowAll(false);
  };

  return (
    <section className={styles.panel} aria-labelledby="own-heading">
      <h2 id="own-heading" className={styles.panelTitle}>
        Gameweek {file.gw}: top {file.n_managers.toLocaleString()} managers
      </h2>
      <p className={styles.panelSub}>
        Picks of <span className="data">{count(file.n_managers, "manager")}</span> for gameweek{" "}
        <span className="data">{file.gw}</span>, collected{" "}
        <span className="data">{formatUtc(file.fetched_at)}</span>. Overall ownership is the
        field&rsquo;s selected-by % at that time.
        {file.chips.length > 0 && (
          <>
            {" "}
            Chips played:{" "}
            {file.chips.map((c, i) => (
              <span key={c.chip}>
                {i > 0 && ", "}
                {CHIP_LABEL[c.chip] ?? c.chip} <span className="data">{c.count}</span>
              </span>
            ))}
            .
          </>
        )}
      </p>

      <div className={styles.filters}>
        <div className={styles.group} role="group" aria-label="Position filter">
          {["all", ...POSITIONS].map((position) => (
            <button
              key={position}
              type="button"
              className={styles.chip}
              aria-pressed={filters.position === position}
              onClick={() => patch({ position })}
            >
              {position === "all" ? "All" : position}
            </button>
          ))}
        </div>
        <label className={styles.field}>
          <span className={styles.fieldLabel}>Search</span>
          <input
            type="search"
            value={filters.query}
            placeholder="Player or club"
            onChange={(e) => patch({ query: e.target.value })}
          />
        </label>
      </div>

      <div className={styles.tableScroll}>
        <table className={styles.table}>
          <thead>
            <tr>
              {COLUMNS.map((c) => (
                <th
                  key={c.key}
                  scope="col"
                  className={c.left ? styles.left : undefined}
                  title={c.title}
                  aria-sort={sort.key === c.key ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}
                >
                  <button type="button" className={styles.sortBtn} onClick={() => setSort(nextSort(sort, c.key))}>
                    {c.label}
                    {sort.key === c.key && <span aria-hidden="true">{sort.dir === "asc" ? " ▲" : " ▼"}</span>}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((p) => (
              <tr key={p.element_id}>
                <th scope="row" className={styles.left}>{p.name}</th>
                <td className={styles.left}>{p.team}</td>
                <td>{p.position}</td>
                <td>{p.now_cost === null ? "n/a" : (p.now_cost / 10).toFixed(1)}</td>
                <td>{fmtPct(p.overall_pct)}</td>
                <td>{fmtPct(p.top_pct)}</td>
                <td>{fmtPct(p.eo_pct)}</td>
                <td>{fmtPct(p.captain_pct)}</td>
                <td
                  className={styles.delta}
                  data-sign={p.delta_pct === null ? undefined : p.delta_pct > 0 ? "up" : p.delta_pct < 0 ? "down" : undefined}
                >
                  {fmtDelta(p.delta_pct)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {visible.length === 0 && <p className={styles.empty}>No players match these filters.</p>}
      {visible.length > rows.length && (
        <button type="button" className={styles.more} onClick={() => setShowAll(true)}>
          Show all {visible.length.toLocaleString()} players
        </button>
      )}
    </section>
  );
}
