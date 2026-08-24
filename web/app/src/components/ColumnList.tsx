/**
 * The column list (§5.4.2).
 *
 * > "Column list built from `columns.json`, grouped by source and sorted
 * > with `position_relevance: primary` first when a position filter is
 * > active."
 *
 * Both halves of that matter. Grouping by source is what keeps `derived`
 * rates distinguishable from what FPL published, which is the first thing
 * a reader needs to know about a number. Sorting by relevance is what
 * stops a goalkeeper question opening on `xg_per90`.
 *
 * Selection here is a *hold*, not an assignment: clicking a column picks
 * it up, and clicking a drop zone puts it down. That is the keyboard path
 * §5.10 requires — "a column can be assigned to a channel without a
 * pointer" — and it is also the pointer path on a touch screen, where
 * HTML5 drag does not exist. Dragging still works for a mouse; it is the
 * convenience, not the mechanism.
 */

import { useMemo, useState } from "react";
import type { ColumnSpec } from "../data/schema";
import styles from "./ColumnList.module.css";

export interface ColumnListProps {
  columns: ColumnSpec[];
  /** The app-level position filter, or "all" (§5.5.3). */
  position: string;
  /** The column currently picked up, if any. */
  held: string | null;
  onHold: (key: string | null) => void;
  /** Which channel each column currently occupies, for the badges. */
  assigned: Map<string, string>;
}

const SOURCE_LABELS: Record<string, string> = {
  fpl_api: "From FPL",
  vaastav_archive: "From the archive",
  derived: "Derived here",
  model: "From the model",
};

const RELEVANCE_ORDER = { primary: 0, secondary: 1, context: 2, none: 3 } as const;

export function ColumnList({ columns, position, held, onHold, assigned }: ColumnListProps) {
  const [query, setQuery] = useState("");

  const groups = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const matching = columns.filter(
      (column) =>
        needle === "" ||
        column.label.toLowerCase().includes(needle) ||
        column.key.toLowerCase().includes(needle),
    );

    const relevance = (column: ColumnSpec): number => {
      if (position === "all") return 0;
      const entry = column.position_relevance[position as "GK" | "DEF" | "MID" | "FWD"];
      return RELEVANCE_ORDER[entry ?? "none"];
    };

    const bySource = new Map<string, ColumnSpec[]>();
    for (const column of matching) {
      const list = bySource.get(column.source) ?? [];
      list.push(column);
      bySource.set(column.source, list);
    }

    for (const list of bySource.values()) {
      list.sort((left, right) => {
        const byRelevance = relevance(left) - relevance(right);
        if (byRelevance !== 0) return byRelevance;
        return left.label.localeCompare(right.label);
      });
    }

    return [...bySource.entries()].sort(([left], [right]) =>
      (SOURCE_LABELS[left] ?? left).localeCompare(SOURCE_LABELS[right] ?? right),
    );
  }, [columns, position, query]);

  return (
    <div className={styles.list}>
      <label className={styles.searchLabel}>
        <span className={styles.searchText}>Columns</span>
        <input
          type="search"
          className={styles.search}
          placeholder="filter"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      </label>

      {held && (
        <p className={styles.holding} role="status">
          Holding <span className="data">{columns.find((c) => c.key === held)?.label ?? held}</span>.
          Choose a zone, or press Escape.
        </p>
      )}

      <div
        className={styles.scroll}
        onKeyDown={(event) => {
          if (event.key === "Escape") onHold(null);
        }}
      >
        {groups.map(([source, entries]) => (
          <section key={source} className={styles.group}>
            <h3 className={styles.groupName}>{SOURCE_LABELS[source] ?? source}</h3>
            <ul className={styles.entries}>
              {entries.map((column) => {
                const channel = assigned.get(column.key);
                const isHeld = held === column.key;
                const relevant =
                  position !== "all" &&
                  column.position_relevance[position as "GK" | "DEF" | "MID" | "FWD"] === "primary";
                return (
                  <li key={column.key}>
                    <button
                      type="button"
                      className={styles.column}
                      data-held={isHeld || undefined}
                      data-assigned={channel ? true : undefined}
                      data-primary={relevant || undefined}
                      aria-pressed={isHeld}
                      draggable
                      onDragStart={(event) => {
                        event.dataTransfer.setData("text/plain", column.key);
                        event.dataTransfer.effectAllowed = "copy";
                        onHold(column.key);
                      }}
                      onDragEnd={() => onHold(null)}
                      onClick={() => onHold(isHeld ? null : column.key)}
                      title={column.definition}
                    >
                      <span className={styles.columnLabel}>{column.label}</span>
                      <span className={styles.role} aria-hidden="true">
                        {column.role[0]!.toUpperCase()}
                      </span>
                      {channel && <span className={styles.badge}>{channel}</span>}
                    </button>
                  </li>
                );
              })}
            </ul>
          </section>
        ))}
      </div>
    </div>
  );
}
