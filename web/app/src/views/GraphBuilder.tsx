/**
 * §5.4.2 — the user-driven exploration surface.
 *
 * > "The user chooses data, not chart type."
 *
 * Everything that decides *what* is drawn lives elsewhere: the §5.4.2
 * table in `encoding/mark.ts`, the SQL in `query/panel.ts`, the seven
 * permitted reductions in `query/reduce.ts`, and the shaping in
 * `encoding/data.ts`. This file is the wiring and the copy — which is
 * most of what makes the surface usable, and none of what makes it
 * correct.
 *
 * This module is the split point for §5.9's engine budget. It is imported
 * only through `lazy()` in `Shell.tsx`, and it is the first thing in the
 * app to touch `query/`. Importing it statically anywhere would pull
 * DuckDB-WASM into the landing bundle.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Chart } from "../components/Chart";
import { ColumnList } from "../components/ColumnList";
import { DropZones } from "../components/DropZones";
import { FilterBar } from "../components/FilterBar";
import { Provenance } from "../components/Provenance";
import { useApp } from "../app/state";
import { loadColumns, type LoadProgress } from "../data/load";
import type { ColumnsFile, ColumnSpec } from "../data/schema";
import { buildPlot, collectKeysFor, groupKeysFor } from "../encoding/data";
import { inferMark } from "../encoding/mark";
import { AGGREGATES, rolesOf, type Channel } from "../encoding/spec";
import { facets as loadFacets, grouped, type PanelFacets } from "../query/panel";
import { openSession, PanelMissingError, type Session } from "../query/session";
import styles from "./GraphBuilder.module.css";

type Engine =
  | { status: "opening"; progress: LoadProgress | null }
  | { status: "ready"; session: Session; facets: PanelFacets; columns: ColumnsFile }
  | { status: "absent" }
  | { status: "error"; error: Error };

export function GraphBuilder() {
  const { state, dispatch } = useApp();
  const [engine, setEngine] = useState<Engine>({ status: "opening", progress: null });
  const [held, setHeld] = useState<string | null>(null);
  const [rows, setRows] = useState<Awaited<ReturnType<typeof grouped>> | null>(null);
  const [querying, setQuerying] = useState(false);
  const [queryError, setQueryError] = useState<Error | null>(null);
  const [cautionDismissed, setCautionDismissed] = useState(false);
  /** Whether the reader has overridden §5.7.3's default for this surface. */
  const [toggleTouched, setToggleTouched] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const columns = await loadColumns();
        const session = await openSession((progress) =>
          cancelled ? undefined : setEngine({ status: "opening", progress }),
        );
        const panelFacets = await loadFacets(session);
        if (!cancelled) setEngine({ status: "ready", session, facets: panelFacets, columns });
      } catch (error) {
        if (cancelled) return;
        setEngine(
          error instanceof PanelMissingError
            ? { status: "absent" }
            : { status: "error", error: error as Error },
        );
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const columnIndex = useMemo(() => {
    if (engine.status !== "ready") return new Map<string, ColumnSpec>();
    return new Map(engine.columns.columns.map((column) => [column.key, column]));
  }, [engine]);

  /*
   * §5.7.3: "Graph Builder — raw when a single position is filtered;
   * normalized when position filter is 'all'. The user asked a
   * cross-position question; answer it honestly."
   *
   * Applied as a default rather than as a lock. The toggle is app-level
   * state (§5.5.3) and stays wherever the reader last put it, so the
   * default only speaks until they disagree with it.
   */
  const onePosition = state.filters.positions.length === 1;
  const normalized = toggleTouched ? state.normalized : !onePosition;

  const normalizedReason = onePosition
    ? normalized
      ? "Within-position z-scores over a single position — the same order as raw, on a different scale."
      : "Raw units. One position is filtered, so values are already comparable."
    : normalized
      ? "Within-position z-scores, so a defender and a forward are each measured against their own group."
      : "Raw units across positions. Forwards will dominate any attacking metric regardless of quality.";

  const roles = useMemo(() => rolesOf(state.encoding, columnIndex), [state.encoding, columnIndex]);
  const inference = useMemo(() => inferMark(roles), [roles]);

  const assigned = useMemo(() => {
    const map = new Map<string, string>();
    for (const channel of ["x", "y", "color", "wrap"] as const) {
      const key = state.encoding[channel];
      if (key) map.set(key, channel === "color" ? "colour" : channel);
    }
    return map;
  }, [state.encoding]);

  // --- the query ------------------------------------------------------
  const session = engine.status === "ready" ? engine.session : null;
  const plan = inference.ok ? inference.plan : null;
  const runId = useRef(0);

  useEffect(() => {
    if (!session || !plan) {
      setRows(null);
      return;
    }
    const id = ++runId.current;
    setQuerying(true);
    setQueryError(null);

    (async () => {
      try {
        const result = await grouped(
          session,
          {
            groupBy: groupKeysFor(plan, state.encoding),
            collect: collectKeysFor(plan, state.encoding),
            filters: state.filters,
            normalized,
          },
          columnIndex,
        );
        // A slow query must not overwrite a faster one issued after it.
        if (id === runId.current) setRows(result);
      } catch (error) {
        if (id === runId.current) setQueryError(error as Error);
      } finally {
        if (id === runId.current) setQuerying(false);
      }
    })();
  }, [session, plan, state.encoding, state.filters, normalized, columnIndex]);

  const plot = useMemo(() => {
    if (!rows || !plan) return null;
    return buildPlot(rows.rows, plan, state.encoding);
  }, [rows, plan, state.encoding]);

  const assign = useCallback(
    (channel: Channel, key: string | null) => dispatch({ type: "encode", channel, key }),
    [dispatch],
  );

  /*
   * §5.7.5's caution. Shown when a `normalizable` metric is plotted in
   * raw units across more than one position — the exact case where
   * "forwards will dominate this sort regardless of quality".
   */
  const rawMixedMetric = useMemo(() => {
    if (normalized || onePosition) return null;
    for (const channel of ["x", "y", "color"] as const) {
      const key = state.encoding[channel];
      if (!key) continue;
      const spec = columnIndex.get(key);
      if (spec?.normalizable) return spec;
    }
    return null;
  }, [normalized, onePosition, state.encoding, columnIndex]);

  if (engine.status === "opening") return <Opening progress={engine.progress} />;
  if (engine.status === "absent") return <PanelAbsent />;
  if (engine.status === "error") return <EngineFailed error={engine.error} />;

  const label = (key: string | null) => (key ? (columnIndex.get(key)?.label ?? key) : "");

  return (
    <main className={styles.builder}>
      <header className={styles.head}>
        <div>
          <h1 className={styles.title}>Graph Builder</h1>
          <p className={styles.sub}>
            Four channels over {engine.facets.rows.toLocaleString()} player-gameweeks. The mark
            follows the data — there is no chart-type menu.
          </p>
        </div>
        <Provenance header={engine.columns.header} basis={engine.columns.header.normalization_basis} />
      </header>

      <FilterBar
        facets={engine.facets}
        filters={state.filters}
        onChange={(filters) => dispatch({ type: "filters", filters })}
        normalized={normalized}
        onNormalized={(next) => {
          setToggleTouched(true);
          dispatch({ type: "normalized", normalized: next });
        }}
        normalizedReason={normalizedReason}
      />

      {rawMixedMetric && !cautionDismissed && (
        /*
         * §5.7.5, in the register §5.8.7 reserves for it: written for the
         * reader who does not yet have the vocabulary, naming the
         * specific distortion rather than the general rule.
         *
         * "This is the highest-value single piece of copy in the
         * application. It is the moment the tool teaches."
         */
        <p className={styles.caution} role="note">
          Raw <span className="data">{rawMixedMetric.label}</span> across positions — forwards
          will dominate this regardless of quality, because they take the shots. Switch to
          within-position to compare each player against his own group.
          <button
            type="button"
            className={styles.dismiss}
            onClick={() => setCautionDismissed(true)}
          >
            Dismiss
          </button>
        </p>
      )}

      <div className={styles.layout}>
        <aside className={styles.side}>
          <ColumnList
            columns={engine.columns.columns}
            position={onePosition ? state.filters.positions[0]! : "all"}
            held={held}
            onHold={setHeld}
            assigned={assigned}
          />
        </aside>

        <section className={styles.canvas}>
          <DropZones
            encoding={state.encoding}
            columns={columnIndex}
            held={held}
            onAssign={assign}
            onHold={setHeld}
          />

          {inference.ok ? (
            <>
              <div className={styles.markBar}>
                <p className={styles.markName}>
                  <span className="data">{inference.plan.mark}</span>
                  {plot && (
                    <span className={styles.markMeta}>
                      {plot.count.toLocaleString()} marks
                      {plot.dropped > 0 && (
                        <>
                          {" · "}
                          <span className={styles.dropped}>
                            {plot.dropped.toLocaleString()} with no value, not drawn
                          </span>
                        </>
                      )}
                      {rows?.truncated && (
                        <>
                          {" · "}
                          <span className={styles.dropped}>capped</span>
                        </>
                      )}
                    </span>
                  )}
                </p>

                <label className={styles.aggregate}>
                  <span>Aggregate</span>
                  <select
                    value={state.encoding.aggregate}
                    onChange={(event) =>
                      dispatch({
                        type: "encoding",
                        encoding: { aggregate: event.target.value as (typeof AGGREGATES)[number] },
                      })
                    }
                  >
                    {AGGREGATES.map((name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              {querying && !plot && <p className={styles.working}>Querying the panel…</p>}
              {queryError && (
                <p className={styles.failure} role="alert">
                  {queryError.message}
                </p>
              )}

              {plot && (
                <Chart
                  plot={plot}
                  plan={inference.plan}
                  xLabel={label(state.encoding.x)}
                  yLabel={
                    inference.plan.mark === "histogram" ? "players" : label(state.encoding.y)
                  }
                  colorLabel={state.encoding.color ? label(state.encoding.color) : null}
                  columns={columnIndex}
                  xKey={state.encoding.x}
                  yKey={state.encoding.y}
                  colorKey={state.encoding.color}
                />
              )}

              {/*
               * §5.6.3: every derived number renders with its provenance.
               * A reduction over a user's own filter is exactly the kind
               * of number that has none unless it is stated.
               */}
              {plot && (
                <p className={styles.provenance}>
                  <span className="data">{state.encoding.aggregate}</span> over the filtered
                  rows, computed in your browser by <span className="data">query/reduce.ts</span>,
                  checked against <span className="data">golden_reductions.json</span> in CI.
                  {rows && rows.normalizedKeys.length > 0 && (
                    <>
                      {" "}
                      <span className="data">{rows.normalizedKeys.join(", ")}</span> read as
                      within-position z-scores from the export; nothing is standardized here.
                    </>
                  )}
                </p>
              )}
            </>
          ) : (
            <p className={styles.guidance} role="note">
              {inference.reason}
            </p>
          )}
        </section>
      </div>
    </main>
  );
}

/** §5.9: a determinate bar with a byte count, never a spinner. */
function Opening({ progress }: { progress: LoadProgress | null }) {
  const mb = (bytes: number) => `${(bytes / 1_048_576).toFixed(1)} MB`;
  const pct =
    progress?.total != null ? Math.round((progress.received / progress.total) * 100) : null;
  return (
    <main className={styles.builder}>
      <h1 className={styles.title}>Graph Builder</h1>
      <div className={styles.loading}>
        <div
          className={styles.track}
          role="progressbar"
          aria-valuenow={pct ?? undefined}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Loading the panel"
        >
          <div className={styles.fill} style={{ inlineSize: pct === null ? "30%" : `${pct}%` }} />
        </div>
        <p className={`${styles.sub} data`}>
          {progress
            ? `panel.parquet — ${mb(progress.received)}${progress.total ? ` of ${mb(progress.total)}` : ""}`
            : "requesting panel.parquet"}
        </p>
      </div>
    </main>
  );
}

/**
 * §5.14.8: "A fresh clone renders every view except Graph Builder, Form
 * Matrix, and Trend Explorer with no pipeline run; those three show their
 * explanatory empty state rather than an error or a blank."
 */
function PanelAbsent() {
  return (
    <main className={styles.builder}>
      <h1 className={styles.title}>Graph Builder</h1>
      <div className={styles.absent}>
        <p className={styles.absentHead}>The panel has not been built.</p>
        <p className={styles.sub}>
          This surface reads <span className="data">panel.parquet</span> — 85,000
          player-gameweeks, which §5.3.4 deliberately does not commit. Every other surface reads
          committed JSON and works from a fresh clone.
        </p>
        <p className={styles.sub}>
          Run <code className="data">python -m web.export panel</code> and reload.
        </p>
      </div>
    </main>
  );
}

function EngineFailed({ error }: { error: Error }) {
  return (
    <main className={styles.builder}>
      <h1 className={styles.title}>Graph Builder</h1>
      <div className={styles.absent} role="alert">
        <p className={styles.absentHead}>The query engine did not start.</p>
        <p className={`${styles.sub} data`}>{error.message}</p>
      </div>
    </main>
  );
}
