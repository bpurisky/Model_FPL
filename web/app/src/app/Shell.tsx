/**
 * The app shell: navigation, route switch, and the surfaces that exist.
 *
 * The Graph Builder is behind `lazy()` and that is a budget decision, not
 * a style one. §5.9 allows the query chunk 1.2 MB and requires it in "no
 * initial-load chunk"; the only thing that actually enforces that is the
 * split point being here, above anything that imports `query/`. A static
 * import of a panel-backed view anywhere in this file would pull the
 * parquet reader into the landing bundle and the budget would erode with
 * no error to notice it by.
 *
 * That mattered more when the chunk was DuckDB-WASM at 4.6 MB brotli
 * (§5.16 D10 replaced it). It still matters: the rule is about where the
 * boundary is, not about how much is currently behind it.
 */

import { lazy, Suspense, useEffect } from "react";
import { Comparison } from "../views/Comparison";
import { CorrelationLab } from "../views/CorrelationLab";
import { Planned } from "../views/Planned";
import { AppState, useApp } from "./state";
import { SURFACES } from "./surfaces";
import styles from "./Shell.module.css";

const GraphBuilder = lazy(() =>
  import("../views/GraphBuilder").then((module) => ({ default: module.GraphBuilder })),
);

const FormMatrix = lazy(() =>
  import("../views/FormMatrix").then((module) => ({ default: module.FormMatrix })),
);

export function App() {
  return (
    <AppState>
      <Shell />
    </AppState>
  );
}

function Shell() {
  const { state, dispatch } = useApp();
  const current = SURFACES.find((surface) => surface.view === state.view) ?? SURFACES[0]!;

  /*
   * §5.5 makes the URL linkable, and a link people keep is a link they
   * bookmark. A tab reading "Correlation Lab" while showing the Graph
   * Builder makes every bookmark and every restored window wrong about
   * what it points at.
   */
  useEffect(() => {
    document.title = `${current.label} — fpl-trends`;
  }, [current.label]);

  return (
    <div className={styles.shell}>
      <nav className={styles.nav} aria-label="Surfaces">
        <span className={styles.wordmark}>
          fpl<span className={styles.wordmarkDim}>-trends</span>
        </span>
        <ul className={styles.list}>
          {SURFACES.map((surface) => {
            const active = surface.view === state.view;
            const reachable = surface.status !== "out_of_phase";
            return (
              <li key={surface.view}>
                <button
                  type="button"
                  className={styles.tab}
                  data-active={active || undefined}
                  data-status={surface.status}
                  aria-current={active ? "page" : undefined}
                  disabled={!reachable}
                  /*
                   * §5.1.3: a Phase 3/4 entry is disabled and says what
                   * will live there. A disabled control with no
                   * explanation is just a dead end, so the blurb is the
                   * title rather than a tooltip nobody finds.
                   */
                  title={reachable ? surface.blurb : `${surface.milestone} — ${surface.blurb}`}
                  onClick={() => dispatch({ type: "navigate", view: surface.view })}
                >
                  {surface.label}
                  {surface.status !== "live" && (
                    <span className={styles.badge}>{surface.milestone}</span>
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className={styles.body}>
        {state.view === "correlations" && <CorrelationLab />}
        {state.view === "graph" && (
          <Suspense fallback={<EngineLoading />}>
            <GraphBuilder />
          </Suspense>
        )}
        {state.view === "form" && (
          <Suspense fallback={<EngineLoading />}>
            <FormMatrix />
          </Suspense>
        )}
        {state.view === "compare" && <Comparison />}
        {current.status !== "live" && <Planned surface={current} />}
      </div>
    </div>
  );
}

/**
 * §5.8.8 forbids skeletons and §5.9 wants the user to know what they are
 * waiting on. This is the chunk, not the data — the data reports its own
 * bytes once the builder mounts — so it says which of the two it is.
 */
function EngineLoading() {
  return (
    <main className={styles.pending}>
      <p className={styles.pendingText}>
        Loading the panel reader. It is fetched only on the routes that read{" "}
        <span className="data">panel.parquet</span>, never on the landing bundle.
      </p>
    </main>
  );
}
