/**
 * The app shell: navigation, route switch, and the two surfaces that
 * exist at 5C.
 *
 * The Graph Builder is behind `lazy()` and that is a budget decision, not
 * a style one. §5.9 allows the DuckDB-WASM chunk 1.2 MB and requires it
 * in "no initial-load chunk"; the only thing that actually enforces that
 * is the split point being here, above anything that imports `query/`.
 * A static import of the builder anywhere in this file would pull the
 * engine into the landing bundle and the budget would be gone with no
 * error to notice it by.
 */

import { lazy, Suspense } from "react";
import { CorrelationLab } from "../views/CorrelationLab";
import { Planned } from "../views/Planned";
import { AppState, useApp } from "./state";
import { SURFACES } from "./surfaces";
import styles from "./Shell.module.css";

const GraphBuilder = lazy(() =>
  import("../views/GraphBuilder").then((module) => ({ default: module.GraphBuilder })),
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
        Loading the query engine. <span className="data">DuckDB-WASM</span> is fetched only on
        this route, never on the landing bundle.
      </p>
    </main>
  );
}
