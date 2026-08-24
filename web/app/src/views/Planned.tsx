/**
 * The empty state for a surface that does not exist yet (§5.1.3).
 *
 * §5.14.14: "No mocked or placeholder data ships in any state, including
 * empty states." So this renders nothing that looks like a chart, no
 * greyed-out mock, no shimmering placeholder — just what the surface will
 * be, which milestone owns it, and where to go instead.
 *
 * The distinction between a Phase 5 surface and a Phase 3/4 one is kept
 * visible because they are different kinds of absence: one is scheduled
 * work in this phase, the other is a boundary this phase deliberately
 * does not cross (§5.0.2).
 */

import { useApp } from "../app/state";
import type { Surface } from "../app/surfaces";
import styles from "./Planned.module.css";

export function Planned({ surface }: { surface: Surface }) {
  const { dispatch } = useApp();
  const outOfPhase = surface.status === "out_of_phase";

  return (
    <main className={styles.planned}>
      <h1 className={styles.title}>{surface.label}</h1>
      <p className={styles.blurb}>{surface.blurb}</p>

      <p className={styles.status}>
        {outOfPhase ? (
          <>
            Outside Phase 5. <span className="data">{surface.milestone}</span> owns this, and
            this phase does not stub it with invented data — a mocked squad is a lie that
            survives into screenshots.
          </>
        ) : (
          <>
            Not built yet. Milestone <span className="data">{surface.milestone}</span> owns it.
            The export it reads from already exists, so this is UI work rather than pipeline
            work.
          </>
        )}
      </p>

      <button
        type="button"
        className={styles.back}
        onClick={() => dispatch({ type: "navigate", view: "correlations" })}
      >
        Back to the Correlation Lab
      </button>
    </main>
  );
}
