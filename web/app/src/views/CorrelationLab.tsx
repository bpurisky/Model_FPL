import { useEffect, useMemo, useState } from "react";
import { Legend } from "../components/Legend";
import { Matrix } from "../components/Matrix";
import { PositionFilter } from "../components/PositionFilter";
import { Provenance } from "../components/Provenance";
import { RankScatter } from "../components/RankScatter";
import {
  ContractError,
  loadColumns,
  loadCorrelations,
  loadPlayers,
  type LoadProgress,
} from "../data/load";
import type { ColumnsFile, CorrelationsFile, PlayersFile } from "../data/schema";
import styles from "./CorrelationLab.module.css";

type State =
  | { status: "loading"; progress: LoadProgress | null }
  | { status: "error"; error: Error }
  | {
      status: "ready";
      correlations: CorrelationsFile;
      columns: ColumnsFile;
      players: PlayersFile | null;
    };

/**
 * §5.4.1, and §5.8.4's hero: the landing view opens directly onto the
 * full-bleed Spearman heatmap. No KPI card row, no big-number hero, no
 * marketing band above it.
 *
 * "The spec's one deliberate risk: opening a tool on a dense matrix
 * rather than a summary is hostile to a first-time visitor and correct
 * for the only user this tool has."
 */
export function CorrelationLab() {
  const [state, setState] = useState<State>({ status: "loading", progress: null });
  const [position, setPosition] = useState("MID");
  const [selected, setSelected] = useState<{ a: string; b: string } | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [correlations, columns] = await Promise.all([
          loadCorrelations((progress) =>
            cancelled ? undefined : setState({ status: "loading", progress }),
          ),
          loadColumns(),
        ]);
        // players.json drives the scatter only, so a failure there costs
        // the scatter rather than the hero. Loaded after, not with.
        let players: PlayersFile | null = null;
        try {
          players = await loadPlayers();
        } catch {
          players = null;
        }
        if (!cancelled) setState({ status: "ready", correlations, columns, players });
      } catch (error) {
        if (!cancelled) {
          setState({ status: "error", error: error as Error });
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const columnIndex = useMemo(() => {
    if (state.status !== "ready") return new Map();
    return new Map(state.columns.columns.map((column) => [column.key, column]));
  }, [state]);

  const visible = useMemo(() => {
    if (state.status !== "ready") return [];
    return state.correlations.cells.filter((cell) => cell.group === position);
  }, [state, position]);

  if (state.status === "loading") {
    return <LoadingBar progress={state.progress} />;
  }

  if (state.status === "error") {
    return <Failure error={state.error} />;
  }

  const { correlations, players } = state;
  const hatched = visible.filter((cell) => cell.n < correlations.min_n_cell).length;
  const selectedCell = selected
    ? visible.find(
        (cell) =>
          (cell.a === selected.a && cell.b === selected.b) ||
          (cell.a === selected.b && cell.b === selected.a),
      )
    : undefined;
  const group = correlations.groups.find((entry) => entry.key === position);

  return (
    <main className={styles.lab}>
      <header className={styles.head}>
        <div>
          <h1 className={styles.title}>Correlation Lab</h1>
          <p className={styles.sub}>
            Within-position Spearman across {correlations.metrics.length} metrics,
            over player-seasons pooled from {correlations.seasons.join(", ")}.
          </p>
        </div>
        <Provenance header={correlations.header} basis={correlations.basis} />
      </header>

      <div className={styles.controls}>
        <PositionFilter
          groups={correlations.groups}
          value={position}
          onChange={(next) => {
            setPosition(next);
            setSelected(null);
          }}
        />
        <Legend minN={correlations.min_n_cell} hatchedCount={hatched} />
      </div>

      {group?.mixed_position && (
        /*
         * §5.7.5's caution, in the register §5.8.7 reserves for it —
         * written for the reader who does not yet have the vocabulary.
         * "This is the highest-value single piece of copy in the
         * application. It is the moment the tool teaches."
         */
        <p className={styles.caution} role="note">
          Pooling positions. Forwards and goalkeepers differ so
          systematically on most of these metrics that position alone
          drives part of any correlation here — a pair can look related
          because one group sits high on both. Filter to a single position
          to remove that.
        </p>
      )}

      <Matrix
        metrics={correlations.metrics}
        cells={visible}
        columns={columnIndex}
        minN={correlations.min_n_cell}
        selected={selected}
        onSelect={setSelected}
        revealKey={position}
      />

      {selected && selectedCell && players && (
        <RankScatter
          a={selected.a}
          b={selected.b}
          players={players.players}
          columns={columnIndex}
          position={position}
          season={players.season}
          rho={selectedCell.rho}
          n={selectedCell.n}
        />
      )}

      {selected && !players && (
        <p className={styles.caution} role="note">
          The scatter needs players.json, which did not load. The matrix
          above is unaffected — it carries its own numbers.
        </p>
      )}
    </main>
  );
}

/**
 * §5.9: the user should know whether they are waiting on 200 KB or 8 MB.
 * §5.8.8 forbids skeletons — they imply content shape before it is known,
 * which on a data tool is a small lie — so this is a determinate bar with
 * a byte count, or an honest indeterminate line when the server did not
 * send a length.
 */
function LoadingBar({ progress }: { progress: LoadProgress | null }) {
  const kb = (bytes: number) => `${Math.round(bytes / 1024)} KB`;
  const pct =
    progress?.total != null ? Math.round((progress.received / progress.total) * 100) : null;
  return (
    <main className={styles.lab}>
      <h1 className={styles.title}>Correlation Lab</h1>
      <div className={styles.loading}>
        <div
          className={styles.track}
          role="progressbar"
          aria-valuenow={pct ?? undefined}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Loading correlations"
        >
          <div className={styles.fill} style={{ inlineSize: pct === null ? "35%" : `${pct}%` }} />
        </div>
        <p className={`${styles.sub} data`}>
          {progress
            ? `${kb(progress.received)}${progress.total ? ` of ${kb(progress.total)}` : ""}`
            : "requesting correlations.json"}
        </p>
      </div>
    </main>
  );
}

/**
 * §5.8.7: errors state what failed and what to do, and never apologise.
 */
function Failure({ error }: { error: Error }) {
  const contract = error instanceof ContractError;
  return (
    <main className={styles.lab}>
      <h1 className={styles.title}>Correlation Lab</h1>
      <div className={styles.failure} role="alert">
        <p className={styles.failureHead}>
          {contract ? `${(error as ContractError).file} failed validation` : "Could not load the export"}
        </p>
        <p className={`${styles.sub} data`}>{error.message}</p>
        <p className={styles.sub}>
          Run <code className="data">python -m web.export all</code> and reload.
          Nothing is rendered from a payload that did not validate.
        </p>
      </div>
    </main>
  );
}
