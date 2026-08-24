/**
 * §5.4.7 — the Model Scorecard.
 *
 * > "Carried from v1 unchanged. Renders `backtest/report.py`'s existing
 * > outputs. Nothing here is new analysis; it is the existing report made
 * > legible."
 *
 * That sentence is the whole design brief and it is also a constraint:
 * every number on this surface comes out of `scorecard.json` exactly as
 * the walk-forward produced it. Nothing is recomputed, nothing is
 * combined, and the one place a reader might expect an average — across
 * seasons — is served by the export's own pooled row rather than by
 * re-aggregating the detail rows, which §5.6 forbids and which would
 * produce a third number matching neither the file nor the paper result.
 *
 * The margins here are small and the surface has to say so. The event
 * model beats the best baseline by 0.0055 MAE — about half a percent —
 * and a scorecard that rendered that as a triumphant bar chart would be
 * lying with a true number. So the model table prints the deltas, and
 * the copy says what they are worth.
 */

import { useEffect, useMemo, useState } from "react";
import { Provenance } from "../components/Provenance";
import { loadBoard, loadScorecard, type LoadProgress } from "../data/load";
import type { BoardFile, ScorecardFile, ScorecardRow } from "../data/schema";
import styles from "./Scorecard.module.css";

type State =
  | { status: "loading"; progress: LoadProgress | null }
  | { status: "error"; error: Error }
  | { status: "ready"; scorecard: ScorecardFile; board: BoardFile | null };

const MODEL_LABELS: Record<string, string> = {
  event_model: "Event model",
  fixture_adjusted_trailing_mean: "Trailing mean, fixture-adjusted",
  fpl_form_approx: "FPL form (approx)",
  trailing_mean: "Trailing mean",
};

const EVENT_LABELS: Record<string, string> = {
  "blank_(0_minutes)": "Did not play",
  played_no_goal_involvement: "Played, no involvement",
  goal_involvement: "Goal or assist",
  clean_sheet: "Clean sheet",
  bonus_earned: "Bonus earned",
};

export function Scorecard() {
  const [data, setData] = useState<State>({ status: "loading", progress: null });
  const [season, setSeason] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const scorecard = await loadScorecard((progress) =>
          cancelled ? undefined : setData({ status: "loading", progress }),
        );
        // The board is a bonus panel here; its absence must not cost the
        // backtest results.
        const board = await loadBoard().catch(() => null);
        if (!cancelled) setData({ status: "ready", scorecard, board });
      } catch (error) {
        if (!cancelled) setData({ status: "error", error: error as Error });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const headline = useMemo(() => {
    if (data.status !== "ready") return [];
    /*
     * `season: null` is the pooled-everything row and `gw: null` is the
     * season rollup. Those nulls are structural, so the grain is selected
     * by *filtering* on them rather than by re-aggregating detail rows.
     */
    return data.scorecard.rows.filter((row) => row.season === season && row.gw === null);
  }, [data, season]);

  if (data.status === "loading") return <Loading progress={data.progress} />;
  if (data.status === "error") return <Failed error={data.error} />;

  const { scorecard, board } = data;
  const best = bestBaseline(headline, scorecard.event_model);
  const event = headline.find((row) => row.model === scorecard.event_model);

  return (
    <main className={styles.scorecard}>
      <header className={styles.head}>
        <div>
          <h1 className={styles.title}>Model Scorecard</h1>
          <p className={styles.sub}>
            The walk-forward backtest, as it was recorded. Every number here is read from{" "}
            <span className="data">scorecard.json</span> — nothing on this page is recomputed
            or combined.
          </p>
        </div>
        <Provenance header={scorecard.header} basis={scorecard.header.normalization_basis} />
      </header>

      <div className={styles.controls}>
        <fieldset className={styles.group}>
          <legend className={styles.legend}>Slice</legend>
          <div className={styles.chips}>
            <button
              type="button"
              className={styles.chip}
              data-on={season === null || undefined}
              onClick={() => setSeason(null)}
              title="Every season pooled — the export's own row, not a mean of the others."
            >
              pooled
            </button>
            {scorecard.seasons.map((entry) => (
              <button
                key={entry}
                type="button"
                className={styles.chip}
                data-on={season === entry || undefined}
                onClick={() => setSeason(entry)}
              >
                {entry}
              </button>
            ))}
          </div>
        </fieldset>
      </div>

      <section className={styles.panel}>
        <h2 className={styles.panelTitle}>Against the baselines</h2>
        <p className={styles.panelSub}>
          Mean absolute error and RMSE in points, lower better; within-position Spearman,
          higher better. The Spearman is the unweighted mean across positions and carries no
          p-value on purpose — no sampling distribution describes it.
        </p>

        <div className={styles.tableScroll}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th scope="col" className={styles.left}>
                  Model
                </th>
                <th scope="col">MAE</th>
                <th scope="col">vs best baseline</th>
                <th scope="col">RMSE</th>
                <th scope="col">Spearman</th>
                <th scope="col">n</th>
              </tr>
            </thead>
            <tbody>
              {headline.map((row) => {
                const isEvent = row.model === scorecard.event_model;
                const delta =
                  isEvent && best?.mae != null && row.mae != null ? row.mae - best.mae : null;
                return (
                  <tr key={row.model} data-event={isEvent || undefined}>
                    <th scope="row" className={styles.left}>
                      {MODEL_LABELS[row.model] ?? row.model}
                    </th>
                    <td className="data">{fmt(row.mae, 4)}</td>
                    <td className="data">
                      {delta === null ? (
                        <span className={styles.muted}>—</span>
                      ) : (
                        /*
                         * The honest headline. 0.0055 MAE is about half a
                         * percent, and it is the whole margin the event
                         * model has over a trailing mean that costs
                         * nothing to compute.
                         */
                        <span className={delta < 0 ? styles.better : styles.worse}>
                          {delta < 0 ? "−" : "+"}
                          {Math.abs(delta).toFixed(4)}
                        </span>
                      )}
                    </td>
                    <td className="data">{fmt(row.rmse, 4)}</td>
                    <td className="data">{fmt(row.spearman_mean, 4)}</td>
                    <td className={`data ${styles.muted}`}>{row.n.toLocaleString()}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {event && best && (
          <p className={styles.finding}>
            The event model wins, and by very little: {fmt(event.mae, 4)} against{" "}
            {fmt(best.mae, 4)} for {MODEL_LABELS[best.model] ?? best.model}, which is{" "}
            <span className="data">
              {(((best.mae! - event.mae!) / best.mae!) * 100).toFixed(2)}%
            </span>
            . A model that decomposes into goals, assists, clean sheets and minutes earns
            about half a percent over taking a player&rsquo;s recent average. That is the
            result; the value of the decomposition is that you can see <em>where</em> a
            projection comes from, not that it is dramatically more accurate.
          </p>
        )}
      </section>

      <ByPosition rows={headline} eventModel={scorecard.event_model} />

      <Calibration file={scorecard} />

      <section className={styles.panel}>
        <h2 className={styles.panelTitle}>Where the error is</h2>
        <p className={styles.panelSub}>
          Mean absolute error split by what actually happened in the gameweek. The model is
          most wrong exactly where the points are.
        </p>
        <Bars
          rows={scorecard.error_by_event
            .filter((row) => row.model === scorecard.event_model)
            .map((row) => ({
              key: row.bucket,
              label: EVENT_LABELS[row.bucket] ?? row.bucket,
              value: row.mae,
              note: `n=${row.n.toLocaleString()}`,
            }))}
          unit="MAE, points"
        />
      </section>

      <section className={styles.panel}>
        <h2 className={styles.panelTitle}>Which head is wrong</h2>
        <p className={styles.panelSub}>
          The projection is a sum of heads, and this is each head&rsquo;s own error. Goals and
          minutes dominate — which is where a decomposed model earns its keep, because you can
          see that rather than infer it.
        </p>
        <Bars
          rows={scorecard.component_decomposition.map((row) => ({
            key: row.component,
            label: row.component.replace(/_/g, " "),
            value: row.mae,
            note: "",
          }))}
          unit="MAE, points"
        />
      </section>

      <section className={styles.panel}>
        <h2 className={styles.panelTitle}>The minutes head</h2>
        <p className={styles.panelSub}>
          Brier scores for the three minutes outcomes, over{" "}
          <span className="data">{scorecard.minutes_head.n.toLocaleString()}</span>{" "}
          player-gameweeks. Lower is better; 0.25 is what a coin flip scores.
        </p>
        <ul className={styles.stats}>
          <Stat label="P(no minutes)" value={scorecard.minutes_head.brier_blank} />
          <Stat label="P(1–59)" value={scorecard.minutes_head.brier_short} />
          <Stat label="P(60+)" value={scorecard.minutes_head.brier_full} />
          <Stat
            label="MAE, expected minutes"
            value={scorecard.minutes_head.mae_expected_minutes}
            digits={2}
          />
        </ul>
      </section>

      {board && <BoardAccuracy board={board} />}
    </main>
  );
}

/**
 * §5.4.7, new in v2: "Model Board accuracy gets its own panel. If the app
 * is going to classify players as rising, it must report how often rising
 * players subsequently outperformed."
 *
 * It renders on the board itself too, and deliberately: a reader who
 * never opens the Scorecard must still meet it. Here it sits among the
 * model's other measured results, which is where it belongs — the
 * classification is a model output like any other and is scored like one.
 */
function BoardAccuracy({ board }: { board: BoardFile }) {
  return (
    <section className={styles.panel}>
      <h2 className={styles.panelTitle}>What the board&rsquo;s buckets were worth</h2>
      <p className={styles.panelSub}>
        Mean points in the following gameweeks, against the players each bucket was picked out
        from. Two of the four are negative.
      </p>
      <Bars
        rows={board.bucket_accuracy.map((row) => ({
          key: row.bucket,
          label: row.bucket,
          value: row.lift,
          note: `n=${row.n.toLocaleString()} · vs ${row.comparison}`,
        }))}
        unit="forward points, lift"
        diverging
      />
      <p className={styles.finding}>
        <span className="data">Rising</span> and <span className="data">declining</span> both
        measured worse than the players they were picked out from. The trend buckets carry no
        edge in this data, and the board says so on its own surface rather than only here.
      </p>
    </section>
  );
}

/** Within-position Spearman, because a pooled rho flatters (§5.7.1). */
function ByPosition({ rows, eventModel }: { rows: ScorecardRow[]; eventModel: string }) {
  const event = rows.find((row) => row.model === eventModel);
  if (!event || event.spearman_by_position.length === 0) return null;

  return (
    <section className={styles.panel}>
      <h2 className={styles.panelTitle}>Ranking, within each position</h2>
      <p className={styles.panelSub}>
        Spearman is carried per position rather than pooled, because pooling is the §5.7.1
        distortion in its original form: a model that only knew forwards outscore defenders
        would post a flattering pooled figure while ranking nobody correctly inside the group
        anyone actually picks from.
      </p>
      <Bars
        rows={event.spearman_by_position.map((row) => ({
          key: row.position,
          label: row.position,
          value: row.rho,
          note: `n=${row.n.toLocaleString()}`,
        }))}
        unit="Spearman rho"
      />
    </section>
  );
}

/**
 * §5.4.7: "Calibration curves **with the diagonal drawn**."
 *
 * The diagonal is the point of the chart. Without it a reader has to hold
 * "predicted equals actual" in their head while reading two axes, and the
 * whole question is how far the curve sits from it.
 */
function Calibration({ file }: { file: ScorecardFile }) {
  const [model, setModel] = useState(file.event_model);
  const bins = file.calibration
    .filter((row) => row.model === model)
    .filter((row) => row.mean_prediction !== null && row.mean_actual !== null)
    .sort((left, right) => left.mean_prediction! - right.mean_prediction!);

  const W = 320;
  const H = 320;
  const PAD = 42;

  const values = bins.flatMap((row) => [row.mean_prediction!, row.mean_actual!]);
  const lo = Math.min(...values, 0);
  const hi = Math.max(...values, 1);
  const px = (value: number) => PAD + ((value - lo) / (hi - lo || 1)) * (W - PAD * 1.4);
  const py = (value: number) => H - PAD - ((value - lo) / (hi - lo || 1)) * (H - PAD * 1.4);

  return (
    <section className={styles.panel}>
      <div className={styles.panelHead}>
        <div>
          <h2 className={styles.panelTitle}>Calibration</h2>
          <p className={styles.panelSub}>
            Mean prediction against mean outcome, by prediction bin. On the diagonal means a
            projection of 6 points really does average 6.
          </p>
        </div>
        <label className={styles.selector}>
          <span>Model</span>
          <select value={model} onChange={(pick) => setModel(pick.target.value)}>
            {file.models.map((entry) => (
              <option key={entry} value={entry}>
                {MODEL_LABELS[entry] ?? entry}
              </option>
            ))}
          </select>
        </label>
      </div>

      {bins.length === 0 ? (
        <p className={styles.muted}>No calibration bins for this model.</p>
      ) : (
        <svg
          className={styles.plot}
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`Calibration for ${MODEL_LABELS[model] ?? model}, ${bins.length} bins`}
        >
          <line x1={PAD} y1={H - PAD} x2={W} y2={H - PAD} className={styles.axis} />
          <line x1={PAD} y1={0} x2={PAD} y2={H - PAD} className={styles.axis} />

          {/* The diagonal §5.4.7 asks for, drawn before the data. */}
          <line
            x1={px(lo)}
            y1={py(lo)}
            x2={px(hi)}
            y2={py(hi)}
            className={styles.diagonal}
          />

          <path
            d={bins
              .map(
                (row, index) =>
                  `${index === 0 ? "M" : "L"}${px(row.mean_prediction!).toFixed(1)},${py(
                    row.mean_actual!,
                  ).toFixed(1)}`,
              )
              .join(" ")}
            className={styles.curve}
          />

          {bins.map((row) => (
            <circle
              key={row.bin}
              cx={px(row.mean_prediction!)}
              cy={py(row.mean_actual!)}
              r={3}
              className={styles.point}
            >
              <title>
                bin {row.bin}: predicted {row.mean_prediction!.toFixed(3)}, actual{" "}
                {row.mean_actual!.toFixed(3)} over n={row.n.toLocaleString()}
              </title>
            </circle>
          ))}

          <text x={PAD + (W - PAD) / 2} y={H - 8} className={styles.axisLabel} textAnchor="middle">
            mean prediction
          </text>
          <text
            x={12}
            y={(H - PAD) / 2}
            className={styles.axisLabel}
            textAnchor="middle"
            transform={`rotate(-90 12 ${(H - PAD) / 2})`}
          >
            mean actual
          </text>
        </svg>
      )}
    </section>
  );
}

// --- small shared pieces ----------------------------------------------

interface BarRow {
  key: string;
  label: string;
  value: number | null;
  note: string;
}

/** A labelled bar list. Nulls render as an em dash, never a zero-length bar. */
function Bars({
  rows,
  unit,
  diverging = false,
}: {
  rows: BarRow[];
  unit: string;
  diverging?: boolean;
}) {
  const extent = Math.max(...rows.map((row) => Math.abs(row.value ?? 0)), 1e-9);

  return (
    <ul className={styles.bars} aria-label={unit}>
      {rows.map((row) => {
        const positive = (row.value ?? 0) >= 0;
        return (
          <li key={row.key} className={styles.barRow}>
            <span className={styles.barLabel}>{row.label}</span>
            <span className={styles.barTrack} data-diverging={diverging || undefined}>
              {diverging && <span className={styles.barZero} aria-hidden="true" />}
              {row.value !== null && (
                <span
                  className={styles.barFill}
                  data-negative={!positive || undefined}
                  style={
                    diverging
                      ? {
                          inlineSize: `${(Math.abs(row.value) / extent) * 50}%`,
                          [positive ? "insetInlineStart" : "insetInlineEnd"]: "50%",
                          position: "absolute",
                          insetBlock: "2px",
                        }
                      : { inlineSize: `${(Math.abs(row.value) / extent) * 100}%` }
                  }
                />
              )}
            </span>
            <span className={`${styles.barValue} data`}>
              {row.value === null
                ? "—"
                : `${diverging && positive ? "+" : diverging ? "−" : ""}${Math.abs(row.value).toFixed(3)}`}
            </span>
            <span className={styles.barNote}>{row.note}</span>
          </li>
        );
      })}
    </ul>
  );
}

function Stat({
  label,
  value,
  digits = 5,
}: {
  label: string;
  value: number | null;
  digits?: number;
}) {
  return (
    <li className={styles.stat}>
      <span className={`${styles.statValue} data`}>{fmt(value, digits)}</span>
      <span className={styles.statLabel}>{label}</span>
    </li>
  );
}

/** §5.3.3: a null is an em dash. */
function fmt(value: number | null | undefined, digits: number): string {
  if (value === null || value === undefined) return "—";
  return value.toFixed(digits);
}

/** The strongest non-event model, which is what the margin is measured against. */
function bestBaseline(rows: ScorecardRow[], eventModel: string): ScorecardRow | null {
  const baselines = rows.filter((row) => row.model !== eventModel && row.mae !== null);
  if (baselines.length === 0) return null;
  return baselines.reduce((best, row) => (row.mae! < best.mae! ? row : best));
}

function Loading({ progress }: { progress: LoadProgress | null }) {
  const kb = (bytes: number) => `${Math.round(bytes / 1024)} KB`;
  const pct =
    progress?.total != null ? Math.round((progress.received / progress.total) * 100) : null;
  return (
    <main className={styles.scorecard}>
      <h1 className={styles.title}>Model Scorecard</h1>
      <div className={styles.loading}>
        <div
          className={styles.track}
          role="progressbar"
          aria-valuenow={pct ?? undefined}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Loading the scorecard"
        >
          <div className={styles.fill} style={{ inlineSize: pct === null ? "30%" : `${pct}%` }} />
        </div>
        <p className={`${styles.sub} data`}>
          {progress
            ? `scorecard.json — ${kb(progress.received)}${progress.total ? ` of ${kb(progress.total)}` : ""}`
            : "requesting scorecard.json"}
        </p>
      </div>
    </main>
  );
}

function Failed({ error }: { error: Error }) {
  return (
    <main className={styles.scorecard}>
      <h1 className={styles.title}>Model Scorecard</h1>
      <div className={styles.failure} role="alert">
        <p className={styles.failureHead}>scorecard.json did not load.</p>
        <p className={`${styles.sub} data`}>{error.message}</p>
        <p className={styles.sub}>
          Run <code className="data">python -m web.export scorecard</code> and reload.
        </p>
      </div>
    </main>
  );
}
