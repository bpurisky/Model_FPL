/**
 * §5.4.6 — the Model Board. The prescriptive surface, and the only one.
 *
 * > "What this surface must never do: suggest a transfer, name a captain,
 * > propose a squad, or display a price-change prediction as a
 * > recommendation. It classifies players. The user decides."
 *
 * Three things make this surface honest rather than merely opinionated,
 * and all three are non-negotiable:
 *
 * **It is walled off (§5.8.6).** Model-authored surfaces sit on `--panel`
 * with a persistent left rule and a monospace `model_git_sha`
 * attribution; user-driven surfaces sit on `--ground` with no such rule.
 * No new colours, no badge, no icon set — the distinction is structural,
 * which is why it survives greyscale, low vision, and screenshots.
 *
 * **It publishes its own hit rate.** §5.4.7: "If the app is going to
 * classify players as rising, it must report how often rising players
 * subsequently outperformed. A prescriptive surface without a published
 * hit rate is exactly the failure mode this repo exists to avoid." The
 * measured answer is **bad for two of the four buckets** — rising is
 * worth −0.077 forward points against the players it was picked out from,
 * and declining −0.144 — and `bucket_accuracy` travels inside
 * `board.json` precisely so no surface can render "Rising" without
 * rendering what Rising was worth. It renders above the players, not
 * below them.
 *
 * **Every card carries "Explain this" (§5.5.4).** Non-negotiable, and the
 * entire difference between a tool that issues verdicts and one that
 * teaches. It sets the Graph Builder's state and navigates; §5.5 already
 * made all of that state linkable, so the bridge costs a dispatch.
 *
 * §5.4.6 offers "ranked cards **or** a sortable table with per-player
 * sparklines". Cards, because "the user is receiving rather than
 * constructing here" — and because the sparkline variant would make this
 * a panel route, trading a 3.1 MB download for a shape the Form Matrix
 * already draws better.
 */

import { useEffect, useMemo, useState } from "react";
import { useApp } from "../app/state";
import { Provenance } from "../components/Provenance";
import { loadBoard, loadColumns, type LoadProgress } from "../data/load";
import type { BoardFile, BoardPlayer, ColumnsFile, ColumnSpec } from "../data/schema";
import styles from "./ModelBoard.module.css";

type State =
  | { status: "loading"; progress: LoadProgress | null }
  | { status: "error"; error: Error }
  | { status: "ready"; board: BoardFile; columns: ColumnsFile };

/** Ordered as the model means them, not alphabetically. */
const BUCKETS = ["optimal", "rising", "declining", "neutral"] as const;
type Bucket = (typeof BUCKETS)[number];

const BUCKET_BLURB: Record<Bucket, string> = {
  optimal: "Highest composite inside the position, right now.",
  rising: "Underlying metrics trending up across the window, whether or not points have followed.",
  declining: "Underlying metrics trending down, whether or not points have fallen.",
  neutral: "No trend the model is willing to call either way.",
};

const POSITIONS = ["GK", "DEF", "MID", "FWD"];

export function ModelBoard() {
  const { state, dispatch } = useApp();
  const [data, setData] = useState<State>({ status: "loading", progress: null });
  const [bucket, setBucket] = useState<Bucket | "all">("all");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [board, columns] = await Promise.all([
          loadBoard((progress) => (cancelled ? undefined : setData({ status: "loading", progress }))),
          loadColumns(),
        ]);
        if (!cancelled) setData({ status: "ready", board, columns });
      } catch (error) {
        if (!cancelled) setData({ status: "error", error: error as Error });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const columnIndex = useMemo(() => {
    if (data.status !== "ready") return new Map<string, ColumnSpec>();
    return new Map(data.columns.columns.map((column) => [column.key, column]));
  }, [data]);

  const position = state.filters.positions.length === 1 ? state.filters.positions[0]! : "all";

  const visible = useMemo(() => {
    if (data.status !== "ready") return [];
    return data.board.players
      .filter((player) => position === "all" || player.position === position)
      .filter((player) => bucket === "all" || player.bucket === bucket);
  }, [data, position, bucket]);

  if (data.status === "loading") return <Loading progress={data.progress} />;
  if (data.status === "error") return <Failed error={data.error} />;

  const { board } = data;
  const label = (key: string) => columnIndex.get(key)?.label ?? key;

  /**
   * §5.5.4's bridge. Five requirements, all set here and none of them new
   * machinery: the player pre-filtered, the driving metrics on the
   * channels, the gameweek range set to the trend window, normalization
   * within-position, and a caption. The caption is not passed through the
   * URL — the Graph Builder reads `board.json` itself for the reverse
   * path anyway, so it reconstructs the sentence from the same card.
   */
  const explain = (player: BoardPlayer) => {
    const driver = player.drivers[0] ?? null;
    dispatch({
      type: "filters",
      filters: {
        elements: [player.element_id],
        positions: [player.position],
        seasons: [board.season],
        gwMin: Math.max(board.gameweek - board.trend_window + 1, 1),
        gwMax: board.gameweek,
      },
    });
    dispatch({ type: "select", ids: [player.element_id] });
    dispatch({ type: "normalized", normalized: true });
    dispatch({
      type: "encoding",
      encoding: { x: "gw", y: driver, color: null, wrap: null, aggregate: "mean" },
    });
    dispatch({ type: "navigate", view: "graph" });
  };

  return (
    <main className={styles.board}>
      <header className={styles.head}>
        <div>
          <h1 className={styles.title}>Model Board</h1>
          <p className={styles.sub}>
            The model&rsquo;s own ranking, within position and never across it. It classifies
            players; it does not pick a squad, name a captain, or suggest a transfer.
          </p>
        </div>
        <Provenance header={board.header} basis={board.header.normalization_basis} />
      </header>

      <Accuracy board={board} />

      <Weights board={board} label={label} />

      <div className={styles.controls}>
        <fieldset className={styles.group}>
          <legend className={styles.legend}>Position</legend>
          <div className={styles.chips}>
            <button
              type="button"
              className={styles.chip}
              data-on={position === "all" || undefined}
              onClick={() => dispatch({ type: "filters", filters: { positions: [] } })}
            >
              All
            </button>
            {POSITIONS.map((entry) => (
              <button
                key={entry}
                type="button"
                className={styles.chip}
                data-on={position === entry || undefined}
                onClick={() => dispatch({ type: "filters", filters: { positions: [entry] } })}
              >
                {entry}
              </button>
            ))}
          </div>
        </fieldset>

        <fieldset className={styles.group}>
          <legend className={styles.legend}>Bucket</legend>
          <div className={styles.chips}>
            <button
              type="button"
              className={styles.chip}
              data-on={bucket === "all" || undefined}
              onClick={() => setBucket("all")}
            >
              All
            </button>
            {BUCKETS.map((entry) => (
              <button
                key={entry}
                type="button"
                className={styles.chip}
                data-on={bucket === entry || undefined}
                // What the bucket means, in the model's own terms — the
                // first question a reader has about the word "rising".
                title={BUCKET_BLURB[entry]}
                onClick={() => setBucket(entry)}
              >
                {entry}
              </button>
            ))}
          </div>
        </fieldset>

        <p className={styles.count}>
          {visible.length} of <span className="data">{board.players.length}</span> classified
          players, {board.season} after gameweek <span className="data">{board.gameweek}</span>.
        </p>
      </div>

      {visible.length === 0 ? (
        <p className={styles.empty}>
          No player in that combination. The board classifies only players with at least{" "}
          <span className="data">{board.min_gameweeks}</span> gameweeks behind them.
        </p>
      ) : (
        <ul className={styles.cards}>
          {visible.map((player) => (
            <Card
              key={player.element_id}
              player={player}
              board={board}
              label={label}
              onExplain={() => explain(player)}
            />
          ))}
        </ul>
      )}
    </main>
  );
}

// --- one card ---------------------------------------------------------

interface CardProps {
  player: BoardPlayer;
  board: BoardFile;
  label: (key: string) => string;
  onExplain: () => void;
}

/**
 * §5.4.6: "Every card carries, without interaction: the composite score
 * and its within-position percentile, the two or three metrics that drove
 * the classification, named, the trend window used, and an amber flag if
 * the classification rests on fewer than the configured minimum
 * gameweeks or minutes."
 *
 * All five, and none of them behind a hover.
 */
function Card({ player, board, label, onExplain }: CardProps) {
  return (
    <li className={styles.card} data-bucket={player.bucket}>
      <div className={styles.cardHead}>
        <span className={styles.rank}>#{player.rank}</span>
        <div className={styles.identity}>
          <span className={styles.name}>{player.name}</span>
          <span className={styles.meta}>
            {player.team} · {player.position}
          </span>
        </div>
        <span className={styles.bucket}>{player.bucket}</span>
      </div>

      <div className={styles.scores}>
        <div className={styles.score}>
          <span className={`${styles.scoreValue} data`}>
            {player.composite === null ? "—" : player.composite.toFixed(2)}
          </span>
          <span className={styles.scoreLabel}>composite</span>
        </div>
        <div className={styles.score}>
          <span className={`${styles.scoreValue} data`}>
            {/*
             * §5.3.3: a null percentile is "the export declined to place
             * him", which is not the same as placing him last.
             */}
            {player.percentile === null ? "—" : Math.round(player.percentile * 100)}
          </span>
          <span className={styles.scoreLabel}>pct in {player.position}</span>
        </div>
      </div>

      <p className={styles.drivers}>
        Driven by{" "}
        {player.drivers.length === 0 ? (
          <span className={styles.muted}>nothing the model will name</span>
        ) : (
          player.drivers.map((key, index) => (
            <span key={key}>
              {index > 0 && index === player.drivers.length - 1 ? " and " : index > 0 ? ", " : ""}
              <span className="data">{label(key)}</span>
            </span>
          ))
        )}
        , over the last <span className="data">{board.trend_window}</span> gameweeks.
      </p>

      {player.low_confidence && (
        /*
         * §5.4.6's amber flag. §5.8.2 reserves amber for exactly this —
         * a statement about how far a number can be trusted.
         */
        <p className={styles.flag}>
          {player.gameweeks_seen} gameweek{player.gameweeks_seen === 1 ? "" : "s"} behind this,
          against a minimum of {board.min_gameweeks}. Treat the classification as provisional.
        </p>
      )}

      <button type="button" className={styles.explain} onClick={onExplain}>
        Explain this
      </button>
    </li>
  );
}

// --- what the buckets were actually worth ------------------------------

/**
 * §5.4.7's requirement, rendered above the players rather than below
 * them.
 *
 * The numbers are not flattering and that is the point. Two of the four
 * buckets have **negative** lift: a player the model called rising went
 * on to score 0.077 fewer points per gameweek than the players he was
 * picked out from, and declining 0.144 fewer. `neutral` — the bucket that
 * means "no trend the model will call" — outperforms both.
 *
 * §5.14.14 forbids shipping placeholder data, and §5.4.6 requires the
 * buckets, so hiding this was never available. Rendering it is the
 * cheaper and more honest half of the same decision.
 */
function Accuracy({ board }: { board: BoardFile }) {
  const worst = Math.max(...board.bucket_accuracy.map((entry) => Math.abs(entry.lift ?? 0)), 0.1);

  return (
    <section className={styles.panel} aria-labelledby="accuracy-heading">
      <h2 id="accuracy-heading" className={styles.panelTitle}>
        What each bucket was worth
      </h2>
      <p className={styles.panelSub}>
        Measured over the archive: mean points in the following gameweeks, against the players
        each bucket was picked out from. A bucket with negative lift is one the model would
        have been better off not calling.
      </p>

      <ul className={styles.accuracy}>
        {board.bucket_accuracy.map((entry) => {
          const lift = entry.lift;
          const good = (lift ?? 0) > 0;
          return (
            <li key={entry.bucket} className={styles.accuracyRow}>
              <span
                className={styles.accuracyName}
                title={BUCKET_BLURB[entry.bucket as Bucket] ?? undefined}
              >
                {entry.bucket}
              </span>
              <div className={styles.accuracyTrack}>
                <span className={styles.accuracyZero} aria-hidden="true" />
                {lift !== null && (
                  <span
                    className={styles.accuracyFill}
                    data-good={good || undefined}
                    style={{
                      inlineSize: `${(Math.abs(lift) / worst) * 50}%`,
                      [good ? "insetInlineStart" : "insetInlineEnd"]: "50%",
                    }}
                  />
                )}
              </div>
              <span className={`${styles.accuracyValue} data`} data-good={good || undefined}>
                {lift === null ? "—" : `${lift > 0 ? "+" : "−"}${Math.abs(lift).toFixed(3)}`}
              </span>
              <span className={styles.accuracyBasis}>
                {entry.forward_points === null ? "—" : entry.forward_points.toFixed(2)} vs{" "}
                {entry.forward_points_other === null
                  ? "—"
                  : entry.forward_points_other.toFixed(2)}{" "}
                over n={entry.n.toLocaleString()}, against {entry.comparison}
              </span>
            </li>
          );
        })}
      </ul>

      <p className={styles.finding}>
        Read this before reading the cards. <span className="data">Rising</span> and{" "}
        <span className="data">declining</span> both measured worse than the players they were
        picked out from, and <span className="data">neutral</span> beat them — the trend
        buckets carry no edge in this data. <span className="data">Optimal</span>, which ranks
        on level rather than slope, is the one that does.
      </p>
    </section>
  );
}

// --- how this is scored ------------------------------------------------

/**
 * §5.4.6: the weight profiles "are **rendered on screen** in a 'How this
 * is scored' panel — the user must be able to read the model's opinion,
 * not just receive its output." §5.14.12 makes it an acceptance criterion.
 *
 * Negative weights are shown as negative. Conceding is bad for a
 * defender, and a panel that hid the sign would be describing a different
 * model from the one that produced the numbers above.
 */
function Weights({ board, label }: { board: BoardFile; label: (key: string) => string }) {
  const [open, setOpen] = useState(false);

  return (
    <section className={styles.panel}>
      <button
        type="button"
        className={styles.disclosure}
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <h2 className={styles.panelTitle}>How this is scored</h2>
        <span className={styles.disclosureMark} aria-hidden="true">
          {open ? "−" : "+"}
        </span>
      </button>

      {open && (
        <>
          <p className={styles.panelSub}>
            The composite is a weighted sum of within-position z-scores, computed in Python and
            exported. These are the weights, fitted against subsequent points rather than
            chosen — a weight profile that has never been validated is an opinion wearing a
            number&rsquo;s clothing.
          </p>

          <div className={styles.weights}>
            {board.weights.map((profile) => {
              const entries = Object.entries(profile.weights).sort(
                ([, left], [, right]) => Math.abs(right) - Math.abs(left),
              );
              const largest = Math.max(...entries.map(([, value]) => Math.abs(value)), 0.01);
              return (
                <div key={profile.position} className={styles.profile}>
                  <h3 className={styles.profileName}>{profile.position}</h3>
                  <ul className={styles.weightList}>
                    {entries.map(([key, value]) => (
                      <li key={key} className={styles.weightRow}>
                        <span className={styles.weightName}>{label(key)}</span>
                        <span className={styles.weightTrack}>
                          <span
                            className={styles.weightFill}
                            data-negative={value < 0 || undefined}
                            style={{ inlineSize: `${(Math.abs(value) / largest) * 100}%` }}
                          />
                        </span>
                        <span className={`${styles.weightValue} data`}>
                          {value < 0 ? "−" : "+"}
                          {Math.abs(value).toFixed(3)}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              );
            })}
          </div>
        </>
      )}
    </section>
  );
}

// --- states ------------------------------------------------------------

function Loading({ progress }: { progress: LoadProgress | null }) {
  const kb = (bytes: number) => `${Math.round(bytes / 1024)} KB`;
  const pct =
    progress?.total != null ? Math.round((progress.received / progress.total) * 100) : null;
  return (
    <main className={styles.board}>
      <h1 className={styles.title}>Model Board</h1>
      <div className={styles.loading}>
        <div
          className={styles.track}
          role="progressbar"
          aria-valuenow={pct ?? undefined}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Loading the board"
        >
          <div className={styles.fill} style={{ inlineSize: pct === null ? "30%" : `${pct}%` }} />
        </div>
        <p className={`${styles.sub} data`}>
          {progress
            ? `board.json — ${kb(progress.received)}${progress.total ? ` of ${kb(progress.total)}` : ""}`
            : "requesting board.json"}
        </p>
      </div>
    </main>
  );
}

function Failed({ error }: { error: Error }) {
  return (
    <main className={styles.board}>
      <h1 className={styles.title}>Model Board</h1>
      <div className={styles.failure} role="alert">
        <p className={styles.failureHead}>board.json did not load.</p>
        <p className={`${styles.sub} data`}>{error.message}</p>
        <p className={styles.sub}>
          Run <code className="data">python -m web.export board</code> and reload.
        </p>
      </div>
    </main>
  );
}
