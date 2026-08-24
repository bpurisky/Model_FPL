/**
 * §5.4.4 — Player Comparison.
 *
 * > "Component decomposition bars — projection broken into heads
 * > (appearance/minutes, goals, assists, clean sheets, defensive
 * > contribution, saves, bonus) side by side. Mirrors
 * > `analytics/evaluate.py`'s existing decomposition; the most
 * > diagnostically useful comparison in the app."
 *
 * Reads `players.json` and nothing else, so unlike the Graph Builder and
 * the Form Matrix it works on a fresh clone with no pipeline run
 * (§5.14.8). That is why it is not behind the panel reader.
 *
 * Two rules here are stated as prohibitions and both are obeyed:
 *
 * **The minutes distribution is never collapsed to a mean.** §5.4.4:
 * "P(blank) / P(short) / P(60+) as a stacked bar, never collapsed to a
 * mean. The model deliberately refuses to produce a mean minutes figure;
 * the UI must not reintroduce one." A player who is 50/50 to start and a
 * player nailed on for 60 minutes can share an expected-minutes number
 * and are not remotely the same asset.
 *
 * **Mixed-position comparison defaults to percentile-within-position.**
 * §5.7.3 and §5.4.4 agree on this, and §5.7.5's caution renders whenever
 * raw units are shown across positions. The percentiles are read from the
 * export; §5.6 forbids computing one here, and `players.json` carries
 * them per metric alongside the population size they were computed
 * against.
 *
 * The radar chart §5.4.4 *permits* is not built. It is permitted rather
 * than required, it is only allowed for same-position normalized rates,
 * and a set of aligned percentile bars answers the same question while
 * staying readable at four players. Not a deviation — an option declined.
 */

import { useEffect, useMemo, useState } from "react";
import { useApp } from "../app/state";
import { BucketBadge } from "../components/BucketBadge";
import { Provenance } from "../components/Provenance";
import { useBoard } from "../data/useBoard";
import { loadColumns, loadPlayers, type LoadProgress } from "../data/load";
import type {
  BoardFile,
  ColumnsFile,
  ColumnSpec,
  PlayerRow,
  PlayersFile,
} from "../data/schema";
import styles from "./Comparison.module.css";

type State =
  | { status: "loading"; progress: LoadProgress | null }
  | { status: "error"; error: Error }
  | { status: "ready"; players: PlayersFile; columns: ColumnsFile };

/** §5.4.4: "Select 2–4 players." Four is where grouped bars stop reading. */
const MAX_PLAYERS = 4;

/**
 * The projection heads, in the order they read as a story: can he play,
 * what does he do when he does, what does his defence do, what does the
 * scoring system add on top.
 */
const HEAD_LABELS: Record<string, string> = {
  minutes: "Appearance",
  goals: "Goals",
  assists: "Assists",
  clean_sheets: "Clean sheets",
  goals_conceded: "Conceding",
  saves: "Saves",
  defensive_contribution: "Def. contribution",
  cards_and_other: "Cards and other",
  bonus: "Bonus",
};

const HEAD_ORDER = Object.keys(HEAD_LABELS);

/** One colour per player, sampled off the diverging poles (§5.8.2). */
const PLAYER_COLORS = [
  "var(--rho-neg)",
  "var(--rho-pos)",
  "color-mix(in oklch, var(--rho-neg) 55%, var(--paper))",
  "color-mix(in oklch, var(--rho-pos) 55%, var(--paper))",
];

export function Comparison() {
  const { state, dispatch } = useApp();
  const [data, setData] = useState<State>({ status: "loading", progress: null });
  const [query, setQuery] = useState("");
  const [rawAcrossPositions, setRawAcrossPositions] = useState(false);
  /* §5.5.4's reverse path: what the model made of each chosen player. */
  const board = useBoard();

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

  const index = useMemo(() => {
    if (data.status !== "ready") return new Map<number, PlayerRow>();
    return new Map(data.players.players.map((player) => [player.element_id, player]));
  }, [data]);

  const columnIndex = useMemo(() => {
    if (data.status !== "ready") return new Map<string, ColumnSpec>();
    return new Map(data.columns.columns.map((column) => [column.key, column]));
  }, [data]);

  const chosen = useMemo(
    () =>
      state.selection
        .map((id) => index.get(id))
        .filter((player): player is PlayerRow => player !== undefined)
        .slice(0, MAX_PLAYERS),
    [state.selection, index],
  );

  /*
   * §5.4.4 and §5.7.3: raw for same-position, percentile-within-position
   * for mixed. The toggle exists either way — §5.7.3 is explicit that
   * "raw values are never hidden", because "which defenders get forward
   * most" is a real question best answered in raw xGI.
   */
  const positions = new Set(chosen.map((player) => player.position));
  const mixed = positions.size > 1;
  const showRaw = mixed ? rawAcrossPositions : true;

  const matches = useMemo(() => {
    if (data.status !== "ready") return [];
    const needle = query.trim().toLowerCase();
    if (needle.length < 2) return [];
    return data.players.players
      .filter(
        (player) =>
          player.name.toLowerCase().includes(needle) ||
          player.team.toLowerCase().includes(needle),
      )
      .slice(0, 40);
  }, [data, query]);

  if (data.status === "loading") return <Loading progress={data.progress} />;
  if (data.status === "error") return <Failed error={data.error} />;

  const { players } = data;

  return (
    <main className={styles.compare}>
      <header className={styles.head}>
        <div>
          <h1 className={styles.title}>Comparison</h1>
          <p className={styles.sub}>
            Where a projection comes from, head by head. {players.season} gameweek{" "}
            <span className="data">{players.gameweek}</span>, projecting{" "}
            <span className="data">{players.projected_gameweek}</span> on a{" "}
            <span className="data">{players.projection_basis.replace("_", " ")}</span> basis.
          </p>
        </div>
        <Provenance header={players.header} basis={players.header.normalization_basis} />
      </header>

      <Picker
        query={query}
        onQuery={setQuery}
        matches={matches}
        chosen={chosen}
        onToggle={(id) => dispatch({ type: "toggleSelect", id })}
        full={chosen.length >= MAX_PLAYERS}
      />

      {chosen.length === 0 && (
        <p className={styles.empty}>
          Search above, or click a player in the Form Matrix. Two to {MAX_PLAYERS} at a time —
          past four, grouped bars stop being comparable and start being a legend.
        </p>
      )}

      {chosen.length > 0 && (
        <>
          {mixed && !showRaw && (
            <p className={styles.basis} role="note">
              Comparing across positions, so rate metrics show{" "}
              <strong>percentile within position</strong>: each player is placed against his own
              group, read from the export. Raw values are one click away and are never hidden.
            </p>
          )}

          {mixed && showRaw && (
            /*
             * §5.7.5's caution, in the register §5.8.7 reserves for it.
             * "This is the highest-value single piece of copy in the
             * application. It is the moment the tool teaches."
             */
            <p className={styles.caution} role="note">
              Raw rates across positions — forwards will out-shoot defenders on every attacking
              metric here regardless of how good either is at his own job. The comparison that
              means something is percentile within position.
            </p>
          )}

          <Decomposition players={chosen} board={board} />

          <Minutes players={chosen} />

          <Rates
            players={chosen}
            columns={columnIndex}
            population={players.population}
            showRaw={showRaw}
            mixed={mixed}
            onToggleRaw={() => setRawAcrossPositions((current) => !current)}
          />
        </>
      )}
    </main>
  );
}

// --- the picker ------------------------------------------------------

interface PickerProps {
  query: string;
  onQuery: (value: string) => void;
  matches: PlayerRow[];
  chosen: PlayerRow[];
  onToggle: (id: number) => void;
  full: boolean;
}

function Picker({ query, onQuery, matches, chosen, onToggle, full }: PickerProps) {
  return (
    <section className={styles.picker}>
      <div className={styles.chosen}>
        {chosen.map((player, index) => (
          <button
            key={player.element_id}
            type="button"
            className={styles.chip}
            style={{ borderColor: PLAYER_COLORS[index] }}
            onClick={() => onToggle(player.element_id)}
            title="Remove"
          >
            <span className={styles.swatch} style={{ background: PLAYER_COLORS[index] }} />
            {player.name}
            <span className={styles.chipMeta}>
              {player.team} · {player.position}
            </span>
            <span aria-hidden="true">×</span>
          </button>
        ))}
      </div>

      <label className={styles.searchLabel}>
        <span className={styles.searchText}>Add a player</span>
        <input
          type="search"
          className={styles.search}
          placeholder="name or club"
          value={query}
          onChange={(event) => onQuery(event.target.value)}
          disabled={full}
        />
      </label>

      {full && <p className={styles.hint}>Four is the limit. Remove one to add another.</p>}

      {!full && matches.length > 0 && (
        <ul className={styles.matches}>
          {matches.map((player) => (
            <li key={player.element_id}>
              <button
                type="button"
                className={styles.match}
                onClick={() => onToggle(player.element_id)}
              >
                <span>{player.name}</span>
                <span className={styles.matchMeta}>
                  {player.team} · {player.position} ·{" "}
                  {player.price === null ? "—" : `£${(player.price / 10).toFixed(1)}m`}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

// --- component decomposition ----------------------------------------

/**
 * §5.4.4's "most diagnostically useful comparison in the app": the
 * projection broken into the heads that produced it, side by side.
 *
 * Grouped rather than stacked. A stack shows each player's total
 * correctly and makes the heads impossible to compare across players,
 * which is the entire question — and the totals are printed anyway.
 */
function Decomposition({
  players,
  board,
}: {
  players: PlayerRow[];
  board: BoardFile | null;
}) {
  const heads = HEAD_ORDER.filter((head) =>
    players.some((player) => {
      const value = player.projection?.components[head];
      return value !== null && value !== undefined && Math.abs(value) > 0.0005;
    }),
  );

  const missing = players.filter((player) => player.projection === null);

  const extent = Math.max(
    ...players.flatMap((player) =>
      heads.map((head) => Math.abs(player.projection?.components[head] ?? 0)),
    ),
    0.5,
  );

  return (
    <section className={styles.panel}>
      <header className={styles.panelHead}>
        <h2 className={styles.panelTitle}>Projection, head by head</h2>
        <p className={styles.panelSub}>
          What the model expects each head to contribute. The totals below are the sum of these
          by construction — nothing else is added.
        </p>
      </header>

      {missing.length > 0 && (
        <p className={styles.note}>
          {missing.map((player) => player.name).join(", ")} carr
          {missing.length === 1 ? "ies" : "y"} no projection: the model had not spoken for{" "}
          {missing.length === 1 ? "him" : "them"} at this gameweek. That is not a zero.
        </p>
      )}

      <div className={styles.totals}>
        {players.map((player, index) => (
          <div key={player.element_id} className={styles.total}>
            <span className={styles.totalName} style={{ color: PLAYER_COLORS[index] }}>
              {player.name}
            </span>
            <span className={`${styles.totalValue} data`}>
              {player.projection?.total === null || player.projection === null
                ? "—"
                : player.projection.total.toFixed(2)}
            </span>
            <span className={styles.totalUnit}>projected points</span>
            <BucketBadge board={board} elementId={player.element_id} />
          </div>
        ))}
      </div>

      <div className={styles.heads}>
        {heads.map((head) => (
          <div key={head} className={styles.headRow}>
            <span className={styles.headLabel}>{HEAD_LABELS[head] ?? head}</span>
            <div className={styles.headBars}>
              {players.map((player, index) => {
                const value = player.projection?.components[head] ?? null;
                return (
                  <div key={player.element_id} className={styles.headBar}>
                    <div className={styles.barTrack}>
                      {value !== null && (
                        <div
                          className={styles.barFill}
                          style={{
                            inlineSize: `${(Math.abs(value) / extent) * 100}%`,
                            background: PLAYER_COLORS[index],
                            // A negative head — conceding, cards — grows
                            // from the same edge but reads as a deficit.
                            opacity: value < 0 ? 0.5 : 1,
                          }}
                          title={`${player.name}: ${value.toFixed(3)}`}
                        />
                      )}
                    </div>
                    <span className={`${styles.barValue} data`}>
                      {value === null ? "—" : value.toFixed(2)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

// --- minutes distribution -------------------------------------------

/**
 * §5.4.4: "P(blank) / P(short) / P(60+) as a stacked bar, **never
 * collapsed to a mean**. The model deliberately refuses to produce a mean
 * minutes figure; the UI must not reintroduce one."
 *
 * So there is no expected-minutes number on this panel, and there is no
 * arithmetic here that could produce one. A player who is 50/50 to start
 * and a player nailed on for an hour can share an expected figure and are
 * not the same asset at all.
 */
function Minutes({ players }: { players: PlayerRow[] }) {
  const bands = [
    { key: "p_full", label: "60+ minutes", color: "var(--rho-neg)" },
    { key: "p_short", label: "1–59 minutes", color: "var(--flag)" },
    { key: "p_blank", label: "No minutes", color: "var(--rule)" },
  ] as const;

  return (
    <section className={styles.panel}>
      <header className={styles.panelHead}>
        <h2 className={styles.panelTitle}>Minutes</h2>
        <p className={styles.panelSub}>
          Three outcomes, not an average. The model refuses to produce a mean minutes figure
          because a coin-flip starter and a nailed-on sixty would share it.
        </p>
      </header>

      <ul className={styles.legendRow}>
        {bands.map((band) => (
          <li key={band.key} className={styles.legendItem}>
            <span className={styles.swatch} style={{ background: band.color }} />
            {band.label}
          </li>
        ))}
      </ul>

      <div className={styles.minutes}>
        {players.map((player) => {
          const projection = player.projection;
          const values = bands.map((band) => ({
            ...band,
            value: projection ? (projection[band.key] ?? null) : null,
          }));
          const known = values.every((band) => band.value !== null);
          return (
            <div key={player.element_id} className={styles.minutesRow}>
              <span className={styles.minutesName}>{player.name}</span>
              {known ? (
                <>
                  <div className={styles.stack}>
                    {values.map((band) => (
                      <div
                        key={band.key}
                        className={styles.stackPart}
                        style={{
                          inlineSize: `${band.value! * 100}%`,
                          background: band.color,
                        }}
                        title={`${band.label}: ${(band.value! * 100).toFixed(1)}%`}
                      />
                    ))}
                  </div>
                  <span className={`${styles.minutesValue} data`}>
                    {values.map((band) => `${Math.round(band.value! * 100)}`).join(" / ")}
                  </span>
                </>
              ) : (
                <span className={styles.note}>
                  No minutes model for this player at this gameweek.
                </span>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

// --- position-relative rate bars -------------------------------------

interface RatesProps {
  players: PlayerRow[];
  columns: ReadonlyMap<string, ColumnSpec>;
  population: Record<string, Record<string, number>>;
  showRaw: boolean;
  mixed: boolean;
  onToggleRaw: () => void;
}

/**
 * §5.4.4's "position-relative bars", and §5.7.4's requirement that a
 * normalized number renders its basis.
 *
 * The percentiles come from `players.json`; §5.6 forbids computing one
 * here, and the file carries the population each was measured against so
 * the tooltip can say "n=147" rather than implying a universal scale.
 */
function Rates({ players, columns, population, showRaw, mixed, onToggleRaw }: RatesProps) {
  const keys = useMemo(() => {
    const seen = new Set<string>();
    for (const player of players) {
      for (const [key, metric] of Object.entries(player.metrics)) {
        if (metric.value !== null || metric.percentile !== null) seen.add(key);
      }
    }
    return [...seen].sort((left, right) =>
      (columns.get(left)?.label ?? left).localeCompare(columns.get(right)?.label ?? right),
    );
  }, [players, columns]);

  return (
    <section className={styles.panel}>
      <header className={styles.panelHead}>
        <div>
          <h2 className={styles.panelTitle}>
            Trailing rates{showRaw ? "" : ", as position percentiles"}
          </h2>
          <p className={styles.panelSub}>
            {showRaw
              ? "Raw per-90 rates over the season to date."
              : "Where each player sits inside his own position group, read from the export."}
          </p>
        </div>
        <button type="button" className={styles.toggle} onClick={onToggleRaw} disabled={!mixed}>
          {showRaw ? "Show percentiles" : "Show raw"}
          {!mixed && <span className={styles.toggleWhy}>same position — raw is comparable</span>}
        </button>
      </header>

      <div className={styles.rates}>
        {keys.map((key) => {
          const spec = columns.get(key);
          return (
            <div key={key} className={styles.rateRow}>
              <span className={styles.rateLabel} title={spec?.definition}>
                {spec?.label ?? key}
                {spec?.higher_is_better === false && (
                  <span className={styles.lower} title="Lower is better for this metric">
                    ↓
                  </span>
                )}
              </span>
              <div className={styles.rateBars}>
                {players.map((player, index) => {
                  const metric = player.metrics[key];
                  const n = population[player.position]?.[key] ?? null;
                  const value = showRaw ? (metric?.value ?? null) : (metric?.percentile ?? null);
                  const width =
                    value === null ? 0
                    : showRaw ? scaleRaw(value, key, players)
                    : value * 100;
                  return (
                    <div key={player.element_id} className={styles.rateBar}>
                      <div className={styles.barTrack}>
                        {value !== null && (
                          <div
                            className={styles.barFill}
                            style={{
                              inlineSize: `${Math.max(width, 1)}%`,
                              background: PLAYER_COLORS[index],
                            }}
                          />
                        )}
                      </div>
                      <span className={`${styles.barValue} data`}>
                        {/*
                         * §5.3.3: a null percentile is "below the minutes
                         * floor, so the export declined to place him",
                         * which is not the same as being at the bottom.
                         */}
                        {value === null
                          ? "—"
                          : showRaw
                            ? value.toFixed(2)
                            : `${Math.round(value * 100)}`}
                      </span>
                      {/* §5.7.4: the basis, in the model's own vocabulary. */}
                      <span className={styles.rateBasis}>
                        {value === null
                          ? "below the minutes floor"
                          : showRaw
                            ? "per 90"
                            : `pct in ${player.position}${n ? `, n=${n}` : ""}`}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

/**
 * Raw bars are scaled against the largest value among the players on
 * screen, which is the only defensible reference when the axis is a rate
 * with no natural maximum. Percentile bars need no such trick — they
 * already run 0 to 100 and mean the same thing on every row.
 */
function scaleRaw(value: number, key: string, players: PlayerRow[]): number {
  const extent = Math.max(
    ...players.map((player) => Math.abs(player.metrics[key]?.value ?? 0)),
    1e-9,
  );
  return (Math.abs(value) / extent) * 100;
}

// --- states -----------------------------------------------------------

function Loading({ progress }: { progress: LoadProgress | null }) {
  const kb = (bytes: number) => `${Math.round(bytes / 1024)} KB`;
  const pct =
    progress?.total != null ? Math.round((progress.received / progress.total) * 100) : null;
  return (
    <main className={styles.compare}>
      <h1 className={styles.title}>Comparison</h1>
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
    <main className={styles.compare}>
      <h1 className={styles.title}>Comparison</h1>
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
