/**
 * §5.4.8 — the Trend Explorer.
 *
 * > "Per-player time series from the distilled shards: price, ownership,
 * > projection. Multi-select overlay. Deadline markers on the x-axis.
 * > Position-normalized overlay option."
 *
 * The one surface whose data is a *market* rather than a match. Price and
 * ownership move between deadlines, driven by what other managers do, and
 * `timeseries.parquet` is the collector's own snapshot history rather
 * than anything the model produced.
 *
 * **It is a build artifact and §5.3.4 does not commit it**, so §5.14.8
 * names this among the three routes that must show an explanatory empty
 * state rather than an error or a blank. On a fresh clone that state is
 * the normal case, not the failure case, and the copy says so.
 *
 * §5.4.8's "position-normalized overlay option" is not offered, and the
 * reason is §5.6 rather than schedule: the series here are `now_cost` and
 * `selected_by_percent`, and the export emits no within-position
 * companions for either. Normalizing them in the browser is precisely
 * what the rule forbids. Recorded as §5.16 deviation D12.
 */

import { useEffect, useMemo, useState } from "react";
import { useApp } from "../app/state";
import { Provenance } from "../components/Provenance";
import { loadColumns, type LoadProgress } from "../data/load";
import type { ColumnsFile } from "../data/schema";
import { openSeries, SeriesMissingError, type SeriesPoint } from "../query/series";
import styles from "./TrendExplorer.module.css";
import { count, noun } from "../design/text";

type State =
  | { status: "loading"; progress: LoadProgress | null }
  | { status: "absent" }
  | { status: "error"; error: Error }
  | { status: "ready"; rows: SeriesPoint[]; columns: ColumnsFile };

/** The three series §5.4.8 names, and what each is measured in. */
const SERIES = [
  { key: "now_cost" as const, label: "Price", format: (v: number) => `£${(v / 10).toFixed(1)}m` },
  {
    key: "selected_by_percent" as const,
    label: "Ownership",
    format: (v: number) => `${v.toFixed(1)}%`,
  },
  {
    key: "model_projection" as const,
    label: "Model projection",
    format: (v: number) => `${v.toFixed(2)} pts`,
  },
] as const;

type SeriesKey = (typeof SERIES)[number]["key"];

export function TrendExplorer() {
  const { state, dispatch } = useApp();
  const [data, setData] = useState<State>({ status: "loading", progress: null });
  const [metric, setMetric] = useState<SeriesKey>("now_cost");
  const [query, setQuery] = useState("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const columns = await loadColumns();
        const rows = await openSeries((progress) =>
          cancelled ? undefined : setData({ status: "loading", progress }),
        );
        if (!cancelled) setData({ status: "ready", rows, columns });
      } catch (error) {
        if (cancelled) return;
        setData(
          error instanceof SeriesMissingError
            ? { status: "absent" }
            : { status: "error", error: error as Error },
        );
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const players = useMemo(() => {
    if (data.status !== "ready") return new Map<number, { name: string; team: string }>();
    const seen = new Map<number, { name: string; team: string }>();
    for (const row of data.rows) {
      if (!seen.has(row.element_id)) seen.set(row.element_id, { name: row.name, team: row.team });
    }
    return seen;
  }, [data]);

  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (needle.length < 2) return [];
    return [...players.entries()]
      .filter(
        ([, who]) =>
          who.name.toLowerCase().includes(needle) || who.team.toLowerCase().includes(needle),
      )
      .slice(0, 30);
  }, [players, query]);

  if (data.status === "loading") return <Loading progress={data.progress} />;
  if (data.status === "absent") return <SeriesAbsent />;
  if (data.status === "error") return <Failed error={data.error} />;

  const chosen = state.selection.filter((id) => players.has(id)).slice(0, 6);
  const series = SERIES.find((entry) => entry.key === metric)!;

  return (
    <main className={styles.trend}>
      <header className={styles.head}>
        <div>
          <h1 className={styles.title}>Trend Explorer</h1>
          <p className={styles.sub}>
            Price, ownership and projection over the collector&rsquo;s snapshot history. These
            move between deadlines and are driven by what other managers do — the only surface
            here describing a market rather than a match.
          </p>
          {/*
           * A trap worth naming rather than leaving to be discovered.
           * This surface reads the *current* season's snapshots, while
           * the panel-backed and players.json surfaces still describe the
           * last completed one — and FPL reissues element ids at every
           * rollover. So a player selected there and a player selected
           * here can share an id and be two different footballers. The
           * names on screen always come from this file, so nothing here
           * is mislabelled; only the carry-across is meaningless, and it
           * stops being so once the panel carries the current season too.
           */}
          <p className={styles.note}>
            Ids here belong to the current season. Until the panel carries it too, a selection
            made on another surface may resolve to a different player — the names below are
            always this file&rsquo;s own.
          </p>
        </div>
        <Provenance
          header={data.columns.header}
          basis={data.columns.header.normalization_basis}
        />
      </header>

      <div className={styles.controls}>
        <label className={styles.searchLabel}>
          <span className={styles.searchText}>Add a player</span>
          <input
            type="search"
            className={styles.search}
            placeholder="name or club"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>

        <fieldset className={styles.group}>
          <legend className={styles.legend}>Series</legend>
          <div className={styles.chips}>
            {SERIES.map((entry) => (
              <button
                key={entry.key}
                type="button"
                className={styles.chip}
                data-on={metric === entry.key || undefined}
                onClick={() => setMetric(entry.key)}
              >
                {entry.label}
              </button>
            ))}
          </div>
        </fieldset>

        <p className={styles.count}>
          <span className="data">{data.rows.length.toLocaleString()}</span>{" "}
          {noun(data.rows.length, "snapshot")} over{" "}
          <span className="data">{players.size.toLocaleString()}</span>{" "}
          {noun(players.size, "player")}
        </p>
      </div>

      {matches.length > 0 && (
        <ul className={styles.matches}>
          {matches.map(([id, who]) => (
            <li key={id}>
              <button
                type="button"
                className={styles.match}
                onClick={() => {
                  dispatch({ type: "toggleSelect", id });
                  setQuery("");
                }}
              >
                <span>{who.name}</span>
                <span className={styles.matchMeta}>{who.team}</span>
              </button>
            </li>
          ))}
        </ul>
      )}

      {chosen.length === 0 ? (
        <p className={styles.empty}>
          Search above, or pick players in the Explorer — the selection carries across surfaces.
          Up to six overlay at once.
        </p>
      ) : (
        <Overlay
          rows={data.rows}
          ids={chosen}
          players={players}
          metric={metric}
          label={series.label}
          format={series.format}
          onRemove={(id) => dispatch({ type: "toggleSelect", id })}
        />
      )}
    </main>
  );
}

const COLORS = [
  "var(--rho-neg)",
  "var(--rho-pos)",
  "var(--paper)",
  "color-mix(in oklch, var(--rho-neg) 55%, var(--paper))",
  "color-mix(in oklch, var(--rho-pos) 55%, var(--paper))",
  "var(--muted)",
];

interface OverlayProps {
  rows: SeriesPoint[];
  ids: number[];
  players: Map<number, { name: string; team: string }>;
  metric: SeriesKey;
  label: string;
  format: (value: number) => string;
  onRemove: (id: number) => void;
}

function Overlay({ rows, ids, players, metric, label, format, onRemove }: OverlayProps) {
  const W = 900;
  const H = 340;
  const PAD = { top: 14, right: 16, bottom: 40, left: 62 };

  const lines = ids.map((id) => ({
    id,
    who: players.get(id)!,
    points: rows
      .filter((row) => row.element_id === id && row[metric] !== null)
      .sort((left, right) => left.snapshot_ts - right.snapshot_ts),
  }));

  const every = lines.flatMap((line) => line.points);
  if (every.length === 0) {
    return <p className={styles.empty}>No snapshots carry {label} for these players.</p>;
  }

  const times = every.map((point) => point.snapshot_ts);
  const values = every.map((point) => point[metric] as number);
  const tMin = Math.min(...times);
  const tMax = Math.max(...times);
  const vMin = Math.min(...values);
  const vMax = Math.max(...values);

  const inner = { w: W - PAD.left - PAD.right, h: H - PAD.top - PAD.bottom };
  const px = (t: number) => PAD.left + ((t - tMin) / (tMax - tMin || 1)) * inner.w;
  const py = (v: number) => PAD.top + inner.h - ((v - vMin) / (vMax - vMin || 1)) * inner.h;

  /*
   * §5.4.8's "deadline markers on the x-axis". The snapshots carry the
   * gameweek they were taken in, so a deadline is where that number
   * changes — which is the honest marker available here rather than a
   * date parsed out of a fixture list this surface does not load.
   */
  const deadlines: { at: number; gw: number }[] = [];
  const ordered = [...every].sort((left, right) => left.snapshot_ts - right.snapshot_ts);
  let lastGw: number | null = null;
  for (const point of ordered) {
    if (point.gw !== null && point.gw !== lastGw) {
      if (lastGw !== null) deadlines.push({ at: point.snapshot_ts, gw: point.gw });
      lastGw = point.gw;
    }
  }

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => vMin + (vMax - vMin) * f);

  return (
    <div className={styles.chart}>
      <ul className={styles.legendRow}>
        {lines.map((line, index) => (
          <li key={line.id} className={styles.legendItem}>
            <span className={styles.swatch} style={{ background: COLORS[index % COLORS.length] }} />
            {line.who.name}
            <span className={styles.legendMeta}>{line.who.team}</span>
            <button
              type="button"
              className={styles.remove}
              onClick={() => onRemove(line.id)}
              aria-label={`Remove ${line.who.name}`}
            >
              ×
            </button>
          </li>
        ))}
      </ul>

      <svg
        className={styles.plot}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`${label} over time for ${count(lines.length, "player")}`}
      >
        {ticks.map((value) => (
          <g key={value}>
            <line
              x1={PAD.left}
              y1={py(value)}
              x2={PAD.left + inner.w}
              y2={py(value)}
              className={styles.grid}
            />
            <text x={PAD.left - 8} y={py(value)} className={styles.tick} textAnchor="end" dominantBaseline="central">
              {format(value)}
            </text>
          </g>
        ))}

        {deadlines.map((deadline) => (
          <g key={deadline.at}>
            <line
              x1={px(deadline.at)}
              y1={PAD.top}
              x2={px(deadline.at)}
              y2={PAD.top + inner.h}
              className={styles.deadline}
            />
            <text
              x={px(deadline.at)}
              y={PAD.top + inner.h + 13}
              className={styles.tick}
              textAnchor="middle"
            >
              gw{deadline.gw}
            </text>
          </g>
        ))}

        <line
          x1={PAD.left}
          y1={PAD.top + inner.h}
          x2={PAD.left + inner.w}
          y2={PAD.top + inner.h}
          className={styles.axis}
        />

        {lines.map((line, index) => (
          <g key={line.id}>
            <path
              d={line.points
                .map(
                  (point, i) =>
                    `${i === 0 ? "M" : "L"}${px(point.snapshot_ts).toFixed(1)},${py(
                      point[metric] as number,
                    ).toFixed(1)}`,
                )
                .join(" ")}
              className={styles.line}
              stroke={COLORS[index % COLORS.length]}
            />
            {line.points.map((point) => (
              <circle
                key={point.snapshot_ts}
                cx={px(point.snapshot_ts)}
                cy={py(point[metric] as number)}
                r={1.6}
                fill={COLORS[index % COLORS.length]}
              >
                <title>
                  {line.who.name} — {format(point[metric] as number)} at{" "}
                  {new Date(point.snapshot_ts).toISOString().slice(0, 16).replace("T", " ")}Z
                  {point.gw === null ? "" : `, gw${point.gw}`}
                </title>
              </circle>
            ))}
          </g>
        ))}

        <text
          x={12}
          y={PAD.top + inner.h / 2}
          className={styles.axisLabel}
          textAnchor="middle"
          transform={`rotate(-90 12 ${PAD.top + inner.h / 2})`}
        >
          {label}
        </text>
      </svg>

      <p className={styles.footnote}>
        Snapshot history from the collector&rsquo;s distilled shards, which are delta-only — a
        flat stretch is a value that did not change, not a gap in the record.
      </p>
    </div>
  );
}

// --- states -----------------------------------------------------------

/**
 * §5.14.8's explanatory empty state, and on a fresh clone this is the
 * normal case rather than the failure case.
 */
function SeriesAbsent() {
  return (
    <main className={styles.trend}>
      <h1 className={styles.title}>Trend Explorer</h1>
      <div className={styles.absent}>
        <p className={styles.absentHead}>The time series has not been built.</p>
        <p className={styles.sub}>
          This surface reads <span className="data">timeseries.parquet</span>, distilled from
          the collector&rsquo;s hourly snapshots. §5.3.4 does not commit it — it grows with
          every snapshot and it is reproducible from data that is committed — so a fresh clone
          arrives here rather than at a chart.
        </p>
        <p className={styles.sub}>
          Run <code className="data">python -m web.export timeseries</code> and reload. Every
          other surface except the Graph Builder and the Form Matrix works without it.
        </p>
      </div>
    </main>
  );
}

function Loading({ progress }: { progress: LoadProgress | null }) {
  const kb = (bytes: number) => `${Math.round(bytes / 1024)} KB`;
  const pct =
    progress?.total != null ? Math.round((progress.received / progress.total) * 100) : null;
  return (
    <main className={styles.trend}>
      <h1 className={styles.title}>Trend Explorer</h1>
      <div className={styles.loading}>
        <div
          className={styles.track}
          role="progressbar"
          aria-valuenow={pct ?? undefined}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Loading the time series"
        >
          <div className={styles.fill} style={{ inlineSize: pct === null ? "30%" : `${pct}%` }} />
        </div>
        <p className={`${styles.sub} data`}>
          {progress
            ? `timeseries.parquet — ${kb(progress.received)}${progress.total ? ` of ${kb(progress.total)}` : ""}`
            : "requesting timeseries.parquet"}
        </p>
      </div>
    </main>
  );
}

function Failed({ error }: { error: Error }) {
  return (
    <main className={styles.trend}>
      <h1 className={styles.title}>Trend Explorer</h1>
      <div className={styles.absent} role="alert">
        <p className={styles.absentHead}>The time series did not load.</p>
        <p className={`${styles.sub} data`}>{error.message}</p>
      </div>
    </main>
  );
}
