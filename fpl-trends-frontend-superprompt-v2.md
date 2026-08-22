# fpl-trends — Phase 5: Frontend Superprompt (v2)

Companion spec to `fpl-trends-superprompt.md`. Section numbers extend that
document (§5.x). Where this spec and the parent conflict, the parent wins and
the conflict is recorded in §5.16.

**Supersedes:** `fpl-trends-frontend-superprompt.md` (v1). v1 remains the
reference for anything this document does not restate. §5.0.4 lists every
place v2 deliberately departs from v1 — read that section before assuming a
v1 rule still holds.

**Prerequisite state:** Phases 0–2 complete (collector, backtest harness,
event model + scoring layer). Phase 3 (squad optimizer) and Phase 4 (paper
trade) are *not* prerequisites — §5.1.3 defines how their surfaces are
stubbed.

---

## §5.0 Purpose and non-goals

### §5.0.1 The two jobs

v1 stated one job. v2 states two, and the ordering between them is load-bearing.

**Job 1 — Is the model right, and where is it wrong?**
A diagnostic instrument for the person who built the model. Every projection
must render alongside enough context to judge how much to trust it.

**Job 2 — Can the user reach their own conclusion, and learn to see what the
model sees?**
A workbench for open-ended exploration of FPL data, plus a path from any
model verdict back to the underlying evidence that produced it.

These are not the same job and they must not be served by the same surface.
Job 1 is *description*: the user drives, the model is silent except as data.
Job 2 has a prescriptive component (§5.4.6) that is deliberately walled off,
visually distinct, and always one click from the evidence behind it.

**The pedagogical constraint that follows:** the model may state a verdict,
but a verdict without a traceable path to its evidence is forbidden. A user
who cannot yet see why a defender is good learns nothing from being told that
he is. §5.5.4 defines the mechanism that enforces this.

### §5.0.2 Explicit non-goals

- **Not a public product.** No accounts, no auth, no multi-user state, no
  analytics/telemetry, no SEO, no marketing page.
- **Not a live-scores site.** In-play polling exists in the collector;
  surfacing it in real time during matches is Phase 6+.
- **Not a squad optimizer.** §5.4.6 ranks and classifies *individual players
  within their position*. It does not solve for a 15-man squad under budget
  and formation constraints. That remains Phase 3, and the boundary is
  precise: Phase 5 answers "is this player trending toward or away from good
  for his position?"; Phase 3 answers "given £100.0m and this squad, what do I
  do?"
- **Not a second implementation of the model.** See §5.6 — still the hardest
  constraint in the spec and still the one most likely to be violated under
  schedule pressure.
- **Not a general-purpose BI tool.** §5.4.2 is a bounded chart builder over a
  fixed, curated column set, not an arbitrary query surface.

### §5.0.3 The one aesthetic-of-thought rule

No position is ever ranked against another position. Not in a table sort, not
in a heatmap color scale, not in a model bucket, not in a default chart axis.
§5.7 states the doctrine and it applies to every surface without exception.

### §5.0.4 Deviations from v1 — recorded per §5.16

| # | v1 said | v2 says | Reason |
|---|---|---|---|
| D1 | "Not a recommendation engine… this phase owns description" (v1 §5.0.2) | §5.4.6 Model Board ships in Phase 5, classifying players as optimal / rising / declining within position | The tool has a second user-facing job (§5.0.1 Job 2). Classification of *individual* players is separable from squad optimization and does not depend on Phase 3. The Phase 3 boundary is re-drawn, not erased — see §5.0.2. |
| D2 | Five fixed views (v1 §5.4) | Eight surfaces, one of which is a bounded drag-and-drop chart builder | The original brief was a JMP Graph Builder analogue. Fixed views serve Job 1; Job 2 requires user-driven exploration. Bounded by §5.6.2 so it cannot pressure client-side statistics. |
| D3 | Statistics rule permitted exactly one exception (Spearman) | Two exception classes: deliberate Spearman port, and exact descriptive reductions (§5.6.2) | A chart builder must aggregate. Exact reductions are deterministic and testable; inferential statistics remain forbidden. The line moved but it is still a line. |
| D4 | §5.14.5 left parquet transport open | Resolved: DuckDB-WASM, lazy-loaded, with its own budget line (§5.9) | The panel grain (§5.3.2) and the builder's group-by requirements make a real query engine the honest choice. Bundle budget is protected by route-level lazy loading, not by avoiding the dependency. |
| D5 | Section numbering §5.7 visual, §5.8 perf, §5.9 a11y, §5.10 test, §5.11 deploy, §5.12 milestones, §5.13 acceptance | Shifted by one from §5.7 onward to accommodate §5.7 Position doctrine | Mechanical. Cross-references in the README must be updated. |

v1's visual direction (v1 §5.7, now §5.8) is **carried forward unchanged** and
extended to the new surfaces. It is not re-opened.

---

## §5.1 Architecture

### §5.1.1 Static-first, no backend

Unchanged from v1. The application is a **static single-page app reading
pre-exported data files.** No API server, no database connection from the
browser, no runtime Python.

Rationale, in priority order:

1. **The pipeline stays the single source of truth.** A server tier would
   invite ad-hoc query logic that duplicates `analytics/` (§5.6).
2. **Zero operational surface.** No uptime, no secrets in a deployed
   environment, no hosting cost, no auth story for a single-user tool.
3. **It matches the existing delivery mechanism.** GitHub Actions already runs
   the collector on a schedule; publishing an export artifact and a built site
   from that workflow is an incremental change.

The addition of DuckDB-WASM (§5.2, D4) does **not** weaken this. It is a query
engine over static files shipped to the browser, not a service. No SQL written
by the app may compute an inferential statistic (§5.6).

A backend may be revisited only if a §5.9 budget is proven unmeetable with
static exports — and only via §5.16.

### §5.1.2 The five layers

```
  analytics/, backtest/          →  existing Python. Unchanged by this phase
           │
           ▼
  web/export/                    →  NEW Python. Serializes to a versioned contract
           │  data/web/v1/*.json, *.parquet
           ▼
  web/app/src/data/              →  TS load + validate. The only place fetch() lives
           │  typed, validated in-memory store
           ▼
  web/app/src/query/             →  NEW. DuckDB-WASM session + reduction helpers
           │  exact reductions only (§5.6.2)
           ▼
  web/app/src/views/             →  render + slice. No statistics (§5.6)
```

Arrows are one-directional. No layer reaches past its neighbour. The view layer
never touches a raw file path, never sees an unvalidated payload, and never
writes SQL directly — it calls named helpers in `query/`.

### §5.1.3 Stubbing unbuilt phases

Phase 3 and 4 surfaces (squad optimizer, paper trade) appear in navigation as
**disabled entries with a one-line explanation of what will live there**, not
as hidden routes and not as fake data. An empty state that explains the phase
boundary is honest; a mocked squad is a lie that will survive into screenshots.

§5.4.6 Model Board is **not** a Phase 3 stub. It ships real, and its empty
state — if the model has insufficient gameweeks to classify trend — says so in
those terms.

---

## §5.2 Locked stack

Additions beyond the parent spec's locked list (§1.1). Each must be justified
at its import site, per existing repo convention.

| Concern | Choice | Why this and not the obvious alternative |
|---|---|---|
| Build | Vite | Fastest static SPA path; no framework server needed. Next.js rejected — SSR/ISR/routing solve problems this app does not have. |
| Language | TypeScript, `strict: true` | The export contract (§5.3) is only worth having if type-checked at the boundary. |
| UI | React 18 | Widest chart-library compatibility; no state-management library. |
| State | React context + `useReducer` | App state is *one dataset + current selection + current encoding*. Redux/Zustand would be ceremony. |
| Charts | Visx (or D3 primitives directly) | Recharts is rejected for the heatmaps and rank-scatter — both need custom scales and per-cell interaction that Recharts fights. Recharts permitted for plain bar/line panels if it saves real time. |
| Chart builder rendering | Visx, driven by the §5.4.2 encoding spec | **Vega-Lite is rejected.** Its spec model is an excellent match for drop zones, but it ships its own theme and its own aggregation/regression transforms — the latter directly invite §5.6 violations. Owning the encoding→mark mapping in ~200 lines is cheaper than policing what a general grammar allows. |
| Tables | TanStack Table (headless) | Sorting/filtering/virtualization over ~700 rows without inheriting anyone's visual defaults. |
| Query | DuckDB-WASM, lazy-loaded | Panel grain (§5.3.2) is ~26k rows × ~60 columns. Arquero was considered and rejected: group-by across the panel with multiple filter predicates is SQL-shaped, and DuckDB's parquet reader removes a separate decode step. Loaded only on routes that need it (§5.9). |
| Styling | CSS modules + custom properties | No Tailwind. §5.8 is a small fixed token set; utility classes would obscure it. |
| Tests | Vitest + Testing Library + Playwright | Playwright required for §5.12.3 only. |

**Deliberately excluded:** any charting library shipping an opinionated theme;
any component kit (MUI, Chakra, shadcn) — §5.8 defines a specific visual
identity and a kit would fight it at every step.

### §5.2.1 Repo layout additions

```
web/
  export/
    __main__.py           # CLI: `uv run python -m web.export`
    contract.py           # pydantic models — THE schema of record
    columns.py            # column metadata registry (§5.3.5)
    players.py            # per-player current-state + projection export
    panel.py              # player × gameweek tidy panel (§5.3.2)
    normalize.py          # within-position z-scores and percentiles (§5.7)
    correlations.py       # precomputed rank-correlation matrices
    board.py              # model classification: optimal/rising/declining (§5.4.6)
    scorecard.py          # backtest metrics export
    timeseries.py         # distilled-shard → per-player trend export
  app/
    src/
      data/               # fetch + zod validation + store
      query/              # DuckDB-WASM session, named reduction helpers
      encoding/           # drop-zone state → mark inference → Visx props
      views/
      components/
      design/tokens.css
    public/
    package.json
data/web/v1/              # export output. Committed? — see §5.3.4
config/frontend.yaml      # thresholds: sample-size floor, staleness, trend window
tests/test_export_*.py
```

---

## §5.3 The export contract

### §5.3.1 Contract-first

`web/export/contract.py` defines pydantic models. `web/app/src/data/schema.ts`
defines mirrored zod schemas. **Both are hand-written and a test asserts they
agree** (§5.12.2). Generating one from the other is permitted; silent
divergence is not.

Every exported file carries a header object:

```json
{
  "contract_version": 1,
  "generated_at": "2026-08-22T14:00:00Z",
  "source_gameweek": 3,
  "scoring_config": "scoring_2026_27.yaml",
  "model_git_sha": "a1b2c3d",
  "normalization_basis": "within_position_season_to_date",
  "rows": 691
}
```

`model_git_sha` is non-negotiable. Every number on screen must be traceable to
the exact code that produced it.

`normalization_basis` is new in v2 and equally non-negotiable — a z-score is
meaningless without knowing what population it was computed against, and the
UI must be able to state that basis in a tooltip (§5.7.4).

### §5.3.2 Files

| File | Grain | Approx size | Contents |
|---|---|---|---|
| `columns.json` | one row per exported column | ~30 KB | **The registry (§5.3.5).** Machine-readable metadata driving the builder, mark inference, and every tooltip definition |
| `players.json` | one row per element | ~600 KB | id, name, team, position, price, ownership, minutes-head probabilities, per-component projections, total projection, trailing actuals, **within-position z-scores and percentiles for every rate metric** |
| `panel.parquet` | element × gameweek | 3–8 MB | **New in v2.** The tidy long table. One row per player per completed gameweek: raw stats, fantasy points, fixture context, and position-normalized companions |
| `board.json` | one row per classified element | ~150 KB | **New in v2.** Model bucket, composite score, per-metric trend slopes, the driving metrics behind each classification (§5.4.6) |
| `correlations.json` | metric × metric, per position + overall | ~40 KB | Spearman ρ, n, and p-value for every pair |
| `scorecard.json` | per model × season × gameweek | ~200 KB | MAE, RMSE, within-position Spearman, calibration bins, component decomposition, minutes-head metrics |
| `timeseries.parquet` | element × snapshot | 2–10 MB | price, ownership, projection over time |
| `fixtures.json` | fixture | ~60 KB | teams, kickoff, custom Elo difficulty *and* FPL's static rating |
| `golden_spearman.json` | metric pair | ~20 KB | Python-computed ρ fixtures for §5.6.1 CI |

### §5.3.3 Nulls are data

Where the pipeline distinguishes "the rule didn't exist yet" from zero — as it
does for `defensive_contribution` in pre-2025-26 seasons — the export preserves
`null` and the UI **renders a distinct not-applicable state, never a dash that
could read as zero.** Any view that coerces null to 0 for charting must label
that coercion visibly.

This extends to the panel and to normalization: a z-score computed over a
population containing nulls must carry its own `n`, and a player with
insufficient minutes to normalize against gets `null`, not `0.0`. A z-score of
zero means *exactly average*, which is a finding; null means *unknown*, which
is not.

### §5.3.4 Committed or generated?

**Committed:** `columns.json`, `players.json`, `board.json`,
`correlations.json`, `scorecard.json`, `fixtures.json`, `golden_spearman.json`.

**Not committed:** `panel.parquet`, `timeseries.parquet` (size, churn) —
produced by the deploy workflow and published as build artifacts.

Consequence, and it is deliberate: a fresh clone renders Correlation Lab,
Comparison, Explorer, Model Board, and Scorecard with no pipeline run. Graph
Builder, Form Matrix, and Trend Explorer require the parquet artifacts and show
an empty state explaining exactly that when absent — never a spinner, never a
silent blank chart.

### §5.3.5 The column registry — `columns.json`

**This is the most important new artifact in v2 and the highest-leverage file
in the frontend.** It is the single source of truth for what a column *is*, and
it drives the builder's column list, mark inference, axis formatting, tooltip
definitions, and position-relevance filtering. Getting it wrong makes every
downstream surface wrong in the same way.

One entry per column:

```json
{
  "key": "xgi_per90",
  "label": "xGI per 90",
  "role": "quantitative",
  "unit": "per90",
  "format": ".2f",
  "definition": "Expected goal involvements per 90 minutes played.",
  "source": "fpl_api",
  "grain": "player_gameweek",
  "normalizable": true,
  "normalized_key": "xgi_per90_z_pos",
  "position_relevance": { "GK": "none", "DEF": "secondary", "MID": "primary", "FWD": "primary" },
  "higher_is_better": true,
  "available_from_season": "2022-23"
}
```

Field notes that matter:

- **`role`** is one of `quantitative | categorical | ordinal | temporal`. This
  single field drives mark inference (§5.4.2). It is authored in Python, not
  guessed in TypeScript.
- **`position_relevance`** is the registry's expression of §5.7. Values:
  `primary | secondary | context | none`. The builder dims — never hides —
  columns marked `none` for the currently filtered position, with a tooltip
  explaining why. Hiding would teach the user that the column does not exist;
  dimming teaches them that it does not matter *here*, which is the actual
  lesson.
- **`higher_is_better`** drives diverging-scale orientation. `xGC` is a good
  stat to be low on and a heatmap that colors it like `xG` is actively
  misleading.
- **`available_from_season`** feeds the null-vs-zero rendering in §5.3.3.

---

## §5.4 Surfaces

Eight. Ordered by build priority; §5.13 milestones map to this order.

### §5.4.1 Correlation Lab *(hero — see §5.8.4)*

The landing view. Not a dashboard of KPI cards. **Unchanged from v1.**

- **Spearman matrix** across every exported metric, as a diverging heatmap.
  Cell shows ρ; tooltip shows ρ, n, p, and both metric definitions (pulled from
  `columns.json`).
- **Position filter** (GK/DEF/MID/FWD/all) swaps to the precomputed
  per-position matrix. Primary interaction; must be instant — hence
  precomputation, not client-side recompute.
- **Click a cell → rank scatter** below the matrix: the two metrics on *rank*
  axes, because that is what Spearman measures and plotting raw values beside a
  rank statistic misleads. Raw-axis toggle offered, defaulting to rank.
- **Sample-size honesty.** Cells below the configured floor are hatched, not
  colored. ρ of 0.9 on n=11 is not a finding.

### §5.4.2 Graph Builder *(new in v2 — D2)*

The user-driven exploration surface. A JMP Graph Builder analogue, bounded.

**Drop zones — exactly four.** X, Y, Color, Wrap. No Overlay, no Group X/Y, no
Size, no Shape, no Page. Four zones cover the overwhelming majority of real
questions and each additional zone multiplies the mark-inference matrix.

**Column list** built from `columns.json`, grouped by source and sorted with
`position_relevance: primary` first when a position filter is active.

**Mark inference is automatic and not user-selectable.** The user chooses data,
not chart type. This is the single decision that makes the builder usable
without a chart-type menu:

| X role | Y role | Color | Mark |
|---|---|---|---|
| quantitative | quantitative | any | point |
| quantitative | — | — | histogram |
| categorical | quantitative | — | bar (aggregated) |
| ordinal (gameweek) | quantitative | — | line |
| ordinal (gameweek) | quantitative | categorical | line, one per series |
| categorical | categorical | quantitative | **rect (heat map)** |
| ordinal | categorical | quantitative | **rect (heat map)** |
| temporal | quantitative | any | line |

**Aggregation control** appears only when the mark requires it — `sum`, `mean`,
`median`, `count`, `min`, `max`. These are exact reductions and permitted under
§5.6.2. No `regression`, no `loess`, no `confidence band`, no `trendline`.

**Global filter bar.** Position, team, price range, minutes floor, gameweek
range. Filters are app-level state and carry across surfaces (§5.5.3).

**Raw / position-normalized toggle**, defaulting per §5.7.3.

**Explicitly out of scope for this surface:** saved layouts, per-panel local
filters, brushing and linking across wrapped panels, row exclusion, point
labelling, smoothers, fit lines, right-click menus. These are JMP's long tail
and each is a week. Revisit in Phase 6 with evidence of need.

### §5.4.3 Form Matrix *(new in v2)*

A dedicated player × gameweek heat map. Technically expressible in §5.4.2, but
it is the single most useful view for the pedagogical job and deserves a
first-class route with sensible defaults rather than five drag operations.

- **Rows:** players, filtered and sorted (by composite score, by price, by
  ownership, by any exported column).
- **Columns:** gameweeks, in order, with deadline and blank/double markers.
- **Cell color:** a selectable metric, defaulting to position-normalized total
  points, using the §5.8.2 diverging scale oriented by `higher_is_better`.
- **Cell content:** the numeric value. Color is never the only encoding
  (§5.11).
- **Blank vs zero vs did-not-play** are three visually distinct states. A
  player who played 90 minutes and scored zero points and a player who was not
  in the squad are not the same fact and must not share a cell treatment.
- **Row hover** reveals a sparkline of the same metric; **row click** opens
  that player in Comparison.

This surface is where a slump or a hot streak becomes visible in one glance,
which no line chart of the same data achieves.

### §5.4.4 Player Comparison

**Carried from v1, extended per §5.7.**

- Select 2–4 players. Virtualized search across all elements.
- **Component decomposition bars** — projection broken into heads
  (appearance/minutes, goals, assists, clean sheets, defensive contribution,
  saves, bonus) side by side. Mirrors `analytics/evaluate.py`'s existing
  decomposition; the most diagnostically useful comparison in the app.
- **Minutes distribution** — P(blank) / P(short) / P(60+) as a stacked bar,
  never collapsed to a mean. The model deliberately refuses to produce a mean
  minutes figure; the UI must not reintroduce one.
- **Position-relative bars** *(new)*. When comparing players of the same
  position, offer percentile-within-position bars alongside raw values. When
  comparing across positions, percentile-within-position is the **default** and
  raw is opt-in, with a persistent note stating the comparison basis.
- Radar/spider charts **permitted only for normalized rate stats**, must state
  normalization basis on the chart, and — new in v2 — may only compare players
  of the same position. A radar overlaying a defender and a forward is a
  category error rendered attractively.

### §5.4.5 Player Explorer

Sortable, filterable, virtualized table over all elements. Column groups:
identity, price/ownership, projection + components, trailing actuals,
next-N fixture difficulty, **position-normalized companions**. Saved filter
presets in `localStorage`.

New in v2: a **normalization toggle at the column-group level** — the user
flips the whole projection block between raw and within-position z-score
without losing sort or filter state. Sorting a mixed-position table by raw xG
is permitted but shows a persistent inline caution (§5.7.5), because that sort
is exactly the mistake the tool exists to teach the user out of.

Deliberately plain. It is a workbench; the visual budget is spent elsewhere
(§5.8.5).

### §5.4.6 Model Board *(new in v2 — D1)*

The prescriptive surface, and the only one. Visually walled off per §5.8.6 so
it is never mistaken for the user's own analysis.

**Three buckets, computed per position, never across positions:**

- **Optimal** — highest current composite score within position.
- **Rising** — positive slope across the configured trend window on the
  underlying metrics, whether or not points have followed yet. This is the
  bucket with real edge: the model's claim is that the process has improved
  before the output has.
- **Declining** — negative slope on underlying metrics, whether or not points
  have fallen yet.

**Presentation:** ranked cards or a sortable table with per-player sparklines.
Not drag-and-drop — a different paradigm from §5.4.2, deliberately, because the
user is receiving rather than constructing here.

**Every card carries, without interaction:**
- the composite score and its within-position percentile,
- the two or three metrics that drove the classification, named,
- the trend window used,
- an amber flag if the classification rests on fewer than the configured
  minimum gameweeks or minutes.

**Every card carries an "Explain this" action** (§5.5.4). Non-negotiable.

**Composite scoring and position weights.** The composite is computed in Python
(§5.6) as a weighted sum of within-position z-scores. The weight profiles live
in `config/frontend.yaml`, are exported inside `board.json`, and are
**rendered on screen** in a "How this is scored" panel — the user must be able
to read the model's opinion, not just receive its output. Illustrative shape,
to be tuned against backtest rather than adopted as written:

```yaml
position_weights:
  GK:  { saves_p90: 0.25, clean_sheet_prob: 0.25, xgc_p90: -0.20, bonus_p90: 0.15, minutes_stability: 0.15 }
  DEF: { clean_sheet_prob: 0.30, xgc_p90: -0.20, xgi_p90: 0.20, bps_p90: 0.15, minutes_stability: 0.15 }
  MID: { xgi_p90: 0.30, xa_p90: 0.20, xg_p90: 0.20, creativity_p90: 0.15, minutes_stability: 0.15 }
  FWD: { xg_p90: 0.35, xgi_p90: 0.25, shots_in_box_p90: 0.15, minutes_stability: 0.25 }
```

Negative weights are meaningful and must survive the contract: conceding is bad
for a defender, and a scoring layer that cannot express that is not modelling
football.

**What this surface must never do:** suggest a transfer, name a captain,
propose a squad, or display a price-change prediction as a recommendation.
It classifies players. The user decides.

### §5.4.7 Model Scorecard

**Carried from v1 unchanged.** Renders `backtest/report.py`'s existing outputs.
Nothing here is new analysis; it is the existing report made legible.

- Event model vs. all three baselines: MAE, RMSE, within-position Spearman.
- **Calibration curves** with the diagonal drawn.
- **Error decomposition by event occurrence** — the existing breakdown.
- **The defender/goals-conceded finding gets a permanent annotated panel.**
  The `GOALS_CONCEDED_SHRINKAGE = 0.7` plateau is the most interesting result
  the project has produced; a shrinkage-vs-metrics plot showing the 0.6–0.85
  plateau belongs on screen, not only in a docstring.

New in v2: **Model Board accuracy gets its own panel.** If the app is going to
classify players as rising, it must report how often rising players
subsequently outperformed. A prescriptive surface without a published hit rate
is exactly the failure mode this repo exists to avoid.

### §5.4.8 Trend Explorer

Per-player time series from the distilled shards: price, ownership, projection.
Multi-select overlay. Deadline markers on the x-axis. Position-normalized
overlay option.

---

## §5.5 Interaction model

- **Sub-100 ms for anything precomputed** (filter, sort, position swap, cell
  selection). No spinner is acceptable for these.
- **URL is state.** Every view's selection encodes into the query string,
  including the full Graph Builder encoding spec, so a finding can be linked.
  Cheapest possible collaboration feature; costs nothing if done from the start.
- **Cross-view continuity.** Player selection, position filter, and the
  raw/normalized toggle are **app-level** state and carry across every surface.
  Encoding state is view-level.
- **Keyboard first.** Matrices are arrow-key navigable; tables are keyboard
  sortable; drop zones accept keyboard assignment as well as pointer drag.

### §5.5.4 "Explain this" — the pedagogical bridge

**The mechanism that makes §5.0.1 Job 2 real rather than aspirational.**

Every Model Board card exposes an action that navigates to Graph Builder with:

1. the player pre-filtered,
2. the driving metrics pre-assigned to the encoding channels,
3. the gameweek range set to the trend window that produced the classification,
4. the normalization toggle set to within-position,
5. a dismissible caption stating, in one sentence, what the model saw.

This must be built in the **same milestone as Model Board**, not deferred. It
is cheap — it sets existing state and navigates — and it is the entire
difference between a tool that issues verdicts and a tool that teaches. A
Model Board shipped without it is a regression against the spec's stated
purpose, not an incremental step toward it.

The reverse path exists too: any player selected in Graph Builder or Form
Matrix shows their current bucket as a small inline badge, linking to their
Model Board card.

---

## §5.6 Statistics in the browser — the hard rule

> **All inferential statistics are computed in Python and exported. The browser
> renders, slices, and reduces; it does not infer.**

The constraint most likely to erode. It matters because the repo's credibility
rests on leakage-safety and reproducibility, and a second Spearman
implementation in JavaScript — with its own tie-handling — creates a number
that no test covers and that can silently disagree with the paper result.

v2 widens the rule's *wording* (from "does not compute" to "does not infer")
because a chart builder must aggregate. The set of forbidden operations is
unchanged and is enumerated below.

### §5.6.1 Exception class 1 — the Spearman port

Arbitrary user-defined filters (e.g. "defenders over £6.0m with 400+ minutes")
cannot be precomputed. Where such a filter must produce a fresh correlation, a
client-side implementation is permitted **only** under all three conditions:

1. It is a **deliberate port** of `report.py`'s method — Pearson over
   ties-averaged ranks — in a single module, `src/data/spearman.ts`, with the
   Python source referenced in a comment.
2. `web/export/` emits `golden_spearman.json`: at least 50 metric pairs across
   positions with Python-computed ρ.
3. CI fails if the TS implementation disagrees with any golden value by more
   than 1e-9. (§5.12.2)

### §5.6.2 Exception class 2 — exact descriptive reductions *(new in v2 — D3)*

The Graph Builder and Form Matrix require aggregation. Permitted, in
`src/query/` only, and only this closed set:

`count`, `sum`, `mean`, `median`, `min`, `max`, `quantile`

These are exact, deterministic, order-independent reductions with no estimator
choice, no tie-handling convention, and no distributional assumption. They
cannot silently disagree with Python because there is nothing to disagree
about.

**Still forbidden client-side, without exception:**

regression of any kind · smoothing or LOESS · trend slopes · z-scores or any
standardization · percentile *ranks* against a population · significance
testing · confidence intervals · projection · imputation · shrinkage ·
correlation other than §5.6.1

Note carefully: `quantile` is permitted as a reduction over a user-filtered
set; computing a player's *percentile rank within position* is not — that is a
normalization and it ships from `normalize.py` (§5.7). The distinction is the
population: reductions summarize what the user filtered; normalizations
position a player against a reference group the model defines.

**Golden-value tests apply here too** (§5.12.2). Each permitted reduction is
tested against a Python fixture over the panel.

### §5.6.3 Provenance on screen

Every derived number renders with access to its provenance — at minimum n,
normalization basis, and contract version, via tooltip or an always-visible
footer showing `model_git_sha` and `generated_at`. Stale data is
self-announcing: if `generated_at` is older than the configured staleness
threshold, the header says so.

---

## §5.7 Position-relative normalization — the doctrine *(new in v2)*

### §5.7.1 The problem this solves

A defender will not post a forward's xG, and that says nothing about whether he
is a good defender. Any comparison, sort, color scale, or ranking that places
raw attacking metrics from different positions on one axis produces a
conclusion that is not merely useless but actively wrong, and — given §5.0.1
Job 2 — teaches the user something false about football.

### §5.7.2 The rule

**All cross-player comparison is against the player's own position group.**
Normalization is computed in `web/export/normalize.py`, exported as companion
columns, and never computed in the browser (§5.6.2).

For each metric flagged `normalizable` in `columns.json`, the export emits:

- `{key}_z_pos` — z-score against the position group,
- `{key}_pct_pos` — percentile within the position group,
- `{key}_n_pos` — the population size the above were computed against.

The `n` column is not optional. A z-score over eleven qualifying goalkeepers
carries different weight from one over two hundred midfielders, and the UI must
be able to flag the former (§5.8.2, amber).

**Eligibility floor.** Players below a configured minutes threshold are
excluded from the reference population and receive `null` normalized values —
not zero (§5.3.3). A 30-minute cameo should not move the positional mean.

### §5.7.3 Defaults per surface

| Surface | Default | Rationale |
|---|---|---|
| Correlation Lab | position-filtered matrices | already correct in v1 |
| Graph Builder | raw when a single position is filtered; **normalized** when position filter is "all" | the user asked a cross-position question; answer it honestly |
| Form Matrix | normalized | comparing players down the rows is the entire point |
| Comparison | raw for same-position, **normalized** for mixed-position | as §5.4.4 |
| Explorer | raw, with caution on mixed-position sorts of `normalizable` columns | it is a workbench and raw values are legitimately what you often want |
| Model Board | normalized, always | buckets are within-position by definition |

The toggle is always present and always reversible. Raw values are never
hidden — "which defenders get forward most" is a real question best answered in
raw xGI, and removing that capability to enforce a doctrine would be
paternalism rather than design.

### §5.7.4 Stating the basis

Any normalized number renders its basis on hover, in the model's own
vocabulary: the metric, the position group, the population size, the minutes
floor, and the window. "1.34 σ above DEF mean (n=147, ≥450 min, GW1–8)" is the
target register. A z-score without its basis is an unfalsifiable number and
this repo does not ship those.

### §5.7.5 The mixed-position caution

When a user sorts, colors, or plots a `normalizable` metric across mixed
positions in raw units, an inline caution appears — not a modal, not a
blocking dialog, not a tooltip the user must discover. One line, dismissible
per session, in `--flag` amber, naming the specific distortion: *"Raw xG across
positions — forwards will dominate this sort regardless of quality. Switch to
within-position."*

This is the highest-value single piece of copy in the application. It is the
moment the tool teaches.

---

## §5.8 Visual direction

**Carried forward from v1 §5.7 unchanged.** Extended below to the new surfaces.
Not re-opened.

### §5.8.1 The subject

A measuring instrument for a football model. Its visual world is *plotting and
instrumentation* — diverging scales, rank axes, calibration diagonals,
confidence hatching — not turf, not kit colors, not the pitch aesthetic of a
team-picker. **Explicitly reject the green-pitch fan vernacular.** That
aesthetic belongs to a different product and using it here would misrepresent
what the tool is for.

### §5.8.2 Palette

The product's central visual is a diverging correlation scale. So **the
diverging scale is the brand** — the accent colors are not decoration applied to
charts, they are the chart's own encoding promoted to identity.

```
--ground     #212C33   app background — dimmed slate, an instrument housing
--panel      #2C3A43   raised surfaces
--rule       #3E4F5A   hairlines, grid, axes
--paper      #E6EDF0   primary text
--muted      #93A5AF   secondary text, labels
--rho-neg    #1F9E9E   teal   — negative pole of the diverging scale
--rho-pos    #C4456B   rose   — positive pole of the diverging scale
--flag       #E0A83C   amber  — staleness, low-n, provisional values
```

The neutral ground is deliberately *not* near-black and *not* cream. The
diverging pair is deliberately teal↔rose rather than the conventional
blue↔orange: it remains distinguishable under deuteranopia and protanopia, and
does not read as a generic dark-mode-with-one-accent scheme, since neither pole
is subordinate to the other.

Amber is reserved exclusively for epistemic warnings — low sample size, stale
export, provisional 2026/27 GK values, mixed-position cautions (§5.7.5),
insufficient trend window. It never appears decoratively. When the user sees
amber, something about the *trustworthiness* of a number is being flagged.

**Extension for v2:** the same diverging scale serves every heat map in the
app, with orientation set by `higher_is_better` from `columns.json`. A metric
where low is good renders with the poles swapped, and the legend states the
direction explicitly. One scale, consistently meaning "far from the middle,"
across correlation, form, and comparison.

### §5.8.3 Type

Three roles; the data role is a hard requirement, not a nicety.

- **Display:** a wide/expanded grotesque used with restraint — view titles and
  the hero only. Suggested: Archivo Expanded, weights 600–700.
- **Body:** a neutral, high-legibility sans at small sizes. Suggested:
  Public Sans.
- **Data:** a monospace with **true tabular figures and a slashed zero**.
  Suggested: IBM Plex Mono. Every numeral in a table, matrix cell, or axis uses
  this face. Proportional figures in a stats table are a defect, not a
  preference — columns of numbers must align on the decimal.

Set `font-variant-numeric: tabular-nums` globally on data contexts.

### §5.8.4 Signature

**The correlation matrix is the hero.** The landing view opens directly onto the
full-bleed Spearman heatmap, animating in column-by-column on load — one
orchestrated reveal, then stillness. No KPI card row, no big-number-with-
gradient, no marketing hero above it.

The spec's one deliberate risk: opening a tool on a dense matrix rather than a
summary is hostile to a first-time visitor and correct for the only user this
tool has. It states in one screen what the product is.

Model Board does **not** compete for hero status and does not appear on the
landing route.

### §5.8.5 Restraint

Spend the visual budget on the matrices and the decomposition charts. Graph
Builder, Explorer, and Trend are utilitarian. No gradients outside the diverging
scale, no shadows beyond a single elevation step, no motion beyond the load
reveal and sub-150 ms hover states. `prefers-reduced-motion` disables the reveal
entirely.

### §5.8.6 Walling off the model *(new in v2)*

Model Board must be visually unmistakable as *the model speaking* rather than
the user's own analysis. The device: model-authored surfaces sit on `--panel`
with a persistent left rule in `--rule` and a small monospace attribution
carrying `model_git_sha`; user-driven surfaces sit directly on `--ground` with
no such rule.

No new colors, no badge, no icon set. The distinction is structural — a raised
panel with an attribution stripe — which is why it survives greyscale, low
vision, and screenshots, and why it costs nothing from the §5.8.5 budget.

### §5.8.7 Copy

Label things by what the user controls, in the model's own vocabulary —
"minutes head", "trailing rate", "within-position Spearman" — because the user
is the person who wrote those terms. Do not translate into fan-facing language.
Empty states explain the phase boundary (§5.1.3); errors state what failed and
what to do, and never apologize.

**Exception, and it is the point of §5.0.1 Job 2:** the §5.7.5 caution and the
§5.5.4 explanation caption are written for the user who does *not* yet have the
vocabulary. They name the distortion and the evidence in plain terms. Two
registers, deliberately, and the boundary between them is exactly the boundary
between describing the model and teaching football.

### §5.8.8 Contemporary execution *(new in v2)*

§5.8.1–§5.8.7 fix the *direction*. This section fixes the *craft*. What makes
an interface read as current is not its palette — it is density, rhythm,
optical precision, and restraint in motion. All of the following are
requirements, not suggestions.

**Color in OKLCH, not hex.** The §5.8.2 tokens are authored as OKLCH and the
hex values are documentation of intent, not the implementation. This is not
fashion: the diverging scale is the product's central visual, and sRGB
interpolation between teal and rose passes through a desaturated grey-mud
midpoint that makes near-zero correlations look like rendering artifacts.
OKLCH interpolation holds perceived lightness constant across the ramp, so
equal ρ steps read as equal visual steps. Use `oklch()` with `color-mix()` for
the ramp; ship an sRGB fallback via `@supports`.

```css
--rho-neg:  oklch(0.66 0.10 195);
--rho-pos:  oklch(0.56 0.16 010);
--ground:   oklch(0.27 0.02 230);
```

**Fluid type scale, one ratio, `clamp()` throughout.** A 1.2 minor third for
data-dense contexts and 1.333 for display. No fixed pixel type outside the data
face, where optical size stability matters more than fluidity. Variable fonts
only — Archivo and Public Sans both ship variable axes, and loading four static
weights in 2026 is a bundle-budget error as much as a craft one.

**Spacing on a 4px base with a named scale.** Six steps, no arbitrary values.
The single most common tell of an unconsidered interface is spacing that varies
by a few pixels between visually parallel elements.

**Radius discipline: instruments are crisp.** `0` on data surfaces — matrix
cells, table rows, axis containers, chart plot areas. `4px` on interactive
controls only. No large radii anywhere. Rounded cards are the dominant current
default and they actively fight §5.8.1: a measuring instrument does not have
soft corners.

**Density is a feature.** Default to compact. Table row height ≤ 32px, matrix
cells ≤ 28px, control heights ≤ 32px. This is a tool for repeated expert use;
generous whitespace here is not elegance, it is scrolling. Offer a comfortable
density toggle in Explorer only.

**Hairlines at true 1 device pixel.** `--rule` borders render at `0.5px` on
`min-resolution: 2dppx`. Grid lines and axes in charts likewise. On a HiDPI
display a 1px CSS border reads heavy against dense data and coarsens the whole
matrix.

**Layout with container queries, not viewport media queries.** Panels size to
their container, so the same chart component works in a full-bleed hero, a
comparison column, and a Model Board card without variant props. Use `@container`
throughout; reserve `@media` for the §5.10 responsive floor and for
`prefers-reduced-motion` / `prefers-contrast`.

**CSS Grid with named areas and `subgrid`** for the comparison and board
layouts, so component decomposition bars align across cards on a shared track.
Misaligned parallel charts are the defect `subgrid` exists to remove.

**Motion: two durations, one easing, nothing else.** `120ms` for state change
(hover, focus, selection), `320ms` for the §5.8.4 load reveal. Easing
`cubic-bezier(0.2, 0, 0, 1)`. Use the View Transitions API for route changes —
a cross-fade of ~160ms, no slide, no scale. Everything disabled under
`prefers-reduced-motion`. No parallax, no scroll-triggered reveals, no ambient
motion, no skeleton shimmer.

**Loading states are determinate.** §5.9 requires the user to know whether they
are waiting on 200 KB or 8 MB. A progress bar with a byte count, not a shimmer.
Skeleton screens are forbidden — they imply content shape before it is known and
on a data tool that is a small lie.

**Focus is designed, not defaulted.** `:focus-visible` only, a 2px `--paper`
ring at 2px offset, never `outline: none`. §5.10 requires full keyboard
operation of matrices and drop zones, which means focus states here are primary
UI, not an accessibility afterthought.

**Modern CSS is expected, not avoided.** Native nesting, `:has()` for
parent-conditional styling (a row containing a flagged cell, a zone holding an
assigned column), `light-dark()` unnecessary — this app is dark-only by
§5.8.1 and does not ship a light theme. Logical properties throughout.
Custom scrollbars styled to `--rule`, thin, never hidden.

**Explicitly forbidden — the current AI-design defaults.** Glassmorphism,
backdrop blur, gradient mesh or blob backgrounds, bento-box grids, oversized
rounded cards, glow or neon accents, purple-to-indigo gradients, big-number
KPI heroes, emoji in UI copy, icon-and-label buttons where a label suffices,
drop shadows beyond the single elevation step in §5.8.5. Any of these appearing
in a build is a §5.16 deviation requiring written justification, because each
is a choice made *for* the interface by prevailing fashion rather than *by* the
designer for this subject.

---

## §5.9 Performance budgets

| Metric | Budget |
|---|---|
| Initial JS bundle (gzipped, excludes lazy chunks) | ≤ 250 KB |
| DuckDB-WASM chunk (lazy, route-entry only) | ≤ 1.2 MB, never on initial load |
| Time to interactive, cold, 4G | ≤ 2.5 s |
| Precomputed interactions | ≤ 100 ms, no spinner |
| Table sort, ~700 rows | ≤ 50 ms |
| Matrix re-render on filter | ≤ 100 ms |
| Graph Builder encoding change, panel loaded | ≤ 200 ms |
| Form Matrix re-render on metric swap | ≤ 200 ms |
| Panel-dependent route, cold entry | ≤ 2.0 s including engine + parquet |
| Trend Explorer initial load | ≤ 1.5 s |

`panel.parquet` and `timeseries.parquet` load only on the routes that need
them, never on initial page load. Routes that need the engine show a
determinate progress state, not a spinner — the user should know whether they
are waiting on 200 KB or 8 MB.

---

## §5.10 Accessibility floor

Not aspirational — acceptance criteria.

- WCAG AA contrast for all text, verified against §5.8.2 tokens.
- **No heat map encodes meaning by color alone.** Cells carry numeric values;
  low-n cells are hatched (pattern, not hue). Applies to the correlation
  matrix and the Form Matrix equally.
- Graph Builder drop zones are keyboard-operable: a column can be assigned to a
  channel without a pointer, and current assignments are announced.
- Visible keyboard focus everywhere; full keyboard navigation of matrices and
  tables.
- `prefers-reduced-motion` respected.
- Responsive to 768px. Below that, matrices degrade to a scrollable ranked list
  of strongest pairs / most extreme cells rather than an unreadable grid.

---

## §5.11 Testing

### §5.11.1 Export layer
`pytest` over `web/export/`: schema conformance, null preservation (§5.3.3),
header completeness, registry completeness (every panel column has a
`columns.json` entry and vice versa — this test catches the most likely
silent breakage in the whole system), and a round-trip test that exported
projections match `analytics/projections.py` outputs exactly.

### §5.11.2 Contract, statistics, and normalization
- Test asserting `contract.py` and `schema.ts` describe the same shape.
- **Golden-value test for `spearman.ts` against Python**, tolerance 1e-9.
  Failure blocks merge.
- **Golden-value tests for every §5.6.2 reduction** against Python over the
  panel, exact equality for integer reductions and 1e-12 for floats.
- Normalization tests: z-scores recover to mean 0 / sd 1 within each position
  group; below-threshold players are `null`; `n` matches the eligible
  population count.
- **A test asserting no z-score, percentile, slope, or regression appears in
  `src/` outside `data/spearman.ts`.** A grep-level guard is crude and it is
  the guard most likely to catch the §5.6 erosion this spec predicts.

### §5.11.3 UI
- Vitest + Testing Library for data-layer, encoding-inference, and component
  logic. The mark-inference table (§5.4.2) is exhaustively unit-tested — every
  role combination, including the ones that produce no valid mark.
- Playwright for five flows only: load → matrix renders; select cell → scatter
  renders correct pair; select players → decomposition renders; drag column to
  X and Y → correct mark renders; **Model Board card → "Explain this" → Graph
  Builder opens with correct pre-loaded state**.
- Visual regression on the correlation matrix, Form Matrix, and decomposition
  charts.

---

## §5.12 Deployment

- New workflow `.github/workflows/web.yml`: on push to `main` and on successful
  completion of the collector workflow, run the export, build, and publish to
  GitHub Pages.
- The build **fails loudly on contract version mismatch** between committed
  data and the app's expected version. No silent degradation.
- Committed JSON (§5.3.4) means a fresh clone builds and serves without a
  pipeline run; parquet-dependent routes show their explanatory empty state.
- Parquet artifacts published from the same workflow, versioned alongside the
  contract.

---

## §5.13 Milestones

Each independently shippable and independently useful.

**5A — Contract, registry, and export.** `web/export/` producing all nine
files, including `columns.json` and `normalize.py`, tested, committed. No UI.
*Value: the data is inspectable, normalization is settled, and the interface is
frozen before any UI work begins.*

**5B — Correlation Lab.** Design system, app shell, matrix, rank scatter,
position filter, sample-size hatching. *Value: the hero surface exists.*

**5C — Query layer and Graph Builder.** DuckDB-WASM session, `query/` helpers
with golden tests, encoding state, mark inference, four drop zones, global
filter bar, raw/normalized toggle, URL state. *Value: the user can ask their
own questions.*

**5D — Form Matrix and Comparison.** Player × gameweek heat map, component
decomposition, minutes distribution, position-relative bars, cross-view
selection.

**5E — Model Board and the bridge.** Buckets, composite scoring panel, trend
sparklines, and **"Explain this" in the same milestone** (§5.5.4). Shipping
5E without the bridge is not permitted.

**5F — Explorer, Scorecard, Trend, deploy.** Virtualized table, backtest
rendering, calibration curves, shrinkage-plateau panel, board accuracy panel,
parquet-backed time series, Pages deploy.

Phase 3/4 surfaces stay stubbed throughout (§5.1.3).

---

## §5.14 Acceptance criteria

Measurable. All must hold.

1. Every number on screen is traceable to a `model_git_sha` visible in the UI.
2. `spearman.ts` matches Python golden values within 1e-9, enforced in CI.
3. No statistic outside the §5.6.1 and §5.6.2 exception sets is computed
   client-side. Enforced by the §5.11.2 guard test and by review of `src/`.
4. Every column in `panel.parquet` and `players.json` has a `columns.json`
   entry; every entry maps to a real column. Tested.
5. No surface ranks, sorts, colors, or buckets players of different positions
   against each other in raw units without the §5.7.5 caution visible.
6. All §5.9 budgets met, measured on a cold load in CI. DuckDB-WASM appears in
   no initial-load chunk.
7. WCAG AA contrast passes; every heat map's meaning survives greyscale.
8. A fresh clone renders every view except Graph Builder, Form Matrix, and
   Trend Explorer with no pipeline run; those three show their explanatory
   empty state rather than an error or a blank.
9. Null-vs-zero distinction (§5.3.3) preserved in every view, including
   normalized columns and Form Matrix cells, tested.
10. Correlation and Form Matrix cells below the sample-size floor are visually
    distinct by pattern, not hue.
11. Every Model Board card exposes "Explain this," and the resulting Graph
    Builder state matches the metrics named on the card. Tested in Playwright.
12. Model Board renders its position weight profiles on screen.
13. The five Playwright flows pass.
14. No mocked or placeholder data ships in any state, including empty states.
15. The diverging scale is authored in OKLCH with an sRGB fallback, and equal
    steps in ρ produce equal perceived lightness steps across the ramp.
16. No forbidden §5.8.8 pattern appears in the build. Spacing uses only the
    named scale; data surfaces have zero radius; motion uses only the two
    specified durations.
17. Every interactive element has a designed `:focus-visible` state; no
    `outline: none` appears in `src/`.

---

## §5.15 Open questions requiring a decision before 5A

Carried from v1 where unresolved, plus v2 additions.

1. **Sample-size floor for correlation cells.** What n makes a cell
   untrustworthy enough to hatch? Proposed: n < 30, configurable in
   `config/frontend.yaml`. Needs a real answer, not a default.
2. **Metric set for the matrix.** Which ~15–20 metrics? Too many makes the hero
   unreadable; too few makes it trivial. Requires a pass over what
   `features.py` actually exposes.
3. **Staleness threshold.** How old is `generated_at` before the header warns?
   Depends on collector cadence during a live gameweek.
4. **2026/27 provisional values.** GK save-bonus figures are unvalidated and
   flagged in the README. Every projection depending on them should presumably
   render with the amber flag — confirm scope, since that may be most
   goalkeeper numbers in the app.
5. **Minutes floor for normalization eligibility** (§5.7.2). Proposed: 450
   minutes season-to-date, but early season makes this bite hardest exactly
   when the tool is most used. Consider a rolling floor that scales with
   gameweeks elapsed. **Blocks 5A** — it changes the export.
6. **Trend window and slope method** (§5.4.6). How many gameweeks, and OLS
   slope versus rolling delta? A short window is noisy; a long one is late. This
   is the single largest determinant of whether "rising" means anything.
   **Blocks 5E**, but the column must exist in the 5A contract.
7. **Position weight profiles** (§5.4.6). The illustrative YAML is a starting
   point, not a proposal. These need backtest support — a weight profile that
   has never been validated against subsequent points is an opinion wearing a
   number's clothing.
8. **Minimum gameweeks before Model Board classifies at all.** Early season, no
   trend is credible. Options: suppress the board entirely below N gameweeks,
   or render it fully amber-flagged. Prefer suppression with an explanatory
   empty state — a fully-flagged board trains the user to ignore amber.
9. **Does `board.json` carry historical buckets?** Needed for the §5.4.7
   accuracy panel. If yes, the grain becomes element × gameweek and the file
   grows substantially. **Blocks 5A.**
10. **Panel scope.** Current season only, or multi-season? The FPL API does not
    serve historical seasons — bootstrap-static resets at rollover — so
    multi-season requires the archive already used by the collector. Affects
    `panel.parquet` size and every §5.9 budget on panel-dependent routes.

---

## §5.16 Deviations policy

Any departure from this spec is recorded in the README under a Phase 5
deviations heading, in the existing style: what was changed, and the real
reason, at the level of detail already used for the Elo and BPS decisions.

A deviation from §5.6 additionally requires stating what test now covers the
divergent number.

A deviation from §5.7 additionally requires stating what the user will now
conclude that is false, and why that is acceptable.

The §5.0.4 table is the first four entries under that heading and should be
transcribed into the README at 5A.
