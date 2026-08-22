# fpl-trends — Phase 5: Frontend Superprompt

Companion spec to `fpl-trends-superprompt.md`. Section numbers here extend
that document (§5.x). Where this spec and the parent conflict, the parent
wins and the conflict must be recorded in §5.14.

**Prerequisite state:** Phases 0–2 complete (collector, backtest harness,
event model + scoring layer). Phases 3–4 (squad optimizer, paper trade) are
*not* prerequisites — §5.1.3 defines how their surfaces are stubbed so this
phase can ship before them.

---

## §5.0 Purpose and non-goals

### §5.0.1 The one job

This frontend exists to answer one question: **is the model right, and
where is it wrong?** It is a diagnostic instrument for the person who built
the model, not a tip site for the person picking a captain.

Every surface must serve that job. A view that shows a number without
showing how much to trust that number does not belong in this phase.

### §5.0.2 Explicit non-goals

- **Not a public product.** No accounts, no auth, no multi-user state, no
  analytics/telemetry, no SEO, no marketing page.
- **Not a live-scores site.** In-play polling already exists in the
  collector; surfacing it in real time during matches is Phase 6+.
- **Not a recommendation engine.** The UI reports projections and their
  uncertainty. It does not tell the user who to transfer in. Phase 3
  (optimizer) owns prescription; this phase owns description.
- **Not a second implementation of the model.** See §5.6 — this is the
  hardest constraint in the spec and the one most likely to be violated
  under schedule pressure.

---

## §5.1 Architecture

### §5.1.1 Static-first, no backend

The application is a **static single-page app reading pre-exported data
files.** There is no API server, no database connection from the browser,
no runtime Python.

Rationale, in priority order:

1. **The pipeline stays the single source of truth.** A server tier would
   invite ad-hoc query logic that duplicates `analytics/` (§5.6).
2. **Zero operational surface.** No uptime, no secrets in a deployed
   environment, no hosting cost, no auth story for a single-user tool.
3. **It matches the existing delivery mechanism.** GitHub Actions already
   runs the collector on a schedule; publishing an export artifact and a
   built site from that same workflow is an incremental change, not a new
   system.

A backend may be revisited only if a §5.8 performance budget is proven
unmeetable with static exports — and only via §5.14.

### §5.1.2 The four layers

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
  web/app/src/views/             →  render + slice. No statistics (§5.6)
```

The arrows are one-directional. No layer reaches past its neighbour. In
particular the view layer never touches a raw file path and never sees an
unvalidated payload.

### §5.1.3 Stubbing unbuilt phases

Phase 3 and 4 surfaces (optimizer, paper trade) appear in the navigation as
**disabled entries with a one-line explanation of what will live there**,
not as hidden routes and not as fake data. An empty state that explains the
phase boundary is honest; a mocked squad is a lie that will survive into
screenshots.

---

## §5.2 Locked stack

Additions beyond the parent spec's locked list (§1.1). Each must be
justified at its import site, per existing repo convention.

| Concern | Choice | Why this and not the obvious alternative |
|---|---|---|
| Build | Vite | Fastest static SPA path; no framework server needed. Next.js is rejected — SSR/ISR/routing are all solving problems this app does not have. |
| Language | TypeScript, `strict: true` | The export contract (§5.3) is only worth having if it's type-checked at the boundary. |
| UI | React 18 | Widest chart-library compatibility; no state-management library — see below. |
| State | React context + `useReducer` | The entire app state is *one dataset + current selection*. Redux/Zustand would be ceremony. |
| Charts | Visx (or D3 primitives directly) | Recharts is rejected for the heatmap and rank-scatter — both need custom scales and per-cell interaction that Recharts fights. Recharts is permitted for plain bar/line panels if it saves real time. |
| Tables | TanStack Table (headless) | Sorting/filtering/virtualization over ~700 rows without inheriting anyone's visual defaults. |
| Styling | CSS modules + custom properties | No Tailwind. The design system in §5.7 is a small fixed token set; utility classes would obscure it. |
| Tests | Vitest + Testing Library + Playwright | Playwright is required for §5.10.3 only. |

**Deliberately excluded:** any charting library that ships its own opinionated
theme; any component kit (MUI, Chakra, shadcn) — §5.7 defines a specific
visual identity and a kit would fight it at every step.

### §5.2.1 Repo layout additions

```
web/
  export/
    __main__.py           # CLI: `uv run python -m web.export`
    contract.py           # pydantic models — THE schema of record
    players.py            # per-player current-state + projection export
    correlations.py       # precomputed rank-correlation matrices (§5.5.3)
    scorecard.py          # backtest metrics export (§5.5.4)
    timeseries.py         # distilled-shard → per-player trend export
  app/
    src/
      data/               # fetch + zod validation + store
      views/
      components/
      design/tokens.css
    public/
    package.json
data/web/v1/              # export output. Committed? — see §5.3.4
tests/test_export_*.py
```

---

## §5.3 The export contract

### §5.3.1 Contract-first

`web/export/contract.py` defines pydantic models. `web/app/src/data/schema.ts`
defines the mirrored zod schemas. **Both are hand-written and a test asserts
they agree** (§5.10.2). Generating one from the other is permitted but not
required; silent divergence is not.

Every exported file carries a header object:

```json
{
  "contract_version": 1,
  "generated_at": "2026-08-22T14:00:00Z",
  "source_gameweek": 3,
  "scoring_config": "scoring_2026_27.yaml",
  "model_git_sha": "a1b2c3d",
  "rows": 691
}
```

`model_git_sha` is non-negotiable. Every number on screen must be traceable
to the exact code that produced it — the repo already holds this standard
for backtest results and the UI does not get an exemption.

### §5.3.2 Files

| File | Grain | Approx size | Contents |
|---|---|---|---|
| `players.json` | one row per element | ~400 KB | id, name, team, position, price, ownership, minutes-head probabilities, per-component projections, total projection, trailing actuals |
| `correlations.json` | metric × metric, per position + overall | ~40 KB | Spearman ρ, n, and p-value for every pair |
| `scorecard.json` | per model × season × gameweek | ~200 KB | MAE, RMSE, within-position Spearman, calibration bins, component decomposition, minutes-head metrics |
| `timeseries.parquet` | element × snapshot | 2–10 MB | price, ownership, projection over time |
| `fixtures.json` | fixture | ~60 KB | teams, kickoff, custom Elo difficulty *and* FPL's static rating |

### §5.3.3 Nulls are data

Where the pipeline distinguishes "the rule didn't exist yet" from zero — as
it already does for `defensive_contribution` in pre-2025-26 seasons — the
export preserves `null` and the UI **renders a distinct not-applicable
state, never a dash that could read as zero.** Any view that coerces null
to 0 for charting must label that coercion visibly.

### §5.3.4 Committed or generated?

**Decision: `players.json`, `correlations.json`, `scorecard.json`, and
`fixtures.json` are committed. `timeseries.parquet` is not** (size, churn) —
it is produced by the deploy workflow and published as a build artifact.

This follows the parent spec's §8 principle that committed data must
regenerate metrics from a single command. The committed JSON means the site
builds and renders from a fresh clone with no pipeline run, which is what
makes the frontend independently testable.

---

## §5.4 Surfaces

Five views. Ordered by build priority — §5.12 milestones map to this order.

### §5.4.1 Correlation Lab *(hero — see §5.7.4)*

The landing view. Not a dashboard of KPI cards.

- **Spearman matrix** across every exported metric, as a diverging heatmap.
  Cell shows ρ; cell tooltip shows ρ, n, p, and the metric definitions.
- **Position filter** (GK/DEF/MID/FWD/all) swaps to the precomputed
  per-position matrix. This is the primary interaction and it must be
  instant — hence precomputation, not client-side recompute.
- **Click a cell → rank scatter** below the matrix: the two metrics plotted
  on *rank* axes (not raw), because that is what Spearman actually measures
  and plotting raw values next to a rank statistic misleads. Offer a raw-axis
  toggle, defaulting to rank.
- **Sample-size honesty.** Any cell with n below a configured floor is
  hatched, not colored. A ρ of 0.9 on n=11 is not a finding.

### §5.4.2 Player Comparison

- Select 2–4 players. Virtualized search across all elements.
- **Component decomposition bars** — the projection broken into its heads
  (appearance/minutes, goals, assists, clean sheets, defensive contribution,
  saves, bonus) side by side. This mirrors `analytics/evaluate.py`'s existing
  decomposition and is the most diagnostically useful comparison in the app.
- **Minutes distribution** — P(blank) / P(short) / P(60+) as a stacked bar,
  never collapsed to a mean. The model deliberately refuses to produce a mean
  minutes figure; the UI must not reintroduce one.
- Radar/spider charts are **permitted only for normalized rate stats**, and
  must state their normalization basis on the chart. A radar over
  differently-scaled raw metrics is decoration.

### §5.4.3 Player Explorer

Sortable, filterable, virtualized table over all elements. Column groups:
identity, price/ownership, projection + components, trailing actuals,
next-N fixture difficulty. Saved filter presets in `localStorage`. This view
is deliberately plain — it is a workbench, and the visual budget is spent
elsewhere (§5.7.5).

### §5.4.4 Model Scorecard

Renders `backtest/report.py`'s existing outputs. Nothing here is new
analysis; it is the existing report made legible.

- Event model vs. all three baselines: MAE, RMSE, within-position Spearman.
- **Calibration curves** with the diagonal drawn.
- **Error decomposition by event occurrence** — the existing breakdown.
- **The defender/goals-conceded finding gets a permanent annotated panel.**
  The `GOALS_CONCEDED_SHRINKAGE = 0.7` plateau is the single most
  interesting result the project has produced; a shrinkage-vs-metrics plot
  showing the 0.6–0.85 plateau belongs on screen, not only in a docstring.

### §5.4.5 Trend Explorer

Per-player time series from the distilled shards: price, ownership,
projection. Multi-select overlay. Deadline markers on the x-axis.

---

## §5.5 Interaction model

- **Sub-100 ms for anything precomputed** (filter, sort, position swap,
  cell selection). No spinner is acceptable for these.
- **URL is state.** Every view's selection encodes into the query string so
  a finding can be linked. This is the cheapest possible collaboration
  feature and it costs nothing at build time if done from the start.
- **Cross-view continuity.** Selecting players in Comparison carries to
  Explorer and Trend Explorer. Selection is app-level state, not view-level.
- **Keyboard first.** The matrix is arrow-key navigable; the table is
  keyboard sortable. This is a tool for repeated use by one expert user.

---

## §5.6 Statistics in the browser — the hard rule

> **All statistics are computed in Python and exported. The browser renders
> and slices; it does not compute.**

This is the constraint most likely to erode. It matters because the repo's
entire credibility rests on leakage-safety and reproducibility, and a second
Spearman implementation in JavaScript — with its own tie-handling — creates
a number that no test covers and that can silently disagree with the paper
result.

### §5.6.1 The one permitted exception

Arbitrary user-defined filters (e.g. "defenders over £6.0m with 400+
minutes") cannot be precomputed. Where such a filter must produce a fresh
correlation, a client-side implementation is permitted **only** under all
three conditions:

1. It is a **deliberate port** of `report.py`'s method — Pearson over
   ties-averaged ranks — in a single module, `src/data/spearman.ts`, with
   the Python source referenced in a comment.
2. `web/export/` emits a **golden-values fixture**: at least 50 metric pairs
   across positions with Python-computed ρ.
3. CI fails if the TS implementation disagrees with any golden value by more
   than 1e-9. (§5.10.2)

No other statistic gets this exception without an entry in §5.14. In
particular: no client-side regression, smoothing, projection, or
significance testing.

### §5.6.2 Provenance on screen

Every derived number renders with access to its provenance — at minimum n
and the contract version, via tooltip or an always-visible footer showing
`model_git_sha` and `generated_at`. Stale data must be self-announcing: if
`generated_at` is older than a configured staleness threshold, the header
says so.

---

## §5.7 Visual direction

### §5.7.1 The subject

This is a measuring instrument for a football model. Its visual world is
*plotting and instrumentation* — diverging scales, rank axes, calibration
diagonals, confidence hatching — not turf, not kit colors, not the pitch
aesthetic of a team-picker. **Explicitly reject the green-pitch fan
vernacular.** That aesthetic belongs to a different product and using it
here would misrepresent what the tool is for.

### §5.7.2 Palette

The product's central visual is a diverging correlation scale. So **the
diverging scale is the brand** — the accent colors are not decoration
applied to charts, they are the chart's own encoding promoted to identity.

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
blue↔orange: it remains distinguishable under deuteranopia and
protanopia, and it does not read as a generic dark-mode-with-one-accent
scheme, since neither pole is subordinate to the other.

Amber is reserved exclusively for epistemic warnings — low sample size,
stale export, provisional 2026/27 GK values. It never appears decoratively.
When the user sees amber, something about the *trustworthiness* of a number
is being flagged.

### §5.7.3 Type

Three roles, and the data role is a hard requirement, not a nicety:

- **Display:** a wide/expanded grotesque used with restraint — view titles
  and the hero only. Suggested: Archivo Expanded, weights 600–700.
- **Body:** a neutral, high-legibility sans at small sizes. Suggested:
  Public Sans.
- **Data:** a monospace with **true tabular figures and a slashed zero**.
  Suggested: IBM Plex Mono. Every numeral in a table, matrix cell, or axis
  uses this face. Proportional figures in a stats table are a defect, not a
  preference — columns of numbers must align on the decimal.

Set `font-variant-numeric: tabular-nums` globally on data contexts.

### §5.7.4 Signature

**The correlation matrix is the hero.** The landing view opens directly onto
the full-bleed Spearman heatmap, animating in column-by-column on load —
one orchestrated reveal, then stillness. No KPI card row, no big-number-with-
gradient, no marketing hero above it.

This is the spec's one deliberate risk: opening a tool on a dense matrix
rather than a summary is hostile to a first-time visitor and correct for the
only user this tool has. It states in one screen what the product is.

### §5.7.5 Restraint

Spend the visual budget on the matrix and the decomposition charts. Explorer
and Trend are utilitarian. No gradients outside the diverging scale, no
shadows beyond a single elevation step, no motion beyond the load reveal and
sub-150 ms hover states. `prefers-reduced-motion` disables the reveal
entirely.

### §5.7.6 Copy

Label things by what the user controls, in the model's own vocabulary —
"minutes head", "trailing rate", "within-position Spearman" — because the
user is the person who wrote those terms. Do not translate them into
fan-facing language. Empty states explain the phase boundary (§5.1.3);
errors state what failed and what to do, and never apologize.

---

## §5.8 Performance budgets

| Metric | Budget |
|---|---|
| Initial JS bundle (gzipped) | ≤ 250 KB |
| Time to interactive, cold, 4G | ≤ 2.5 s |
| Precomputed interactions | ≤ 100 ms, no spinner |
| Table sort, ~700 rows | ≤ 50 ms |
| Matrix re-render on filter | ≤ 100 ms |
| Trend Explorer initial load | ≤ 1.5 s (lazy — parquet loads on route entry only) |

`timeseries.parquet` is loaded only on Trend Explorer entry, never on
initial page load.

---

## §5.9 Accessibility floor

Not aspirational — these are acceptance criteria.

- WCAG AA contrast for all text, verified against the §5.7.2 tokens.
- **The heatmap does not encode meaning by color alone.** Cells carry the
  numeric value; low-n cells are hatched (pattern, not hue).
- Visible keyboard focus everywhere; full keyboard navigation of matrix and
  table.
- `prefers-reduced-motion` respected.
- Responsive to 768px. Below that, the matrix degrades to a scrollable
  ranked list of strongest pairs rather than an unreadable grid.

---

## §5.10 Testing

### §5.10.1 Export layer
`pytest` over `web/export/`: schema conformance, null preservation (§5.3.3),
header completeness, and a round-trip test that exported projections match
`analytics/projections.py` outputs exactly.

### §5.10.2 Contract and statistics
- Test asserting `contract.py` and `schema.ts` describe the same shape.
- **Golden-value test for `spearman.ts` against Python** (§5.6.1), tolerance
  1e-9. This test failing blocks merge.

### §5.10.3 UI
- Vitest + Testing Library for data-layer and component logic.
- Playwright for three flows only: load → matrix renders; select cell →
  scatter renders correct pair; select players → decomposition renders.
- Visual regression on the matrix and decomposition charts.

---

## §5.11 Deployment

- New workflow `.github/workflows/web.yml`: on push to `main` and on
  successful completion of the collector workflow, run the export, build,
  and publish to GitHub Pages.
- The build **fails loudly on contract version mismatch** between the
  committed data and the app's expected version. No silent degradation.
- Committed JSON (§5.3.4) means a fresh clone builds and serves without a
  pipeline run.

---

## §5.12 Milestones

Each is independently shippable and independently useful.

**5A — Contract and export.** `web/export/` producing all five files, tested,
committed. No UI. *Value: the data is inspectable and the interface is
frozen before any UI work begins.*

**5B — Correlation Lab.** Design system, app shell, matrix, rank scatter,
position filter, sample-size hatching. *Value: the hero surface exists.*

**5C — Comparison and Explorer.** Component decomposition, minutes
distribution, virtualized table, cross-view selection, URL state.

**5D — Scorecard and Trend.** Backtest rendering, calibration curves, the
shrinkage-plateau panel, parquet-backed time series, Pages deploy.

Phase 3/4 surfaces stay stubbed throughout (§5.1.3).

---

## §5.13 Acceptance criteria

Measurable, in the style of §4.4. All must hold.

1. Every number on screen is traceable to a `model_git_sha` visible in the
   UI.
2. `spearman.ts` matches Python golden values within 1e-9, enforced in CI.
3. No statistic other than Spearman is computed client-side. Verified by
   review of `src/` against §5.6.
4. All §5.8 budgets met, measured on a cold load in CI.
5. WCAG AA contrast passes; heatmap meaning survives greyscale.
6. A fresh clone renders every view except Trend Explorer with no pipeline
   run.
7. Null-vs-zero distinction (§5.3.3) preserved in every view, tested.
8. Correlation cells below the sample-size floor are visually distinct from
   colored cells by pattern, not hue.
9. The three Playwright flows pass.
10. No mocked or placeholder data ships in any state, including empty states.

---

## §5.14 Open questions requiring a decision before 5A

1. **Sample-size floor.** What n makes a correlation cell untrustworthy
   enough to hatch? Proposed: n < 30, configurable in
   `config/frontend.yaml`. Needs a real answer, not a default.
2. **Metric set for the matrix.** Which ~15–20 metrics? Too many makes the
   hero unreadable; too few makes it trivial. Requires a pass over what
   `features.py` actually exposes.
3. **Staleness threshold.** How old is `generated_at` before the header
   warns? Depends on collector cadence during a live gameweek.
4. **2026/27 provisional values.** The GK save-bonus figures are unvalidated
   and flagged as such in the README. Every projection depending on them
   should presumably render with the amber flag — confirm scope, since that
   may be most goalkeeper numbers in the app.
5. **Trend Explorer transport.** Parquet-in-browser needs DuckDB-WASM or
   hyparquet, both of which push the §5.8 bundle budget. Alternative: export
   a pruned JSON time series and defer parquet to Phase 6. Decide at 5A, not
   5D.

---

## §5.15 Deviations policy

Any departure from this spec is recorded in the README under a Phase 5
deviations heading, in the existing style: what was changed, and the real
reason, at the level of detail already used for the Elo and BPS decisions.
A deviation from §5.6 additionally requires stating what test now covers the
divergent number.
