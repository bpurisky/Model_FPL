/**
 * The export contract, in zod (§5.3.1, §5.12.2).
 *
 * The mirror of `web/export/contract.py`. Both halves are hand-written and
 * a test asserts they describe the same shape —
 * `tests/test_schema_ts.py::test_schema_ts_matches_the_python_contract`,
 * which parses this file and compares it against `contract_shape()`. That
 * test runs in the Python suite rather than in vitest because it has to
 * guard the boundary from today, and the TypeScript toolchain arrives at
 * milestone 5B. It fails on a field added to either side alone.
 *
 * Conventions this file keeps, because the agreement test parses it:
 *
 *   - one field per line, `name: <zod expression>,`
 *   - `.nullable()` mirrors a Python `X | None`
 *   - `.optional()` mirrors a Python field with a default
 *   - each model is `export const Name = z.object({ ... });`
 *
 * Keep them. A clever refactor here breaks the only thing standing
 * between the app and a silently divergent contract.
 *
 * Nulls are load-bearing throughout and are never coerced to zero
 * (§5.3.3). A null rho is "no correlation is defined over this pair",
 * a null z-score is "this player is below the minutes floor", and a null
 * `model_projection` is "the model had not spoken yet". None of them is
 * a small number, and any view that renders them as one is wrong.
 */

import { z } from "zod";

/** §5.3.1's header, on every exported file. */
export const Header = z.object({
  contract_version: z.number().int().optional(),
  generated_at: z.string().datetime(),
  source_gameweek: z.number().int().nullable(),
  scoring_config: z.string(),
  model_git_sha: z.string().nullable(),
  normalization_basis: z.string(),
  rows: z.number().int(),
  current_season: z.string().nullable().optional(),
});

export const Role = z.enum(["quantitative", "categorical", "ordinal", "temporal"]);
export const Relevance = z.enum(["primary", "secondary", "context", "none"]);
export const Grain = z.enum([
  "player",
  "player_gameweek",
  "fixture",
  "metric_pair",
  "model_gameweek",
]);
export const Source = z.enum(["fpl_api", "vaastav_archive", "derived", "model"]);
export const Position = z.enum(["GK", "DEF", "MID", "FWD"]);

/** One entry in `columns.json` (§5.3.5) — the registry the builder reads. */
export const ColumnSpec = z.object({
  key: z.string(),
  label: z.string(),
  role: Role,
  unit: z.string().nullable(),
  format: z.string(),
  definition: z.string(),
  source: Source,
  grain: Grain,
  normalizable: z.boolean(),
  normalized_key: z.string().nullable(),
  position_relevance: z.record(Position, Relevance),
  higher_is_better: z.boolean().nullable(),
  available_from_season: z.string().nullable(),
  available_to_season: z.string().nullable().optional(),
});

export const ColumnsFile = z.object({
  header: Header,
  columns: z.array(ColumnSpec),
});

/**
 * One metric pair within one position group. `n` counts rows where *both*
 * metrics are present, not the size of the group — four metrics exist only
 * from 2025-26, so their cells draw on a third of the pooled population.
 */
export const CorrelationCell = z.object({
  group: z.string(),
  a: z.string(),
  b: z.string(),
  rho: z.number().nullable(),
  n: z.number().int(),
  p_value: z.number().nullable(),
});

/** `mixed_position` is the flag §5.7.5's caution copy attaches to. */
export const GroupSummary = z.object({
  key: z.string(),
  n_player_seasons: z.number().int(),
  mixed_position: z.boolean(),
});

export const CorrelationsFile = z.object({
  header: Header,
  basis: z.string(),
  min_n_cell: z.number().int(),
  seasons: z.array(z.string()),
  metrics: z.array(z.string()),
  groups: z.array(GroupSummary),
  cells: z.array(CorrelationCell),
});

export const PositionSpearman = z.object({
  position: z.string(),
  rho: z.number().nullable(),
  n: z.number().int(),
  p_value: z.number().nullable(),
});

/**
 * `season: null` is the pooled-across-seasons row and `gw: null` is the
 * season rollup. Those nulls are structural: select a grain by filtering
 * on them, never by re-aggregating the detail rows (§5.6), which would
 * produce a third number matching neither this file nor the report.
 */
export const ScorecardRow = z.object({
  model: z.string(),
  season: z.string().nullable(),
  gw: z.number().int().nullable(),
  n: z.number().int(),
  mae: z.number().nullable(),
  rmse: z.number().nullable(),
  spearman_mean: z.number().nullable(),
  spearman_by_position: z.array(PositionSpearman),
});

export const CalibrationBin = z.object({
  model: z.string(),
  bin: z.number().int(),
  n: z.number().int(),
  mean_prediction: z.number().nullable(),
  mean_actual: z.number().nullable(),
});

export const EventErrorBucket = z.object({
  model: z.string(),
  bucket: z.string(),
  n: z.number().int(),
  mae: z.number().nullable(),
});

export const ReturnGroupError = z.object({
  model: z.string(),
  group: z.string(),
  n: z.number().int(),
  mae: z.number().nullable(),
  rmse: z.number().nullable(),
  mean_prediction: z.number().nullable(),
  mean_actual: z.number().nullable(),
});

export const ComponentError = z.object({
  component: z.string(),
  mae: z.number().nullable(),
});

export const MinutesHead = z.object({
  brier_blank: z.number().nullable(),
  brier_short: z.number().nullable(),
  brier_full: z.number().nullable(),
  mae_expected_minutes: z.number().nullable(),
  n: z.number().int(),
});

export const ScorecardFile = z.object({
  header: Header,
  models: z.array(z.string()),
  seasons: z.array(z.string()),
  event_model: z.string(),
  rows: z.array(ScorecardRow),
  calibration: z.array(CalibrationBin),
  error_by_event: z.array(EventErrorBucket),
  component_decomposition: z.array(ComponentError),
  minutes_head: MinutesHead,
  error_by_return: z.array(ReturnGroupError).optional(),
});

export const DifficultyBasis = z.enum(["pre_match", "current_elo"]);

/**
 * `difficulty_basis` keeps the two Elo figures from being read as one
 * number: a played fixture reports what each club carried *into* it, an
 * unplayed one reports what they hold today.
 */
export const FixtureRow = z.object({
  fixture: z.number().int(),
  gw: z.number().int().nullable(),
  team_h: z.string(),
  team_a: z.string(),
  kickoff_time: z.string().datetime().nullable(),
  played: z.boolean(),
  team_h_difficulty: z.number().int().nullable(),
  team_a_difficulty: z.number().int().nullable(),
  custom_difficulty_home: z.number().nullable(),
  custom_difficulty_away: z.number().nullable(),
  difficulty_basis: DifficultyBasis,
});

export const FixturesFile = z.object({
  header: Header,
  season: z.string(),
  elo_matches: z.number().int(),
  elo_seeded_from: z.array(z.string()),
  unseeded_teams: z.array(z.string()),
  fixtures: z.array(FixtureRow),
});

/**
 * `rows` is a positional matrix aligned to `metrics`. Nulls in it are
 * real — four metrics do not exist before 2025-26 — and `spearman.ts`
 * must drop incomplete pairs rather than rank a column containing them.
 */
export const GoldenSample = z.object({
  group: z.string(),
  metrics: z.array(z.string()),
  rows: z.array(z.array(z.number().nullable())),
});

export const GoldenPair = z.object({
  group: z.string(),
  a: z.string(),
  b: z.string(),
  n: z.number().int(),
  rho: z.number().nullable(),
});

export const GoldenSpearmanFile = z.object({
  header: Header,
  method: z.string(),
  tolerance: z.number(),
  precision: z.number().int(),
  samples: z.array(GoldenSample),
  pairs: z.array(GoldenPair),
});

/**
 * One reduction applied to one column of the embedded sample (§5.6.2).
 *
 * `q` is populated only for `quantile` and is null for the other six —
 * the parameter is what makes `quantile` the single name in §5.6.2's
 * list with competing conventions in the wild, so it travels with the
 * answer rather than being implied by the caller.
 *
 * `value` is nullable and `n` is not. That asymmetry is the rule under
 * test: an empty set has no mean, no sum and no maximum, but it always
 * has a count, and it is zero.
 */
export const ReductionCase = z.object({
  column: z.string(),
  fn: z.string(),
  q: z.number().nullable(),
  value: z.number().nullable(),
  n: z.number().int(),
});

export const GoldenReductionsFile = z.object({
  header: Header,
  method: z.string(),
  tolerance: z.number(),
  precision: z.number().int(),
  columns: z.array(z.string()),
  rows: z.array(z.array(z.number().nullable())),
  cases: z.array(ReductionCase),
});

/**
 * One shrinkage value, and what the model scored at it (§5.4.7).
 *
 * The two verdict flags are computed in the export, not here: both are
 * comparisons against a baseline the browser does not hold, and §5.6
 * wants a verdict the export reached rather than one a view inferred.
 */
export const ShrinkagePoint = z.object({
  shrinkage: z.number(),
  mae: z.number().nullable(),
  spearman_mean: z.number().nullable(),
  spearman_focus: z.number().nullable(),
  n: z.number().int(),
  beats_mae_bar: z.boolean(),
  beats_spearman_bar: z.boolean(),
  beats_focus_bar: z.boolean(),
});

export const ShrinkageFile = z.object({
  header: Header,
  default: z.number(),
  focus_position: z.string(),
  baseline_mae: z.number().nullable(),
  baseline_mae_model: z.string(),
  baseline_spearman_mean: z.number().nullable(),
  baseline_spearman_focus: z.number().nullable(),
  baseline_spearman_model: z.string(),
  points: z.array(ShrinkagePoint),
});

/**
 * §5.4.6's weight profile for one position. `weights` is a metric->number
 * map and the numbers are signed: conceding is bad for a defender, and a
 * scoring layer that cannot express that is not modelling football.
 */
export const PositionWeights = z.object({
  position: z.string(),
  weights: z.record(z.string(), z.number()),
});

/**
 * What a bucket was actually worth. `lift` is negative for `declining`,
 * which is the right sign for a warning. The surface must not present a
 * bucket without being able to show this alongside it.
 */
export const BoardBucketAccuracy = z.object({
  bucket: z.string(),
  n: z.number().int(),
  comparison: z.string(),
  forward_points: z.number().nullable(),
  forward_points_other: z.number().nullable(),
  lift: z.number().nullable(),
});

/** One card, or one row of the ranked list — the same record serves both. */
export const BoardPlayer = z.object({
  element_id: z.number().int(),
  name: z.string(),
  team: z.string(),
  position: z.string(),
  composite: z.number().nullable(),
  percentile: z.number().nullable(),
  rank: z.number().int(),
  bucket: z.string(),
  drivers: z.array(z.string()),
  gameweeks_seen: z.number().int(),
  low_confidence: z.boolean(),
});

export const BoardFile = z.object({
  header: Header,
  season: z.string(),
  gameweek: z.number().int(),
  trend_window: z.number().int(),
  min_gameweeks: z.number().int(),
  weights: z.array(PositionWeights),
  bucket_accuracy: z.array(BoardBucketAccuracy),
  players: z.array(BoardPlayer),
});

/**
 * One rate metric for one player. `z` and `percentile` are null below the
 * minutes floor — unknown, not average (§5.3.3). The population size is
 * not here: it belongs to the position group, and lives on the file.
 */
export const PlayerMetric = z.object({
  value: z.number().nullable(),
  z: z.number().nullable(),
  percentile: z.number().nullable(),
});

/** `total` is the sum of `components` by construction. */
export const PlayerProjection = z.object({
  components: z.record(z.string(), z.number().nullable()),
  total: z.number().nullable(),
  p_blank: z.number().nullable(),
  p_short: z.number().nullable(),
  p_full: z.number().nullable(),
});

export const PlayerRow = z.object({
  element_id: z.number().int(),
  name: z.string(),
  team: z.string(),
  position: z.string(),
  price: z.number().int().nullable(),
  selected: z.number().int().nullable(),
  gameweeks: z.number().int(),
  actuals: z.record(z.string(), z.number().nullable()),
  metrics: z.record(z.string(), PlayerMetric),
  projection: PlayerProjection.nullable(),
  // Added after 5D, contract_version unchanged: set-piece taker rank,
  // 1 = first choice. Absent in a file built before the collector kept them.
  penalties_order: z.number().int().nullable().optional(),
  corners_order: z.number().int().nullable().optional(),
  direct_freekicks_order: z.number().int().nullable().optional(),
});

export const ProjectionBasis = z.enum(["next_fixture", "fixture_neutral"]);

export const PlayersFile = z.object({
  header: Header,
  season: z.string(),
  gameweek: z.number().int(),
  projected_gameweek: z.number().int(),
  projection_basis: ProjectionBasis,
  population: z.record(z.string(), z.record(z.string(), z.number().int())),
  players: z.array(PlayerRow),
});

/** One fixture a club plays in a projected gameweek. */
export const ProjectionFixture = z.object({
  opponent: z.string(),
  home: z.boolean(),
  difficulty: z.number().int().nullable(),
});

/** None is a blank, two a double. Present for every gameweek of the horizon. */
export const ProjectionTeamGameweek = z.object({
  gw: z.number().int(),
  fixtures: z.array(ProjectionFixture),
});

export const ProjectionTeam = z.object({
  team: z.string(),
  short_name: z.string(),
  gameweeks: z.array(ProjectionTeamGameweek),
});

/**
 * `xp[i]` belongs to `ProjectionsFile.gameweeks[i]`. The minutes
 * probabilities are the first projected gameweek's; `p_full` is P(60+).
 */
export const ProjectionPlayer = z.object({
  element_id: z.number().int(),
  name: z.string(),
  team: z.string(),
  position: z.string(),
  price: z.number().int().nullable(),
  xp: z.array(z.number().nullable()),
  p_blank: z.number().nullable(),
  p_short: z.number().nullable(),
  p_full: z.number().nullable(),
});

export const ProjectionsFile = z.object({
  header: Header,
  season: z.string(),
  gameweek: z.number().int(),
  gameweeks: z.array(z.number().int()),
  teams: z.array(ProjectionTeam),
  players: z.array(ProjectionPlayer),
});

/**
 * What a season actually covers. `partial` keeps the season selector
 * honest: a rho over two gameweeks and one over thirty-eight look
 * identical once they are both a rho.
 */
export const SeasonSummary = z.object({
  season: z.string(),
  gameweeks: z.number().int(),
  players: z.number().int(),
  partial: z.boolean(),
});

/**
 * One eligible player-season. `values` is positional, aligned to
 * `ObservationsFile.metrics` — the same convention `GoldenSample.rows`
 * uses. Nulls are real and must be dropped pairwise, never ranked.
 */
export const ObservationRow = z.object({
  season: z.string(),
  element_id: z.number().int(),
  name: z.string(),
  team: z.string(),
  position: z.string(),
  values: z.array(z.number().nullable()),
});

export const ObservationsFile = z.object({
  header: Header,
  basis: z.string(),
  seasons: z.array(SeasonSummary),
  metrics: z.array(z.string()),
  rows: z.array(ObservationRow),
});

/** `papertrade.json` (§5.16 D14) — §6.3-6.5 made legible. */
export const PaperTradeGwNote = z.object({
  gw: z.number().int(),
  reason: z.string(),
});

export const PaperTradeSquadLevelGw = z.object({
  gw: z.number().int(),
  real_points: z.number().int(),
  shadow_points: z.number().int(),
});

export const PaperTradeSquadLevel = z.object({
  n_gameweeks: z.number().int(),
  per_gw: z.array(PaperTradeSquadLevelGw),
  excluded_gameweeks: z.array(PaperTradeGwNote),
  cumulative_real_points: z.number().int(),
  cumulative_shadow_points: z.number().int(),
  shadow_minus_real: z.number().int(),
  warning: z.string(),
});

export const PaperTradePlayerLevelGw = z.object({
  gw: z.number().int(),
  n: z.number().int(),
  mae: z.number().nullable(),
  spearman_mean: z.number().nullable(),
  fpl_n: z.number().int().nullable().optional(),
  fpl_mae: z.number().nullable().optional(),
  fpl_spearman_mean: z.number().nullable().optional(),
  baseline_trailing_mean_mae: z.number().nullable().optional(),
  baseline_trailing_mean_spearman_mean: z.number().nullable().optional(),
  baseline_fpl_form_mae: z.number().nullable().optional(),
  baseline_fpl_form_spearman_mean: z.number().nullable().optional(),
  baseline_fixture_adj_mae: z.number().nullable().optional(),
  baseline_fixture_adj_spearman_mean: z.number().nullable().optional(),
});

export const PaperTradeLeakageCheck = z.object({
  ran: z.boolean(),
  passed: z.boolean(),
  n_features: z.number().int(),
  latest_feature_available_at: z.string().datetime().nullable(),
  deadline: z.string().datetime().nullable(),
});

/** That gameweek's own frozen sha -- not `PaperTradeFile.header.model_git_sha`,
 * which is the live model today. */
export const PaperTradeFreezeProvenance = z.object({
  gw: z.number().int(),
  leakage_check: PaperTradeLeakageCheck.nullable(),
  leakage_verified: z.boolean(),
  manual_correction: z.string().nullable(),
  records_manual_correction_field: z.boolean(),
  model_git_sha: z.string().nullable(),
});

export const PaperTradeGateCriterion = z.object({
  status: z.string(),
  detail: z.string(),
});

export const PaperTradeLaunchGate = z.object({
  ready_to_launch: z.boolean(),
  gameweeks_evaluated: z.number().int(),
  gameweeks_excluded: z.array(PaperTradeGwNote),
  freeze_provenance: z.array(PaperTradeFreezeProvenance),
  criteria: z.record(z.string(), PaperTradeGateCriterion),
});

export const PaperTradePriceEval = z.object({
  n: z.number().int(),
  n_moves_predicted: z.number().int(),
  hit_rate: z.number().nullable(),
  ci_low: z.number().nullable(),
  ci_high: z.number().nullable(),
});

export const PaperTradeFile = z.object({
  header: Header,
  player_level: z.array(PaperTradePlayerLevelGw),
  player_level_excluded: z.array(PaperTradeGwNote),
  player_level_skipped: z.array(PaperTradeGwNote),
  squad_level: PaperTradeSquadLevel,
  launch_gate: PaperTradeLaunchGate,
  price_eval: PaperTradePriceEval,
});

/** `prices.json` — Price Watch: the rise-only price model's call for the
 * current daily cycle, and its own track record. */
export const PriceWatchStatus = z.enum(["ok", "no_data", "stale", "insufficient_history", "deadline_in_window"]);

export const PriceWatchPlayer = z.object({
  element_id: z.number().int(),
  name: z.string(),
  team: z.string(),
  position: z.string(),
  /** Tenths of a million. */
  now_cost: z.number().int(),
  net_transfers_24h: z.number().nullable(),
  /** `net_transfers_24h / threshold`: negative for an outflow, above 1 past the cut. */
  progress: z.number().nullable(),
  eligible: z.boolean(),
  ineligible_reason: z.enum(["moved_within_72h", "insufficient_history"]).nullable(),
  call: z.enum(["rise", "stable"]),
});

/** `config/price_model.yaml`'s fit statistics: IN-SAMPLE, never the headline. */
export const PriceInSampleFit = z.object({
  fit_cutoff: z.string().nullable(),
  fit_span: z.string().nullable(),
  fit_cycles: z.number().int().nullable(),
  rise_f1: z.number().nullable(),
  rise_n_called: z.number().int().nullable(),
  hit_rate: z.number().nullable(),
  fitted_at: z.string().nullable(),
});

export const PriceCycleOutcome = z.object({
  date: z.string(),
  calls: z.number().int(),
  hits: z.number().int(),
  out_of_sample: z.boolean(),
});

export const PriceTrackRecord = z.object({
  status: z.string(),
  n_calls: z.number().int(),
  n_cycles: z.number().int(),
  min_calls: z.number().int(),
  hit_rate: z.number().nullable(),
  ci_low: z.number().nullable(),
  ci_high: z.number().nullable(),
  span: z.string().nullable(),
  in_sample: PriceInSampleFit.nullable(),
  recent_cycles: z.array(PriceCycleOutcome),
});

export const PricesFile = z.object({
  header: Header,
  status: PriceWatchStatus,
  status_detail: z.string().nullable(),
  snapshot_ts: z.string().datetime().nullable(),
  boundary_ts: z.string().datetime().nullable(),
  next_boundary_ts: z.string().datetime().nullable(),
  locked: z.boolean().nullable(),
  threshold: z.number(),
  recent_move_hours: z.number().int(),
  players: z.array(PriceWatchPlayer),
  track_record: PriceTrackRecord,
});

/** `ownership.json` — Elite Ownership: the top-N managers' ownership and
 * effective ownership for the latest gameweek with collected picks. */
export const OwnershipPlayer = z.object({
  element_id: z.number().int(),
  name: z.string(),
  team: z.string(),
  position: z.string(),
  /** Tenths of a million. */
  now_cost: z.number().int().nullable(),
  /** Field-wide selected-by %, 0-100. */
  overall_pct: z.number().nullable(),
  /** Share of the top N holding him in their 15, 0-100. */
  top_pct: z.number(),
  top_start_pct: z.number(),
  captain_pct: z.number(),
  /** sum(multiplier) / n_managers * 100; can exceed 100. */
  eo_pct: z.number(),
  /** top_pct - overall_pct. */
  delta_pct: z.number().nullable(),
});

export const OwnershipChip = z.object({
  chip: z.string(),
  count: z.number().int(),
});

export const OwnershipFile = z.object({
  header: Header,
  status: z.enum(["ok", "no_data"]),
  gw: z.number().int().nullable(),
  n_managers: z.number().int(),
  fetched_at: z.string().datetime().nullable(),
  chips: z.array(OwnershipChip),
  players: z.array(OwnershipPlayer),
});

export type Header = z.infer<typeof Header>;
export type ColumnSpec = z.infer<typeof ColumnSpec>;
export type ColumnsFile = z.infer<typeof ColumnsFile>;
export type CorrelationCell = z.infer<typeof CorrelationCell>;
export type CorrelationsFile = z.infer<typeof CorrelationsFile>;
export type ScorecardRow = z.infer<typeof ScorecardRow>;
export type ReturnGroupError = z.infer<typeof ReturnGroupError>;
export type ScorecardFile = z.infer<typeof ScorecardFile>;
export type ShrinkagePoint = z.infer<typeof ShrinkagePoint>;
export type ShrinkageFile = z.infer<typeof ShrinkageFile>;
export type FixtureRow = z.infer<typeof FixtureRow>;
export type FixturesFile = z.infer<typeof FixturesFile>;
export type GoldenSpearmanFile = z.infer<typeof GoldenSpearmanFile>;
export type ReductionCase = z.infer<typeof ReductionCase>;
export type GoldenReductionsFile = z.infer<typeof GoldenReductionsFile>;
export type BoardPlayer = z.infer<typeof BoardPlayer>;
export type BoardFile = z.infer<typeof BoardFile>;
export type PlayerRow = z.infer<typeof PlayerRow>;
export type PlayersFile = z.infer<typeof PlayersFile>;
export type ProjectionFixture = z.infer<typeof ProjectionFixture>;
export type ProjectionTeam = z.infer<typeof ProjectionTeam>;
export type ProjectionPlayer = z.infer<typeof ProjectionPlayer>;
export type ProjectionsFile = z.infer<typeof ProjectionsFile>;
export type ObservationRow = z.infer<typeof ObservationRow>;
export type ObservationsFile = z.infer<typeof ObservationsFile>;
export type SeasonSummary = z.infer<typeof SeasonSummary>;
export type PaperTradeFile = z.infer<typeof PaperTradeFile>;
export type PaperTradeGateCriterion = z.infer<typeof PaperTradeGateCriterion>;
export type PaperTradeFreezeProvenance = z.infer<typeof PaperTradeFreezeProvenance>;
export type PaperTradeSquadLevelGw = z.infer<typeof PaperTradeSquadLevelGw>;
export type PaperTradePlayerLevelGw = z.infer<typeof PaperTradePlayerLevelGw>;
export type PaperTradeGwNote = z.infer<typeof PaperTradeGwNote>;

/**
 * The contract version this app understands. §5.12 requires the build to
 * fail loudly on a mismatch against committed data rather than degrade
 * silently — a file written under a different contract may have the same
 * field names and different meanings.
 */
export const EXPECTED_CONTRACT_VERSION = 1;
export type OwnershipFile = z.infer<typeof OwnershipFile>;
export type OwnershipPlayer = z.infer<typeof OwnershipPlayer>;
export type PricesFile = z.infer<typeof PricesFile>;
export type PriceWatchPlayer = z.infer<typeof PriceWatchPlayer>;
export type PriceTrackRecord = z.infer<typeof PriceTrackRecord>;
export type PriceCycleOutcome = z.infer<typeof PriceCycleOutcome>;
