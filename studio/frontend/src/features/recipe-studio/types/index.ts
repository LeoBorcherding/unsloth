// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Node } from "@xyflow/react";

export type SamplerType =
  | "category"
  | "subcategory"
  | "uniform"
  | "gaussian"
  | "bernoulli"
  | "datetime"
  | "timedelta"
  | "uuid"
  | "person"
  | "person_from_faker";

export type LlmType = "text" | "structured" | "code" | "judge";
export type ValidatorCodeLang =
  | "javascript"
  | "typescript"
  | "jsx"
  | "tsx"
  | "python"
  | "json"
  | "markdown"
  | "sql:sqlite"
  | "sql:postgres"
  | "sql:mysql"
  | "sql:tsql"
  | "sql:bigquery"
  | "sql:ansi";
export type ValidatorType = "code" | "oxc" | "json" | "markdown";
export type OxcValidationMode = "syntax" | "lint" | "syntax+lint";
export type OxcCodeShape = "auto" | "module" | "snippet";

export type ExpressionDtype = "str" | "int" | "float" | "bool";

export type LayoutDirection = "LR" | "TB";

export type SeedSamplingStrategy = "ordered" | "shuffle";
export type SeedSelectionType = "none" | "index_range" | "partition_block";
export type SeedSourceType = "hf" | "local" | "unstructured" | "github_repo";

export type GithubItemType = "issues" | "pulls" | "commits";
export type GithubStateFilter = "all" | "open" | "closed";
export const INFRA_NODE_KINDS = new Set([
  "model_provider",
  "model_config",
  "tool_config",
]);

export type RecipeNodeData = {
  title: string;
  name: string;
  kind:
    | "sampler"
    | "llm"
    | "validator"
    | "expression"
    | "seed"
    | "note"
    | "train"
    | "benchmark"
    | "loop"
    | "model_provider"
    | "model_config"
    | "tool_config";
  subtype: string;
  blockType:
    | SamplerType
    | LlmType
    | "validator_python"
    | "validator_sql"
    | "validator_oxc"
    | "validator_json"
    | "validator_markdown"
    | "expression"
    | "seed"
    | "markdown_note"
    | "train"
    | "benchmark"
    | "loop"
    | "model_provider"
    | "model_config"
    | "tool_config";
  layoutDirection?: LayoutDirection;
  runtimeState?: "idle" | "running" | "done";
  executionLocked?: boolean;
};

export type RecipeNode = Node<RecipeNodeData, "builder">;

export type CategoryConditionalParams = {
  // biome-ignore lint/style/useNamingConvention: api schema
  sampler_type: "category";
  values: string[];
  weights?: Array<number | null>;
};

export type SamplerConfig = {
  id: string;
  kind: "sampler";
  advancedOpen?: boolean;
  // biome-ignore lint/style/useNamingConvention: api schema
  sampler_type: SamplerType;
  name: string;
  drop?: boolean;
  // biome-ignore lint/style/useNamingConvention: api schema
  convert_to?: "float" | "int" | "str";
  values?: string[];
  weights?: Array<number | null>;
  low?: string;
  high?: string;
  mean?: string;
  std?: string;
  p?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  datetime_start?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  datetime_end?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  datetime_unit?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  dt_min?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  dt_max?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  reference_column_name?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  timedelta_unit?: "D" | "h" | "m" | "s";
  // biome-ignore lint/style/useNamingConvention: api schema
  uuid_format?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  person_locale?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  person_sex?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  person_age_range?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  person_city?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  person_with_synthetic_personas?: boolean;
  // biome-ignore lint/style/useNamingConvention: api schema
  subcategory_parent?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  subcategory_mapping?: Record<string, string[]>;
  // biome-ignore lint/style/useNamingConvention: api schema
  conditional_params?: Record<string, CategoryConditionalParams>;
};

export type ScoreOption = {
  value: string;
  description: string;
};

export type Score = {
  name: string;
  description: string;
  options: ScoreOption[];
};

export type McpProviderType = "stdio" | "sse" | "streamable_http";

export type McpEnvVar = {
  key: string;
  value: string;
};

export type LlmMcpProviderConfig = {
  id: string;
  name: string;
  // biome-ignore lint/style/useNamingConvention: ui schema
  provider_type: McpProviderType;
  command?: string;
  args?: string[];
  env?: McpEnvVar[];
  endpoint?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  api_key?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  api_key_env?: string;
};

export type LlmToolConfig = {
  id: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  tool_alias: string;
  providers: string[];
  // biome-ignore lint/style/useNamingConvention: api schema
  allow_tools?: string[];
  // biome-ignore lint/style/useNamingConvention: api schema
  max_tool_call_turns?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  timeout_sec?: string;
};

export type ToolProfileConfig = {
  id: string;
  kind: "tool_config";
  name: string;
  // biome-ignore lint/style/useNamingConvention: ui schema
  mcp_providers: LlmMcpProviderConfig[];
  // biome-ignore lint/style/useNamingConvention: ui schema
  fetched_tools_by_provider?: Record<string, string[]>;
  // biome-ignore lint/style/useNamingConvention: api schema
  allow_tools?: string[];
  // biome-ignore lint/style/useNamingConvention: api schema
  max_tool_call_turns?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  timeout_sec?: string;
};

export type LlmImageContextConfig = {
  enabled: boolean;
  // biome-ignore lint/style/useNamingConvention: api schema
  column_name: string;
};

export type LlmTraceType = "none" | "last_message" | "all_messages";

export type LlmConfig = {
  id: string;
  kind: "llm";
  advancedOpen?: boolean;
  // biome-ignore lint/style/useNamingConvention: api schema
  llm_type: LlmType;
  name: string;
  drop?: boolean;
  // biome-ignore lint/style/useNamingConvention: api schema
  model_alias: string;
  prompt: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  system_prompt: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  code_lang?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  output_format?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  tool_alias?: string;
  scores?: Score[];
  // ui-only, serialized into multi_modal_context
  // biome-ignore lint/style/useNamingConvention: ui schema
  image_context?: LlmImageContextConfig;
  // biome-ignore lint/style/useNamingConvention: api schema
  with_trace?: LlmTraceType;
  // biome-ignore lint/style/useNamingConvention: api schema
  extract_reasoning_content?: boolean;
};

export type ModelProviderConfig = {
  id: string;
  kind: "model_provider";
  name: string;
  endpoint: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  provider_type: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  api_key_env?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  api_key?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  extra_headers?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  extra_body?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  is_local?: boolean;
};

export type ModelConfig = {
  id: string;
  kind: "model_config";
  name: string;
  model: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  gguf_variant?: string;
  provider: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  inference_temperature?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  inference_top_p?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  inference_max_tokens?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  inference_timeout?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  inference_extra_body?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  skip_health_check?: boolean;
};

export type ExpressionConfig = {
  id: string;
  kind: "expression";
  name: string;
  drop?: boolean;
  expr: string;
  dtype: ExpressionDtype;
};

export type ValidatorConfig = {
  id: string;
  kind: "validator";
  advancedOpen?: boolean;
  name: string;
  drop?: boolean;
  // biome-ignore lint/style/useNamingConvention: api schema
  target_columns: string[];
  validator_type: ValidatorType;
  // biome-ignore lint/style/useNamingConvention: api schema
  code_lang: ValidatorCodeLang;
  oxc_validation_mode: OxcValidationMode;
  oxc_code_shape: OxcCodeShape;
  // serialized to int in payload
  batch_size: string;
};

export type MarkdownNoteConfig = {
  id: string;
  kind: "markdown_note";
  name: string;
  markdown: string;
  note_color?: string;
  // ui-only, 0-100 as a string
  note_opacity?: string;
};

export type BenchmarkScorer =
  | "reference"
  | "json"
  | "markdown"
  | "python"
  | "judge";
export type BenchmarkBaselineMode = "none" | "adapter_off" | "alias";

export type BenchmarkScoreSummary = {
  total: number;
  scored: number;
  passed: number;
  errors: number;
  score: number | null;
};

export type BenchmarkFailedExample = {
  prompt: string;
  reference: string | null;
  output: string | null;
  reason: string;
  row: Record<string, unknown>;
};

export type BenchmarkResult = {
  candidate: BenchmarkScoreSummary;
  baseline?: BenchmarkScoreSummary;
  delta?: number | null;
  cancelled: boolean;
  // biome-ignore lint/style/useNamingConvention: api schema
  failed_examples: BenchmarkFailedExample[];
  // biome-ignore lint/style/useNamingConvention: api schema
  failed_total: number;
};

export type BenchmarkConfig = {
  id: string;
  kind: "benchmark";
  name: string;
  // a task id from the Benchmarks registry (/api/benchmarks/evals/tasks)
  task: string;
  // lm-eval tasks: score the first N samples (string for input)
  limit?: string;
  // biome-ignore lint/style/useNamingConvention: ui schema
  model_alias: string;
  // biome-ignore lint/style/useNamingConvention: ui schema
  baseline_mode: BenchmarkBaselineMode;
  // biome-ignore lint/style/useNamingConvention: ui schema
  baseline_alias?: string;
  // biome-ignore lint/style/useNamingConvention: ui schema
  prompt_column: string;
  // biome-ignore lint/style/useNamingConvention: ui schema
  reference_column?: string;
  // biome-ignore lint/style/useNamingConvention: ui schema
  system_prompt?: string;
  scorer: BenchmarkScorer;
  // biome-ignore lint/style/useNamingConvention: ui schema
  judge_alias?: string;
  rubric?: string;
  // ui ergonomics (string for input)
  // biome-ignore lint/style/useNamingConvention: ui schema
  holdout_percent: string;
  // biome-ignore lint/style/useNamingConvention: ui schema
  max_rows: string;
  // ui-only: last finished run, so the card and its output handle survive a reload
  // biome-ignore lint/style/useNamingConvention: ui schema
  last_result?: BenchmarkResult & {
    task?: string;
    // biome-ignore lint/style/useNamingConvention: ui schema
    run_ids?: Record<string, string>;
    // biome-ignore lint/style/useNamingConvention: ui schema
    finished_at: number;
    // biome-ignore lint/style/useNamingConvention: ui schema
    execution_id: string;
    rows: number;
  };
};

export type SeedConfig = {
  id: string;
  kind: "seed";
  advancedOpen?: boolean;
  name: string;
  drop?: boolean;
  seed_drop_columns?: string[];
  seed_source_type: SeedSourceType;
  // ui-only: a local seed the Loop card rewrites each round from the Benchmark card's failures
  // biome-ignore lint/style/useNamingConvention: ui schema
  loop_feedback?: boolean;
  hf_repo_id: string;
  hf_subset?: string;
  hf_split?: string;
  hf_path: string;
  hf_token?: string;
  hf_endpoint?: string;
  local_file_name?: string;
  // Stable per-block upload id, since node ids collide across imports.
  unstructured_upload_uid?: string;
  unstructured_file_ids?: string[];
  unstructured_file_names?: string[];
  unstructured_file_sizes?: number[];
  // biome-ignore lint/style/useNamingConvention: api schema
  github_repo_slug?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  github_token?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  github_limit?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  github_item_types?: GithubItemType[];
  // biome-ignore lint/style/useNamingConvention: api schema
  github_state?: GithubStateFilter;
  // biome-ignore lint/style/useNamingConvention: api schema
  github_include_comments?: boolean;
  // biome-ignore lint/style/useNamingConvention: api schema
  github_max_comments_per_item?: string;
  resolved_paths?: string[];
  seed_preview_rows?: Record<string, unknown>[];
  // ui-only
  seed_total_rows?: number;
  // ui-only (string for input ergonomics)
  unstructured_chunk_size?: string;
  unstructured_chunk_overlap?: string;
  seed_splits?: string[];
  // biome-ignore lint/style/useNamingConvention: ui schema
  seed_globs_by_split?: Record<string, string>;
  seed_columns?: string[];
  sampling_strategy: SeedSamplingStrategy;
  selection_type: SeedSelectionType;
  selection_start?: string;
  selection_end?: string;
  selection_index?: string;
  selection_num_partitions?: string;
};

export type SchemaTransformProcessorConfig = {
  id: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  processor_type: "schema_transform";
  name: string;
  template: string;
};

export type RecipeProcessorConfig = SchemaTransformProcessorConfig;

export type TrainCardMethod = "qlora" | "lora";

/** The last training run a Train card started: what a downstream card reads off its output handle. */
export type TrainCardRun = {
  jobId: string | null;
  executionId: string;
  /** The recipe run the split came from, so the held-out rows can be read back. */
  sourceJobId: string | null;
  artifactPath: string | null;
  baseModel: string;
  outputName: string;
  promptColumn: string;
  responseColumn: string;
  holdoutPercent: number;
  seed: number;
  trainPath: string;
  holdoutPath: string | null;
  holdoutRowIds: number[];
  trainRows: number;
  holdoutRows: number;
  phase: string;
  outputDir: string | null;
  startedAt: number;
};

/** UI-only like markdown_note: not a dataset column, kept in `ui.nodes`. Inputs are strings like the other cards. */
export type TrainCardConfig = {
  id: string;
  kind: "train";
  name: string;
  baseModel: string;
  method: TrainCardMethod;
  promptColumn: string;
  responseColumn: string;
  maxSteps: string;
  learningRate: string;
  loraRank: string;
  loraAlpha: string;
  batchSize: string;
  contextLength: string;
  holdoutPercent: string;
  seed: string;
  outputName: string;
  lastRun?: TrainCardRun | null;
};

/** UI-only like the Train card: the loop's stop rules and round size. Inputs are strings. */
export type LoopCardConfig = {
  id: string;
  kind: "loop";
  name: string;
  targetScore: string;
  maxRounds: string;
  stopOnPlateau: boolean;
  plateauRounds: string;
  rowsPerRound: string;
  failureShare: string;
};

export type NodeConfig =
  | SamplerConfig
  | LlmConfig
  | ValidatorConfig
  | ExpressionConfig
  | MarkdownNoteConfig
  | BenchmarkConfig
  | SeedConfig
  | TrainCardConfig
  | LoopCardConfig
  | ModelProviderConfig
  | ModelConfig
  | ToolProfileConfig;
