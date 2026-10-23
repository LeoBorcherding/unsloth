// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { RecipeExecutionRecord } from "../execution-types";
import type {
  BenchmarkConfig,
  BenchmarkFailedExample,
  BenchmarkResult,
  ModelConfig,
  ModelProviderConfig,
  NodeConfig,
} from "../types";
import { buildModelConfig, buildModelProvider } from "./payload/builders-model";

export const BENCHMARK_MAX_ROWS = 200;

export type BenchmarkHoldout = {
  offset: number;
  limit: number;
  holdout: number;
};

function clampInt(raw: string, fallback: number, min: number, max: number): number {
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, parsed));
}

// The held-out split is the tail of the run: a Train card trains on rows [0, offset) only.
export function getBenchmarkHoldout(
  total: number,
  holdoutPercent: string,
  maxRows: string,
): BenchmarkHoldout {
  if (total <= 0) {
    return { offset: 0, limit: 0, holdout: 0 };
  }
  const percent = clampInt(holdoutPercent, 20, 1, 100);
  const holdout = Math.min(total, Math.max(1, Math.ceil((total * percent) / 100)));
  const cap = clampInt(maxRows, 20, 1, BENCHMARK_MAX_ROWS);
  const offset = total - holdout;
  return { offset, limit: Math.min(holdout, cap), holdout };
}

export function pickBenchmarkExecution(
  executions: RecipeExecutionRecord[],
): RecipeExecutionRecord | null {
  const finished = executions.filter(
    (execution) =>
      execution.status === "completed" &&
      (execution.jobId || execution.dataset.length > 0),
  );
  finished.sort((a, b) => (b.finishedAt ?? b.createdAt) - (a.finishedAt ?? a.createdAt));
  return finished.find((execution) => execution.kind === "full") ?? finished[0] ?? null;
}

export function getBenchmarkColumns(
  execution: RecipeExecutionRecord | null,
): string[] {
  const columns = new Set<string>();
  for (const row of execution?.dataset ?? []) {
    for (const key of Object.keys(row)) {
      columns.add(key);
    }
  }
  return Array.from(columns);
}

export function getBenchmarkAliases(config: BenchmarkConfig): string[] {
  const aliases = [config.model_alias.trim()];
  if (config.baseline_mode === "alias" && config.baseline_alias?.trim()) {
    aliases.push(config.baseline_alias.trim());
  }
  if (config.scorer === "judge" && config.judge_alias?.trim()) {
    aliases.push(config.judge_alias.trim());
  }
  return Array.from(new Set(aliases.filter(Boolean)));
}

export function validateBenchmarkConfig(config: BenchmarkConfig): string[] {
  const errors: string[] = [];
  if (!config.model_alias.trim()) {
    errors.push("Connect or pick the model to test.");
  }
  if (!config.prompt_column.trim()) {
    errors.push("Pick the column with the prompt.");
  }
  if (config.scorer === "reference" && !config.reference_column?.trim()) {
    errors.push("Reference match needs a reference column.");
  }
  if (config.scorer === "judge" && !config.judge_alias?.trim()) {
    errors.push("Pick a judge model.");
  }
  if (config.baseline_mode === "alias" && !config.baseline_alias?.trim()) {
    errors.push("Pick a baseline model.");
  }
  return errors;
}

export type BenchmarkRequest = {
  rows: Record<string, unknown>[];
  // biome-ignore lint/style/useNamingConvention: api schema
  prompt_column: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  reference_column: string | null;
  // biome-ignore lint/style/useNamingConvention: api schema
  system_prompt: string;
  scorer: BenchmarkConfig["scorer"];
  rubric: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  model_alias: string;
  baseline: { mode: BenchmarkConfig["baseline_mode"]; alias: string | null };
  // biome-ignore lint/style/useNamingConvention: api schema
  judge_alias: string | null;
  // biome-ignore lint/style/useNamingConvention: api schema
  model_providers: Record<string, unknown>[];
  // biome-ignore lint/style/useNamingConvention: api schema
  model_configs: Record<string, unknown>[];
};

export function buildBenchmarkRequest(
  config: BenchmarkConfig,
  configs: NodeConfig[],
  rows: Record<string, unknown>[],
): { request: BenchmarkRequest | null; errors: string[] } {
  const errors = validateBenchmarkConfig(config);
  if (rows.length === 0) {
    errors.push("No held-out rows. Run the recipe first.");
  }
  const modelConfigs = new Map<string, ModelConfig>();
  const providers = new Map<string, ModelProviderConfig>();
  for (const item of configs) {
    if (item.kind === "model_config") {
      modelConfigs.set(item.name, item);
    } else if (item.kind === "model_provider") {
      providers.set(item.name, item);
    }
  }
  const builtConfigs: Record<string, unknown>[] = [];
  const builtProviders = new Map<string, Record<string, unknown>>();
  for (const alias of getBenchmarkAliases(config)) {
    const modelConfig = modelConfigs.get(alias);
    if (!modelConfig) {
      errors.push(`Model ${alias} is not on the canvas.`);
      continue;
    }
    builtConfigs.push(buildModelConfig(modelConfig, errors));
    const provider = providers.get(modelConfig.provider);
    if (!provider) {
      errors.push(`Model ${alias} has no provider.`);
      continue;
    }
    if (!builtProviders.has(provider.name)) {
      builtProviders.set(provider.name, buildModelProvider(provider, errors));
    }
  }
  if (errors.length > 0) {
    return { request: null, errors };
  }
  return {
    request: {
      rows,
      // biome-ignore lint/style/useNamingConvention: api schema
      prompt_column: config.prompt_column.trim(),
      // biome-ignore lint/style/useNamingConvention: api schema
      reference_column: config.reference_column?.trim() || null,
      // biome-ignore lint/style/useNamingConvention: api schema
      system_prompt: config.system_prompt ?? "",
      scorer: config.scorer,
      rubric: config.rubric ?? "",
      // biome-ignore lint/style/useNamingConvention: api schema
      model_alias: config.model_alias.trim(),
      baseline: {
        mode: config.baseline_mode,
        alias:
          config.baseline_mode === "alias"
            ? config.baseline_alias?.trim() || null
            : null,
      },
      // biome-ignore lint/style/useNamingConvention: api schema
      judge_alias:
        config.scorer === "judge" ? config.judge_alias?.trim() || null : null,
      // biome-ignore lint/style/useNamingConvention: api schema
      model_providers: Array.from(builtProviders.values()),
      // biome-ignore lint/style/useNamingConvention: api schema
      model_configs: builtConfigs,
    },
    errors: [],
  };
}

export function formatBenchmarkScore(score: number | null | undefined): string {
  return typeof score === "number" ? `${Math.round(score * 100)}%` : "n/a";
}

export function formatBenchmarkDelta(delta: number | null | undefined): string {
  if (typeof delta !== "number") {
    return "n/a";
  }
  const points = Math.round(delta * 100);
  return `${points > 0 ? "+" : ""}${points} pts`;
}

// Output handle contract: what a downstream card (loop controller, regenerate step) reads.
export type BenchmarkOutput = {
  score: number | null;
  // biome-ignore lint/style/useNamingConvention: handle schema
  baseline_score: number | null;
  delta: number | null;
  // biome-ignore lint/style/useNamingConvention: handle schema
  failed_examples: BenchmarkFailedExample[];
};

export function getBenchmarkOutput(config: BenchmarkConfig): BenchmarkOutput | null {
  const result = config.last_result;
  if (!result) {
    return null;
  }
  return {
    score: result.candidate.score,
    // biome-ignore lint/style/useNamingConvention: handle schema
    baseline_score: result.baseline?.score ?? null,
    delta: result.delta ?? null,
    // biome-ignore lint/style/useNamingConvention: handle schema
    failed_examples: result.failed_examples,
  };
}

export function failedExamplesToJsonl(result: BenchmarkResult): string {
  return result.failed_examples
    .map((example) =>
      JSON.stringify({
        prompt: example.prompt,
        reference: example.reference,
        output: example.output,
        reason: example.reason,
      }),
    )
    .join("\n");
}

const BENCHMARK_SCORERS = new Set(["reference", "json", "markdown", "python", "judge"]);
const BENCHMARK_BASELINES = new Set(["none", "adapter_off", "alias"]);

export function serializeBenchmarkSettings(
  config: BenchmarkConfig,
): Record<string, unknown> {
  const { id: _id, kind: _kind, name: _name, ...settings } = config;
  return settings;
}

function readText(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

export function parseBenchmarkSettings(
  id: string,
  name: string,
  raw: unknown,
): BenchmarkConfig {
  const input = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  const scorer = readText(input.scorer) ?? "";
  const baseline = readText(input.baseline_mode) ?? "";
  const lastResult = input.last_result;
  return {
    id,
    kind: "benchmark",
    name,
    // biome-ignore lint/style/useNamingConvention: ui schema
    model_alias: readText(input.model_alias) ?? "",
    // biome-ignore lint/style/useNamingConvention: ui schema
    baseline_mode: (BENCHMARK_BASELINES.has(baseline)
      ? baseline
      : "none") as BenchmarkConfig["baseline_mode"],
    // biome-ignore lint/style/useNamingConvention: ui schema
    baseline_alias: readText(input.baseline_alias),
    // biome-ignore lint/style/useNamingConvention: ui schema
    prompt_column: readText(input.prompt_column) ?? "",
    // biome-ignore lint/style/useNamingConvention: ui schema
    reference_column: readText(input.reference_column),
    // biome-ignore lint/style/useNamingConvention: ui schema
    system_prompt: readText(input.system_prompt),
    scorer: (BENCHMARK_SCORERS.has(scorer)
      ? scorer
      : "reference") as BenchmarkConfig["scorer"],
    // biome-ignore lint/style/useNamingConvention: ui schema
    judge_alias: readText(input.judge_alias),
    rubric: readText(input.rubric),
    // biome-ignore lint/style/useNamingConvention: ui schema
    holdout_percent: readText(input.holdout_percent) ?? "20",
    // biome-ignore lint/style/useNamingConvention: ui schema
    max_rows: readText(input.max_rows) ?? "20",
    ...(lastResult && typeof lastResult === "object" && "candidate" in lastResult
      ? // biome-ignore lint/style/useNamingConvention: ui schema
        { last_result: lastResult as BenchmarkConfig["last_result"] }
      : {}),
  };
}
