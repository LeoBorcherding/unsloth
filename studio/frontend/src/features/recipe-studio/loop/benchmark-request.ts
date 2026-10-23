// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { BenchmarkConfig, NodeConfig } from "../types";
import { BENCHMARK_MAX_ROWS, type RecipeHeldoutSpec } from "../utils/benchmark";
import { buildModelConfig, buildModelProvider } from "../utils/payload/builders-model";

// The round's adapter is whatever Chat has loaded; "local" makes /v1 answer with the resident model.
const CANDIDATE_ALIAS = "loop-candidate";
const CANDIDATE_PROVIDER = "loop-local";

function clampRows(raw: string): number {
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? Math.max(1, Math.min(BENCHMARK_MAX_ROWS, parsed)) : 20;
}

/** The card's row cap, applied to each of the probe and test splits. */
export function loopBenchmarkRowCap(benchmark: BenchmarkConfig): number {
  return clampRows(benchmark.max_rows);
}

/**
 * The Benchmark card's "Recipe held-out" scoring pointed at the round's adapter: same columns and
 * scorer as the card, on one split of the held-out rows.
 */
export function buildLoopBenchmarkRequest(input: {
  benchmark: BenchmarkConfig;
  configs: NodeConfig[];
  rows: Record<string, unknown>[];
  // The base model is the same every round, so only the round that measures it asks for it.
  withBaseline: boolean;
}): { request: RecipeHeldoutSpec | null; errors: string[] } {
  const { benchmark } = input;
  const errors: string[] = [];
  const providers: Record<string, unknown>[] = [
    {
      name: CANDIDATE_PROVIDER,
      endpoint: "",
      // biome-ignore lint/style/useNamingConvention: api schema
      provider_type: "openai",
      // biome-ignore lint/style/useNamingConvention: api schema
      extra_headers: {},
      // biome-ignore lint/style/useNamingConvention: api schema
      extra_body: {},
      // biome-ignore lint/style/useNamingConvention: api schema
      is_local: true,
    },
  ];
  const modelConfigs: Record<string, unknown>[] = [
    {
      alias: CANDIDATE_ALIAS,
      model: "local",
      provider: CANDIDATE_PROVIDER,
      // biome-ignore lint/style/useNamingConvention: api schema
      inference_parameters: { temperature: 0, max_tokens: 256 },
    },
  ];
  const judgeAlias =
    benchmark.scorer === "judge" ? benchmark.judge_alias?.trim() || null : null;
  if (benchmark.scorer === "judge" && !judgeAlias) {
    errors.push("Pick a judge model on the Benchmark card.");
  }
  if (judgeAlias) {
    const judge = input.configs.find(
      (config) => config.kind === "model_config" && config.name === judgeAlias,
    );
    const provider =
      judge?.kind === "model_config"
        ? input.configs.find(
            (config) =>
              config.kind === "model_provider" && config.name === judge.provider,
          )
        : undefined;
    if (judge?.kind !== "model_config" || provider?.kind !== "model_provider") {
      errors.push(`Judge model ${judgeAlias} is not on the canvas.`);
    } else {
      modelConfigs.push(buildModelConfig(judge, errors));
      providers.push(buildModelProvider(provider, errors));
    }
  }
  if (!benchmark.prompt_column.trim()) {
    errors.push("Pick the prompt column on the Benchmark card.");
  }
  if (benchmark.scorer === "reference" && !benchmark.reference_column?.trim()) {
    errors.push("Pick the reference column on the Benchmark card.");
  }
  if (errors.length > 0) {
    return { request: null, errors };
  }
  return {
    request: {
      rows: input.rows,
      // biome-ignore lint/style/useNamingConvention: api schema
      prompt_column: benchmark.prompt_column.trim(),
      // biome-ignore lint/style/useNamingConvention: api schema
      reference_column: benchmark.reference_column?.trim() || null,
      // biome-ignore lint/style/useNamingConvention: api schema
      system_prompt: benchmark.system_prompt ?? "",
      scorer: benchmark.scorer,
      rubric: benchmark.rubric ?? "",
      // biome-ignore lint/style/useNamingConvention: api schema
      model_alias: CANDIDATE_ALIAS,
      // The base model with the adapter switched off is the only baseline one loaded model can give.
      baseline: {
        mode: input.withBaseline && benchmark.baseline_mode === "adapter_off" ? "adapter_off" : "none",
        alias: null,
      },
      // biome-ignore lint/style/useNamingConvention: api schema
      judge_alias: judgeAlias,
      // biome-ignore lint/style/useNamingConvention: api schema
      model_providers: providers,
      // biome-ignore lint/style/useNamingConvention: api schema
      model_configs: modelConfigs,
    },
    errors: [],
  };
}
