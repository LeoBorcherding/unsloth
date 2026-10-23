// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { BenchmarkConfig, BenchmarkResult, NodeConfig } from "../types";
import { BENCHMARK_MAX_ROWS, type RecipeHeldoutSpec } from "../utils/benchmark";
import { buildModelConfig, buildModelProvider } from "../utils/payload/builders-model";
import type { LoopScore } from "./types";

// The round's adapter is whatever Chat has loaded; "local" makes /v1 answer with the resident model.
const CANDIDATE_ALIAS = "loop-candidate";
const CANDIDATE_PROVIDER = "loop-local";

function clampRows(raw: string): number {
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? Math.max(1, Math.min(BENCHMARK_MAX_ROWS, parsed)) : 20;
}

/**
 * The Benchmark card's "Recipe held-out" scoring pointed at the round's adapter: same columns,
 * scorer and row cap as the card, rows read server side from the Train card's held-out split.
 */
export function buildLoopBenchmarkRequest(input: {
  benchmark: BenchmarkConfig;
  configs: NodeConfig[];
  holdoutPath: string;
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
      rows: [],
      // biome-ignore lint/style/useNamingConvention: api schema
      holdout_path: input.holdoutPath,
      // biome-ignore lint/style/useNamingConvention: api schema
      max_rows: clampRows(benchmark.max_rows),
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
        mode: benchmark.baseline_mode === "adapter_off" ? "adapter_off" : "none",
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

/** A finished benchmark as a round's score; a run with nothing scored is an error, not a 0. */
export function loopScoreFromResult(result: BenchmarkResult): LoopScore {
  const { candidate } = result;
  if (candidate.score === null || candidate.scored === 0) {
    throw new Error(
      candidate.errors > 0
        ? `Every benchmark row errored (${candidate.errors}). Check that the adapter loaded in Chat.`
        : "The benchmark scored no rows.",
    );
  }
  return {
    score: candidate.score,
    passed: candidate.passed,
    scored: candidate.scored,
    baseline: result.baseline?.score ?? null,
    failedExamples: result.failed_examples.map((example) => ({
      prompt: example.prompt,
      reference: example.reference,
      output: example.output,
      row: example.row,
    })),
    failedTotal: result.failed_total,
  };
}
