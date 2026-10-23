// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { getInferenceStatus, loadModel, unloadModel } from "@/features/chat";
import { readFastApiError } from "@/lib/format-fastapi-error";
import { cancelBenchmark, runBenchmark } from "@/features/benchmark/api/benchmark-api";
import { createRecipeTrainingSplit, inspectSeedUpload } from "../api";
import type { RecipeExecutionRecord } from "../execution-types";
import type { BenchmarkResult, NodeConfig, SeedConfig, TrainCardRun } from "../types";
import { RECIPE_HELDOUT_TASK } from "../utils/benchmark";
import { findLoopParts, loopRecipeColumns } from "../utils/loop-card";
import {
  type TrainCardSettings,
  listTrainableColumns,
  parseTrainCardSettings,
} from "../utils/train-card";
import { buildLoopBenchmarkRequest, loopScoreFromResult } from "./benchmark-request";
import { feedbackSeedColumns, feedbackSeedToJsonl } from "./feedback-seed";
import { buildLoopTrainingRequest } from "./training-request";
import type { FeedbackSeedRow, LoopHoldout, LoopSteps } from "./types";

const POLL_MS = 2000;
const TERMINAL_PHASES = new Set(["completed", "error", "stopped"]);
const EXAMPLES_PER_ROUND = 3;

type TrainStatus = {
  // biome-ignore lint/style/useNamingConvention: api schema
  job_id: string;
  phase: string;
  message: string;
  error: string | null;
  // biome-ignore lint/style/useNamingConvention: api schema
  start_request_state?: "pending" | "accepted" | "rejected" | null;
  // biome-ignore lint/style/useNamingConvention: api schema
  details?: { output_dir?: string | null } | null;
};

export type StudioLoopDeps = {
  tag: string;
  recipeName: () => string;
  configs: () => Record<string, NodeConfig>;
  updateConfig: (id: string, patch: Partial<NodeConfig>) => void;
  runRecipe: (rows: number, runName: string) => Promise<RecipeExecutionRecord | null>;
};

function abortError(): Error {
  const error = new Error("Stopped.");
  error.name = "AbortError";
  return error;
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(abortError());
      return;
    }
    const onAbort = () => {
      clearTimeout(timer);
      reject(abortError());
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

function untilAborted(signal: AbortSignal): Promise<never> {
  return new Promise((_, reject) => {
    if (signal.aborted) {
      reject(abortError());
      return;
    }
    signal.addEventListener("abort", () => reject(abortError()), { once: true });
  });
}

function toBase64(text: string): string {
  const bytes = new TextEncoder().encode(text);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

function requireFeedbackSeed(configs: Record<string, NodeConfig>): SeedConfig {
  const { feedback } = findLoopParts(configs);
  if (!feedback) {
    throw new Error("The Benchmark failures seed is gone. Add it back from the Loop card.");
  }
  return feedback;
}

function requireTrainSettings(configs: Record<string, NodeConfig>): {
  id: string;
  settings: TrainCardSettings;
} {
  const { train } = findLoopParts(configs);
  if (!train) {
    throw new Error("The Train card is gone.");
  }
  const { settings, errors } = parseTrainCardSettings(train, listTrainableColumns(configs));
  if (errors.length > 0) {
    throw new Error(`Train card: ${errors[0]}`);
  }
  return { id: train.id, settings };
}

/** Chat, the recipe's generator and training share one GPU: free it before training. */
async function freeInferenceModel(): Promise<void> {
  const status = await getInferenceStatus().catch(() => null);
  const active = status?.active_model?.trim();
  if (active) {
    await unloadModel({ model_path: active });
  }
}

async function readTrainStatus(signal: AbortSignal): Promise<TrainStatus> {
  const res = await authFetch("/api/train/status", { signal });
  if (!res.ok) {
    throw new Error(await readFastApiError(res));
  }
  return (await res.json()) as TrainStatus;
}

async function trainThroughStudio(
  request: Record<string, unknown>,
  signal: AbortSignal,
  onStarted: (jobId: string) => void,
): Promise<{ jobId: string; outputDir: string }> {
  const startRequestId = `recipe-loop-${crypto.randomUUID()}`;
  const res = await authFetch("/api/train/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // biome-ignore lint/style/useNamingConvention: api schema
    body: JSON.stringify({ ...request, start_request_id: startRequestId }),
    signal,
  });
  if (!res.ok) {
    throw new Error(await readFastApiError(res));
  }
  const started = (await res.json()) as {
    // biome-ignore lint/style/useNamingConvention: api schema
    job_id: string;
    status: string;
    message: string;
    error?: string | null;
  };
  if (started.status === "error") {
    void authFetch(
      `/api/train/start-requests/${encodeURIComponent(startRequestId)}/acknowledge`,
      { method: "POST" },
    ).catch(() => undefined);
    throw new Error(started.error || started.message || "Training refused.");
  }
  const jobId = started.job_id;
  onStarted(jobId);
  let outputDir: string | null = null;
  try {
    while (true) {
      await sleep(POLL_MS, signal);
      const status = await readTrainStatus(signal);
      if (status.job_id !== jobId) {
        continue;
      }
      outputDir = status.details?.output_dir ?? outputDir;
      if (status.start_request_state === "rejected") {
        throw new Error(status.error || status.message);
      }
      if (!TERMINAL_PHASES.has(status.phase)) {
        continue;
      }
      if (status.phase === "completed") {
        break;
      }
      throw new Error(
        status.error || (status.phase === "stopped" ? "Training was stopped." : status.message),
      );
    }
  } catch (error) {
    if (signal.aborted) {
      await authFetch("/api/train/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // biome-ignore lint/style/useNamingConvention: api schema
        body: JSON.stringify({ save: false, expected_job_id: jobId }),
      }).catch(() => undefined);
    }
    throw error;
  }
  if (!outputDir) {
    throw new Error("Training finished without reporting where it saved the adapter.");
  }
  return { jobId, outputDir };
}

/** Uploads one round's feedback seed rows; the patch points the failures seed at the new file. */
export async function uploadFeedbackSeed(
  seedRows: FeedbackSeedRow[],
  round: number,
  recipeColumns: string[],
): Promise<Partial<SeedConfig>> {
  const filename = `benchmark-failures-round-${round}.jsonl`;
  const upload = await inspectSeedUpload({
    filename,
    // biome-ignore lint/style/useNamingConvention: api schema
    content_base64: toBase64(feedbackSeedToJsonl(seedRows)),
    // biome-ignore lint/style/useNamingConvention: api schema
    preview_size: 5,
    // biome-ignore lint/style/useNamingConvention: api schema
    seed_source_type: "local",
  });
  return {
    // biome-ignore lint/style/useNamingConvention: api schema
    seed_source_type: "local",
    // biome-ignore lint/style/useNamingConvention: api schema
    hf_path: upload.resolved_path,
    // biome-ignore lint/style/useNamingConvention: api schema
    local_file_name: filename,
    // biome-ignore lint/style/useNamingConvention: api schema
    seed_columns: feedbackSeedColumns(recipeColumns),
    // biome-ignore lint/style/useNamingConvention: api schema
    seed_preview_rows: upload.preview_rows,
    // biome-ignore lint/style/useNamingConvention: api schema
    seed_total_rows: seedRows.length,
    // biome-ignore lint/style/useNamingConvention: api schema
    sampling_strategy: "ordered",
    // biome-ignore lint/style/useNamingConvention: api schema
    selection_type: "none",
  };
}

/** Generation through the recipe, training through Unsloth Studio's trainer, scoring through the Benchmark card's API. */
export function createStudioLoopSteps(deps: StudioLoopDeps): LoopSteps {
  return {
    generate: async ({ round, seedRows, signal }) => {
      const configs = deps.configs();
      const feedback = requireFeedbackSeed(configs);
      deps.updateConfig(
        feedback.id,
        (await uploadFeedbackSeed(seedRows, round, loopRecipeColumns(configs))) as Partial<NodeConfig>,
      );

      const base = deps.recipeName().trim() || "Recipe";
      const runName = `${base} loop ${deps.tag} round ${round}`;
      const execution = await Promise.race([
        deps.runRecipe(seedRows.length, runName),
        untilAborted(signal),
      ]);
      if (!execution) {
        throw new Error("Generating the round's rows failed. The Runs tab has the details.");
      }
      if (!(execution.jobId && execution.artifact_path)) {
        throw new Error("The recipe run finished without saving a dataset.");
      }
      const { train } = findLoopParts(deps.configs());
      const writtenColumn = train?.promptColumn.trim() ?? "";
      const examples = execution.dataset
        .filter((row) => row.origin === "failure")
        .slice(0, EXAMPLES_PER_ROUND)
        .map((row) => ({
          failed: String(row.failed_prompt ?? ""),
          written: String(row[writtenColumn] ?? ""),
        }));
      return {
        executionId: execution.id,
        jobId: execution.jobId,
        artifactPath: execution.artifact_path,
        runName,
        rows: execution.datasetTotal ?? execution.rows,
        fromFailures: seedRows.filter((row) => row.origin === "failure").length,
        examples,
      };
    },

    train: async ({ round, dataset, holdout, continueFrom, signal }) => {
      const { id: trainId, settings } = requireTrainSettings(deps.configs());
      // Round 1 carves the held-out rows; later rounds train on all of their new rows.
      const split = await createRecipeTrainingSplit(dataset.jobId, {
        // biome-ignore lint/style/useNamingConvention: api schema
        artifact_path: dataset.artifactPath,
        // biome-ignore lint/style/useNamingConvention: api schema
        holdout_percent: holdout ? 0 : settings.holdoutPercent,
        seed: settings.seed,
      });
      let fixed: LoopHoldout;
      if (holdout) {
        fixed = holdout;
      } else {
        if (!split.holdout_path) {
          throw new Error("Round 1 held no rows out. Raise the Train card's hold out.");
        }
        fixed = {
          sourceJobId: dataset.jobId,
          artifactPath: dataset.artifactPath,
          holdoutPath: split.holdout_path,
          rows: split.holdout_rows,
        };
      }
      await freeInferenceModel();
      const lastRun = (jobId: string | null, phase: string, outputDir: string | null) =>
        ({
          jobId,
          executionId: dataset.executionId,
          sourceJobId: fixed.sourceJobId,
          artifactPath: fixed.artifactPath,
          baseModel: settings.baseModel,
          outputName: settings.outputName,
          promptColumn: settings.promptColumn,
          responseColumn: settings.responseColumn,
          holdoutPercent: settings.holdoutPercent,
          seed: settings.seed,
          trainPath: split.train_path,
          holdoutPath: fixed.holdoutPath,
          holdoutRowIds: [],
          trainRows: split.train_rows,
          holdoutRows: fixed.rows,
          phase,
          outputDir,
          startedAt: Date.now(),
        }) satisfies TrainCardRun;
      const run = await trainThroughStudio(
        buildLoopTrainingRequest({ settings, trainPath: split.train_path, continueFrom, round }),
        signal,
        (jobId) => deps.updateConfig(trainId, { lastRun: lastRun(jobId, "starting", null) }),
      );
      deps.updateConfig(trainId, {
        lastRun: lastRun(run.jobId, "completed", run.outputDir),
      });
      return {
        adapter: {
          jobId: run.jobId,
          outputDir: run.outputDir,
          continuedFrom: continueFrom,
          trainRows: split.train_rows,
        },
        holdout: fixed,
      };
    },

    score: async ({ adapter, holdout, signal }) => {
      const configs = deps.configs();
      const { benchmark } = findLoopParts(configs);
      if (!benchmark) {
        throw new Error("The Benchmark card is gone.");
      }
      const { settings } = requireTrainSettings(configs);
      const { request, errors } = buildLoopBenchmarkRequest({
        benchmark,
        configs: Object.values(configs),
        holdoutPath: holdout.holdoutPath,
      });
      if (!request) {
        throw new Error(errors.join(" "));
      }
      await Promise.race([
        loadModel({
          // biome-ignore lint/style/useNamingConvention: api schema
          model_path: adapter.outputDir,
          // biome-ignore lint/style/useNamingConvention: api schema
          hf_token: null,
          // biome-ignore lint/style/useNamingConvention: api schema
          max_seq_length: settings.contextLength,
          // biome-ignore lint/style/useNamingConvention: api schema
          load_in_4bit: settings.method === "qlora",
          // biome-ignore lint/style/useNamingConvention: api schema
          is_lora: true,
        }),
        untilAborted(signal),
      ]);
      const status = await getInferenceStatus();
      const onAbort = () => void cancelBenchmark().catch(() => undefined);
      signal.addEventListener("abort", onAbort, { once: true });
      try {
        // The Benchmark card's Recipe held-out task, run as a Benchmarks evals run.
        const response = await Promise.race([
          runBenchmark({
            // biome-ignore lint/style/useNamingConvention: api schema
            checkpoint_path: status.active_model ?? adapter.outputDir,
            // biome-ignore lint/style/useNamingConvention: api schema
            model_source: "local",
            task: RECIPE_HELDOUT_TASK,
            recipe: request,
          }),
          untilAborted(signal),
        ]);
        if (signal.aborted || response.details?.cancelled) {
          throw abortError();
        }
        if (!response.success) {
          throw new Error(response.message || "The benchmark failed.");
        }
        const result = response.details?.recipe as
          | (BenchmarkResult & {
              // biome-ignore lint/style/useNamingConvention: api schema
              run_ids?: Record<string, string>;
            })
          | undefined;
        if (!result) {
          throw new Error("The benchmark returned no result.");
        }
        deps.updateConfig(benchmark.id, {
          // biome-ignore lint/style/useNamingConvention: ui schema
          last_result: {
            ...result,
            task: RECIPE_HELDOUT_TASK,
            // biome-ignore lint/style/useNamingConvention: ui schema
            run_ids: result.run_ids,
            // biome-ignore lint/style/useNamingConvention: ui schema
            finished_at: Date.now(),
            // biome-ignore lint/style/useNamingConvention: ui schema
            execution_id: "",
            rows: result.candidate.total,
          },
        });
        return loopScoreFromResult(result);
      } finally {
        signal.removeEventListener("abort", onAbort);
      }
    },
  };
}
