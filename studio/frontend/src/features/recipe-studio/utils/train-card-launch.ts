// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  isTrainingStartPending,
  useTrainingConfigStore,
  useTrainingRuntimeStore,
} from "@/features/training";
import {
  type HoldoutRowsResponse,
  createRecipeTrainingSplit,
  getRecipeHoldoutRows,
} from "../api";
import type { RecipeExecutionRecord } from "../execution-types";
import type { TrainCardRun } from "../types";
import {
  type TrainCardHoldout,
  type TrainCardSettings,
  buildTrainCardMapping,
} from "./train-card";

const SETUP_TIMEOUT_MS = 120_000;

function waitForTrainingStore(
  ready: (state: ReturnType<typeof useTrainingConfigStore.getState>) => boolean,
  what: string,
): Promise<void> {
  if (ready(useTrainingConfigStore.getState())) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      unsubscribe();
      reject(new Error(`Timed out ${what}.`));
    }, SETUP_TIMEOUT_MS);
    const unsubscribe = useTrainingConfigStore.subscribe((state) => {
      if (ready(state)) {
        clearTimeout(timer);
        unsubscribe();
        resolve();
      }
    });
  });
}

/**
 * Carves the held-out split, loads the card's settings into the shared training config (the
 * same one the Train page shows, so the run appears there too) and starts it.
 */
export async function launchTrainCardRun(
  settings: TrainCardSettings,
  execution: RecipeExecutionRecord,
  startTrainingRun: () => Promise<boolean>,
): Promise<TrainCardRun> {
  if (isTrainingStartPending(useTrainingRuntimeStore.getState())) {
    throw new Error("Another training run is active. Stop it first.");
  }
  if (!(execution.jobId && execution.artifact_path)) {
    throw new Error("This recipe run has no saved dataset to train on.");
  }
  const split = await createRecipeTrainingSplit(execution.jobId, {
    // biome-ignore lint/style/useNamingConvention: api schema
    artifact_path: execution.artifact_path,
    // biome-ignore lint/style/useNamingConvention: api schema
    holdout_percent: settings.holdoutPercent,
    seed: settings.seed,
  });

  const store = useTrainingConfigStore.getState();
  store.reset();
  store.setSelectedModel(settings.baseModel);
  await waitForTrainingStore(
    (state) => !state.isLoadingModelDefaults,
    "loading the model's defaults",
  );
  const loaded = useTrainingConfigStore.getState();
  if (loaded.modelDefaultsError) {
    throw new Error(loaded.modelDefaultsError);
  }
  loaded.selectLocalDataset(split.train_path);
  await waitForTrainingStore(
    (state) => !state.isCheckingDataset,
    "checking the dataset",
  );

  const configured = useTrainingConfigStore.getState();
  configured.setTrainingMethod(settings.method);
  configured.setMaxSteps(settings.maxSteps);
  configured.setLearningRate(settings.learningRate);
  configured.setLoraRank(settings.loraRank);
  configured.setLoraAlpha(settings.loraAlpha);
  configured.setBatchSize(settings.batchSize);
  configured.setContextLength(settings.contextLength);
  configured.setRandomSeed(settings.seed);
  configured.setProjectName(settings.outputName);
  configured.setDatasetManualMapping(buildTrainCardMapping(settings));

  const started = await startTrainingRun();
  const runtime = useTrainingRuntimeStore.getState();
  if (!started) {
    throw new Error(runtime.startError || "Training did not start.");
  }
  return {
    jobId: runtime.jobId,
    executionId: execution.id,
    sourceJobId: execution.jobId,
    artifactPath: execution.artifact_path,
    baseModel: settings.baseModel,
    outputName: settings.outputName,
    promptColumn: settings.promptColumn,
    responseColumn: settings.responseColumn,
    holdoutPercent: settings.holdoutPercent,
    seed: settings.seed,
    trainPath: split.train_path,
    holdoutPath: split.holdout_path,
    holdoutRowIds: split.holdout_row_ids ?? [],
    trainRows: split.train_rows,
    holdoutRows: split.holdout_rows,
    phase: runtime.phase === "idle" ? "starting" : runtime.phase,
    outputDir: null,
    startedAt: Date.now(),
  };
}

/** Reads back the exact rows a Train card held out, for a scorer downstream of its output handle. */
export function readTrainCardHoldoutRows(
  holdout: TrainCardHoldout,
  page: { offset?: number; limit?: number } = {},
): Promise<HoldoutRowsResponse> {
  return getRecipeHoldoutRows(holdout.source_job_id, {
    // biome-ignore lint/style/useNamingConvention: api schema
    artifact_path: holdout.artifact_path,
    // biome-ignore lint/style/useNamingConvention: api schema
    holdout_percent: holdout.percent,
    seed: holdout.seed,
    offset: page.offset ?? 0,
    limit: page.limit ?? 20,
  });
}
