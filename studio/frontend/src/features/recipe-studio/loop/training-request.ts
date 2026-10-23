// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LoopDataset } from "./types";

export function leafName(path: string | null | undefined): string | null {
  if (!path) return null;
  const parts = path.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] ?? null;
}

/** A recipe dataset as the trainer's local dataset resolver names it. */
export function recipeDatasetRef(dataset: LoopDataset): string {
  return `recipes/${dataset.version}/parquet-files`;
}

/** The folder a recipe run wrote, from the artifact path its execution recorded. */
export function datasetVersionFromArtifact(
  artifactPath: string | null,
): string | null {
  return leafName(artifactPath);
}

export function buildLoopTrainingRequest(input: {
  datasets: LoopDataset[];
  holdout: LoopDataset | null;
  config: { modelName: string; epochs: number; learningRate: string };
  iteration: number;
}): Record<string, unknown> {
  const { datasets, holdout, config } = input;
  return {
    model_name: config.modelName,
    project_name: "Recipe loop",
    training_type: "LoRA/QLoRA",
    format_type: "auto",
    local_datasets: datasets.map(recipeDatasetRef),
    local_eval_datasets: holdout ? [recipeDatasetRef(holdout)] : [],
    // Two evals per run; the last one is the iteration's score.
    eval_steps: 0.5,
    num_epochs: config.epochs,
    batch_size: 8,
    gradient_accumulation_steps: 1,
    learning_rate: config.learningRate,
    load_in_4bit: true,
    max_seq_length: 1024,
    random_seed: 3407 + input.iteration,
  };
}
