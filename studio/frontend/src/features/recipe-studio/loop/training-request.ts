// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainCardSettings } from "../utils/train-card";
import { buildTrainCardMapping } from "../utils/train-card";

/**
 * The training request for one round. Round 1 starts from the Train card's base model; later rounds
 * pass the previous round's adapter folder as the model, which the trainer continues instead of
 * stacking a new adapter.
 */
export function buildLoopTrainingRequest(input: {
  settings: TrainCardSettings;
  trainPath: string;
  continueFrom: string | null;
  round: number;
}): Record<string, unknown> {
  const { settings, trainPath, continueFrom, round } = input;
  return {
    // biome-ignore lint/style/useNamingConvention: api schema
    model_name: continueFrom ?? settings.baseModel,
    // biome-ignore lint/style/useNamingConvention: api schema
    model_format: continueFrom ? "adapter" : null,
    // biome-ignore lint/style/useNamingConvention: api schema
    project_name: `${settings.outputName} round ${round}`.slice(0, 80),
    // biome-ignore lint/style/useNamingConvention: api schema
    training_type: "LoRA/QLoRA",
    // biome-ignore lint/style/useNamingConvention: api schema
    load_in_4bit: settings.method === "qlora",
    // biome-ignore lint/style/useNamingConvention: api schema
    local_datasets: [trainPath],
    // biome-ignore lint/style/useNamingConvention: api schema
    format_type: "auto",
    // biome-ignore lint/style/useNamingConvention: api schema
    custom_format_mapping: buildTrainCardMapping(settings),
    // biome-ignore lint/style/useNamingConvention: api schema
    max_steps: settings.maxSteps,
    // biome-ignore lint/style/useNamingConvention: api schema
    learning_rate: String(settings.learningRate),
    // biome-ignore lint/style/useNamingConvention: api schema
    batch_size: settings.batchSize,
    // biome-ignore lint/style/useNamingConvention: api schema
    gradient_accumulation_steps: 1,
    // biome-ignore lint/style/useNamingConvention: api schema
    lora_r: settings.loraRank,
    // biome-ignore lint/style/useNamingConvention: api schema
    lora_alpha: settings.loraAlpha,
    // biome-ignore lint/style/useNamingConvention: api schema
    max_seq_length: settings.contextLength,
    // biome-ignore lint/style/useNamingConvention: api schema
    random_seed: settings.seed + round - 1,
    // biome-ignore lint/style/useNamingConvention: api schema
    train_on_completions: true,
  };
}
