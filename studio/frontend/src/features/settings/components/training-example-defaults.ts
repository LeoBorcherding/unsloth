// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Training starts from a hub repo, unlike chat snippets, which must name what /v1/models serves.
export const TRAIN = {
  model: "unsloth/Llama-3.2-1B-Instruct",
  dataset: "mlabonne/FineTome-100k",
  maxSteps: 60,
  imageBase: "stabilityai/stable-diffusion-xl-base-1.0",
  imageData: "my-images",
  imageOut: "my-images-lora",
  imagePrompt: "a photo of sks cat",
  imageSteps: 500,
} as const;
