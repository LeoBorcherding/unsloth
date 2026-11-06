// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MergeMethod, MergeModelsParams } from "../api/export-api";

interface MergeMethodInfo {
  value: MergeMethod;
  label: string;
  hint: string;
  /** Merges each model's changes over a separate base model. */
  base: boolean;
  density: boolean;
}

export const MERGE_METHODS: readonly MergeMethodInfo[] = [
  {
    value: "linear",
    label: "Linear",
    hint: "A weighted average of the models.",
    base: false,
    density: false,
  },
  {
    value: "slerp",
    label: "SLERP",
    hint: "A spherical blend of two models. The first is the base: t = 0 keeps it, t = 1 gives the second.",
    base: false,
    density: false,
  },
  {
    value: "ties",
    label: "TIES",
    hint: "Adds each model's largest changes over the base, resolving sign conflicts.",
    base: true,
    density: true,
  },
  {
    value: "dare_ties",
    label: "DARE TIES",
    hint: "TIES with each model's changes randomly pruned and rescaled.",
    base: true,
    density: true,
  },
  {
    value: "task_arithmetic",
    label: "Task arithmetic",
    hint: "Adds each model's changes over the base.",
    base: true,
    density: false,
  },
];

export const MERGE_DTYPES = ["bfloat16", "float16", "float32"] as const;

export interface MergeRow {
  model: string;
  weight: number;
  density: number;
}

export interface MergeSettings {
  method: MergeMethod;
  rows: MergeRow[];
  baseModel: string;
  t: number;
  dtype: (typeof MERGE_DTYPES)[number];
  outputName: string;
  cuda: boolean;
}

export const DEFAULT_MERGE_SETTINGS: MergeSettings = {
  method: "ties",
  rows: [
    { model: "", weight: 0.5, density: 0.5 },
    { model: "", weight: 0.5, density: 0.5 },
  ],
  baseModel: "",
  t: 0.5,
  dtype: "bfloat16",
  outputName: "merged-model",
  cuda: false,
};

export function mergeMethodInfo(method: MergeMethod): MergeMethodInfo {
  return MERGE_METHODS.find((m) => m.value === method) ?? MERGE_METHODS[0];
}

/** Why the merge cannot start yet, mirroring the backend's request validation; null when it can. */
export function mergeBlockReason(s: MergeSettings): string | null {
  const info = mergeMethodInfo(s.method);
  const names = s.rows.map((r) => r.model.trim());
  if (names.some((n) => !n)) return "Fill in every model.";
  if (new Set(names).size !== names.length) {
    return "Each model may appear only once.";
  }
  if (s.method === "slerp" && names.length !== 2) {
    return "SLERP merges exactly two models.";
  }
  if (s.method === "linear" && names.length < 2) {
    return "Add at least two models.";
  }
  if (info.base) {
    const base = s.baseModel.trim();
    if (!base) return `${info.label} needs a base model.`;
    if (names.includes(base)) {
      return "The base model goes in the base field only.";
    }
  }
  if (s.method === "slerp" && !(s.t >= 0 && s.t <= 1)) {
    return "t must be between 0 and 1.";
  }
  if (info.density && s.rows.some((r) => !(r.density > 0 && r.density <= 1))) {
    return "Density must be above 0 and at most 1.";
  }
  if (s.method !== "slerp" && s.rows.some((r) => !Number.isFinite(r.weight))) {
    return "Every model needs a weight.";
  }
  const name = s.outputName.trim();
  if (!name || name.startsWith(".") || /[/\\:*?"<>|]/.test(name)) {
    return "Use a plain folder name for the output.";
  }
  return null;
}

export function mergeRequest(
  s: MergeSettings,
): Omit<MergeModelsParams, "hf_token" | "install_missing_dependencies"> {
  const info = mergeMethodInfo(s.method);
  const slerp = s.method === "slerp";
  return {
    method: s.method,
    models: s.rows.map((r) => ({
      model: r.model.trim(),
      ...(slerp ? {} : { weight: r.weight }),
      ...(info.density ? { density: r.density } : {}),
    })),
    base_model: slerp
      ? s.rows[0].model.trim()
      : info.base
        ? s.baseModel.trim()
        : null,
    t: slerp ? s.t : null,
    dtype: s.dtype,
    output_name: s.outputName.trim(),
    cuda: s.cuda,
  };
}
