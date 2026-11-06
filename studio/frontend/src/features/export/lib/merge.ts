// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  MergeGateMode,
  MergeMethod,
  MergeModelsParams,
  MoeArchitecture,
} from "../api/export-api";

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
  {
    value: "moe",
    label: "Mixture of experts",
    hint: "Puts each model's MLP layers behind a router; the rest comes from the base. Prompts steer which expert takes each token.",
    base: true,
    density: false,
  },
];

export const MERGE_GATE_MODES: readonly {
  value: MergeGateMode;
  label: string;
}[] = [
  { value: "hidden", label: "Hidden states (best, runs the base)" },
  { value: "cheap_embed", label: "Token embeddings (fast)" },
  { value: "random", label: "Random (for training afterwards)" },
  {
    value: "uniform_random",
    label: "Uniform random (for training afterwards)",
  },
];

export const MOE_ARCHITECTURES: readonly {
  value: MoeArchitecture | "auto";
  label: string;
}[] = [
  { value: "auto", label: "Automatic" },
  { value: "mixtral", label: "Mixtral (Llama, Mistral)" },
  { value: "qwen3", label: "Qwen3 MoE (Qwen3)" },
  { value: "deepseek", label: "DeepSeek MoE (Llama, Mistral)" },
];

export const MERGE_DTYPES = ["bfloat16", "float16", "float32"] as const;

export const MERGE_TOKENIZERS = [
  { value: "base", label: "Base model's" },
  { value: "union", label: "Union of all models" },
] as const;

export interface MergeRow {
  model: string;
  weight: number;
  density: number;
  /** MoE only, as typed: one prompt per line, or comma separated. */
  prompts: string;
}

export interface MergeSettings {
  method: MergeMethod;
  rows: MergeRow[];
  baseModel: string;
  t: number;
  dtype: (typeof MERGE_DTYPES)[number];
  tokenizerSource: (typeof MERGE_TOKENIZERS)[number]["value"];
  gateMode: MergeGateMode;
  expertsPerToken: number;
  architecture: (typeof MOE_ARCHITECTURES)[number]["value"];
  outputName: string;
  cuda: boolean;
}

export const EMPTY_MERGE_ROW: MergeRow = {
  model: "",
  weight: 0.5,
  density: 0.5,
  prompts: "",
};

export const DEFAULT_MERGE_SETTINGS: MergeSettings = {
  method: "ties",
  rows: [EMPTY_MERGE_ROW, EMPTY_MERGE_ROW],
  baseModel: "",
  t: 0.5,
  dtype: "bfloat16",
  tokenizerSource: "base",
  gateMode: "hidden",
  expertsPerToken: 2,
  architecture: "auto",
  outputName: "merged-model",
  cuda: false,
};

/** SLERP takes exactly two rows and hides Add/Remove, so pad or trim to two when switching to it. */
export function rowsForMethod(
  method: MergeMethod,
  rows: MergeRow[],
): MergeRow[] {
  if (method !== "slerp") return rows;
  const two = rows.slice(0, 2);
  while (two.length < 2) two.push(EMPTY_MERGE_ROW);
  return two;
}

export function mergeMethodInfo(method: MergeMethod): MergeMethodInfo {
  return MERGE_METHODS.find((m) => m.value === method) ?? MERGE_METHODS[0];
}

/** One prompt per line; a single line may list them comma separated instead. */
export function parsePrompts(text: string): string[] {
  const parts = text.includes("\n") ? text.split("\n") : text.split(",");
  return parts.map((p) => p.trim()).filter(Boolean);
}

function moeBlockReason(s: MergeSettings, names: string[]): string | null {
  if (names.length < 2)
    return "A mixture of experts needs at least two experts.";
  if (!s.baseModel.trim()) return "Mixture of experts needs a base model.";
  if (
    !Number.isInteger(s.expertsPerToken) ||
    s.expertsPerToken < 1 ||
    s.expertsPerToken > names.length
  ) {
    return `Experts per token must be a whole number from 1 to ${names.length}.`;
  }
  // MergeKit skips its prompt checks for random gates only.
  if (s.gateMode === "random") return null;
  const prompts = s.rows.map((r) => parsePrompts(r.prompts).join("\n"));
  if (prompts.some((p) => !p)) return "Give every expert at least one prompt.";
  if (new Set(prompts).size === 1) {
    return "Give each expert different prompts, or the router cannot tell them apart.";
  }
  return null;
}

/** Why the merge cannot start yet, mirroring the backend's request validation; null when it can. */
export function mergeBlockReason(s: MergeSettings): string | null {
  const info = mergeMethodInfo(s.method);
  const names = s.rows.map((r) => r.model.trim());
  if (names.some((n) => !n)) return "Fill in every model.";
  if (new Set(names).size !== names.length) {
    return "Each model may appear only once.";
  }
  if (s.method === "moe") {
    const reason = moeBlockReason(s, names);
    if (reason) return reason;
  }
  if (s.method === "slerp" && names.length !== 2) {
    return "SLERP merges exactly two models.";
  }
  if (s.method === "linear" && names.length < 2) {
    return "Add at least two models.";
  }
  // A MoE base may also be one of its experts.
  if (info.base && s.method !== "moe") {
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
  if (
    s.method !== "slerp" &&
    s.method !== "moe" &&
    s.rows.some((r) => !Number.isFinite(r.weight))
  ) {
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
  if (s.method === "moe") return moeRequest(s);
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
    // Linear has no base, so its vocabularies must match unless the union is asked for.
    tokenizer_source:
      s.method === "linear" && s.tokenizerSource === "base"
        ? null
        : s.tokenizerSource,
    output_name: s.outputName.trim(),
    cuda: s.cuda,
  };
}

function moeRequest(
  s: MergeSettings,
): Omit<MergeModelsParams, "hf_token" | "install_missing_dependencies"> {
  return {
    method: "moe",
    models: s.rows.map((r) => {
      const prompts = parsePrompts(r.prompts);
      return {
        model: r.model.trim(),
        ...(prompts.length ? { positive_prompts: prompts } : {}),
      };
    }),
    base_model: s.baseModel.trim(),
    t: null,
    dtype: s.dtype,
    tokenizer_source: null,
    gate_mode: s.gateMode,
    experts_per_token: s.expertsPerToken,
    architecture: s.architecture === "auto" ? null : s.architecture,
    output_name: s.outputName.trim(),
    cuda: s.cuda,
  };
}
