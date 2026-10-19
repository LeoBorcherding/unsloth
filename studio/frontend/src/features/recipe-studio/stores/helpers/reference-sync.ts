// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  LlmConfig,
  ModelConfig,
  NodeConfig,
  RecipeProcessorConfig,
  SamplerConfig,
} from "../../types";
import { removeRef, replaceRef } from "../../utils/refs";

function updateTemplateFields(
  config: NodeConfig,
  updater: (value: string) => string,
): NodeConfig {
  if (config.kind === "llm") {
    const nextPrompt = updater(config.prompt);
    const nextSystem = updater(config.system_prompt);
    const nextOutput =
      typeof config.output_format === "string"
        ? updater(config.output_format)
        : config.output_format;
    if (
      nextPrompt === config.prompt &&
      nextSystem === config.system_prompt &&
      nextOutput === config.output_format
    ) {
      return config;
    }
    return {
      ...config,
      prompt: nextPrompt,
      // biome-ignore lint/style/useNamingConvention: api schema
      system_prompt: nextSystem,
      // biome-ignore lint/style/useNamingConvention: api schema
      output_format: nextOutput,
    };
  }
  if (config.kind === "expression") {
    const nextExpr = updater(config.expr);
    if (nextExpr === config.expr) {
      return config;
    }
    return { ...config, expr: nextExpr };
  }
  return config;
}

export function applyRenameToConfig(
  config: NodeConfig,
  from: string,
  to: string,
): NodeConfig {
  let next = updateTemplateFields(config, (value) =>
    replaceRef(value, from, to),
  );
  if (
    config.kind === "sampler" &&
    config.sampler_type === "subcategory" &&
    config.subcategory_parent === from
  ) {
    const base = next as SamplerConfig;
    next = {
      ...base,
      // biome-ignore lint/style/useNamingConvention: api schema
      subcategory_parent: to,
    };
  }
  if (
    config.kind === "sampler" &&
    config.sampler_type === "timedelta" &&
    config.reference_column_name === from
  ) {
    const base = next as SamplerConfig;
    next = {
      ...base,
      // biome-ignore lint/style/useNamingConvention: api schema
      reference_column_name: to,
    };
  }
  if (config.kind === "model_config" && config.provider === from) {
    const base = next as ModelConfig;
    next = { ...base, provider: to };
  }
  if (config.kind === "llm" && config.model_alias === from) {
    const base = next as LlmConfig;
    next = { ...base, model_alias: to };
  }
  if (config.kind === "llm" && config.tool_alias === from) {
    const base = next as LlmConfig;
    next = { ...base, tool_alias: to };
  }
  if (config.kind === "validator") {
    const targets = config.target_columns ?? [];
    if (targets.includes(from)) {
      const base = next as typeof config;
      next = {
        ...base,
        // biome-ignore lint/style/useNamingConvention: api schema
        target_columns: targets.map((target) =>
          target === from ? to : target,
        ),
      };
    }
  }
  return next;
}

export function applyRemovalToConfig(
  config: NodeConfig,
  ref: string,
): NodeConfig {
  let next = updateTemplateFields(config, (value) => removeRef(value, ref));
  if (
    config.kind === "sampler" &&
    config.sampler_type === "subcategory" &&
    config.subcategory_parent === ref
  ) {
    const base = next as SamplerConfig;
    next = {
      ...base,
      // biome-ignore lint/style/useNamingConvention: api schema
      subcategory_parent: "",
      // biome-ignore lint/style/useNamingConvention: api schema
      subcategory_mapping: {},
    };
  }
  if (
    config.kind === "sampler" &&
    config.sampler_type === "timedelta" &&
    config.reference_column_name === ref
  ) {
    const base = next as SamplerConfig;
    next = {
      ...base,
      // biome-ignore lint/style/useNamingConvention: api schema
      reference_column_name: "",
    };
  }
  if (config.kind === "model_config" && config.provider === ref) {
    const base = next as ModelConfig;
    next = {
      ...base,
      provider: "",
      model: "",
      // biome-ignore lint/style/useNamingConvention: api schema
      gguf_variant: undefined,
    };
  }
  if (config.kind === "llm" && config.model_alias === ref) {
    const base = next as LlmConfig;
    next = { ...base, model_alias: "" };
  }
  if (config.kind === "llm" && config.tool_alias === ref) {
    const base = next as LlmConfig;
    next = { ...base, tool_alias: "" };
  }
  if (config.kind === "validator") {
    const targets = (config.target_columns ?? []).filter(
      (target) => target !== ref,
    );
    if (targets.length !== (config.target_columns ?? []).length) {
      const base = next as typeof config;
      next = {
        ...base,
        // biome-ignore lint/style/useNamingConvention: api schema
        target_columns: targets,
      };
    }
  }
  return next;
}

function applyConfigTransform(
  configs: Record<string, NodeConfig>,
  transform: (config: NodeConfig) => NodeConfig,
): Record<string, NodeConfig> {
  let next = configs;
  for (const [id, config] of Object.entries(configs)) {
    const updated = transform(config);
    if (updated !== config) {
      if (next === configs) {
        next = { ...configs };
      }
      next[id] = updated;
    }
  }
  return next;
}

export function applyRenameToConfigs(
  configs: Record<string, NodeConfig>,
  from: string,
  to: string,
): Record<string, NodeConfig> {
  if (!from || !to || from === to) {
    return configs;
  }
  return applyConfigTransform(configs, (config) =>
    applyRenameToConfig(config, from, to),
  );
}

export function applyRenameToProcessors(
  processors: RecipeProcessorConfig[],
  from: string,
  to: string,
): RecipeProcessorConfig[] {
  if (!from || !to || from === to) {
    return processors;
  }
  let changed = false;
  const next = processors.map((processor) => {
    const template = replaceRef(processor.template, from, to);
    if (template === processor.template) {
      return processor;
    }
    changed = true;
    return { ...processor, template };
  });
  return changed ? next : processors;
}

/** A name references can follow: not empty, not another block's (notes aren't referenced), and not
 * one another block's references still wait on. */
export function isFreeName(
  configs: Record<string, NodeConfig>,
  id: string,
  name: string,
  renameSources: Record<string, string> = {},
): boolean {
  if (!name.trim()) {
    return false;
  }
  if (Object.entries(renameSources).some(([other, source]) => other !== id && source === name)) {
    return false;
  }
  return !Object.values(configs).some(
    (config) =>
      config.id !== id &&
      config.kind !== "markdown_note" &&
      config.name === name,
  );
}

export type NameEditState = {
  configs: Record<string, NodeConfig>;
  processors: RecipeProcessorConfig[];
  // Block id -> the name its references still use while its field holds an empty or taken name.
  renameSources: Record<string, string>;
};

/** One edit of a block's name field, which fires per keystroke. References only follow the field
 * while it holds a free name, so clearing it or typing through another block's name doesn't wipe
 * or merge them; the next free name picks them up from where they were. */
export function applyNameEdit(
  state: NameEditState,
  id: string,
  oldName: string,
  newName: string,
): NameEditState {
  // Notes aren't referenced; a block may share a note's name, so a note rename must not move its refs.
  if (state.configs[id]?.kind === "markdown_note") {
    return state;
  }
  const source = state.renameSources[id] ?? oldName;
  if (!isFreeName(state.configs, id, newName, state.renameSources)) {
    if (state.renameSources[id] === source) {
      return state;
    }
    return { ...state, renameSources: { ...state.renameSources, [id]: source } };
  }
  const renameSources = { ...state.renameSources };
  delete renameSources[id];
  return {
    configs: applyRenameToConfigs(state.configs, source, newName),
    processors: applyRenameToProcessors(state.processors, source, newName),
    renameSources,
  };
}

/** Pending renames whose block now holds a free name (the other holder was renamed or removed)
 * catch up, and pending renames of removed blocks are dropped. */
export function reconcileRenameSources(state: NameEditState): NameEditState {
  let next = state;
  for (const [id, source] of Object.entries(state.renameSources)) {
    const config = next.configs[id];
    if (!config) {
      const renameSources = { ...next.renameSources };
      delete renameSources[id];
      next = { ...next, renameSources };
    } else if (isFreeName(next.configs, id, config.name, next.renameSources)) {
      next = applyNameEdit(next, id, source, config.name);
    }
  }
  return next;
}

export function applyRemovalToConfigs(
  configs: Record<string, NodeConfig>,
  ref: string,
): Record<string, NodeConfig> {
  if (!ref) {
    return configs;
  }
  return applyConfigTransform(configs, (config) =>
    applyRemovalToConfig(config, ref),
  );
}
