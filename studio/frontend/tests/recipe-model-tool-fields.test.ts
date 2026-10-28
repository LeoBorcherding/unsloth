// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Local provider promised Chat's model and Run refused without a pick; a blank tool
// timeout meant 15 s for Load tools and no limit in the run; a model timeout of 0.5 went out as 0.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";
import { loadWithStubs, stubJsxRuntime } from "./helpers/module-stubs.ts";

registerBundlerResolver();

const { importRecipePayload } = await import(
  "../src/features/recipe-studio/utils/import/importer.ts"
);
const { buildRecipePayload } = await import(
  "../src/features/recipe-studio/utils/payload/build-payload.ts"
);
const buildersLlm = await import(
  "../src/features/recipe-studio/utils/payload/builders-llm.ts"
);
const { buildModelConfig } = await import(
  "../src/features/recipe-studio/utils/payload/builders-model.ts"
);

const LOADED = "unsloth/gemma-4-E2B-it-GGUF";

function localRecipe(model: string) {
  return {
    recipe: {
      model_providers: [{ name: "local", endpoint: "", is_local: true }],
      model_configs: [{ alias: "writer", model, provider: "local" }],
      columns: [
        {
          column_type: "llm-text",
          name: "answer",
          model_alias: "writer",
          prompt: "Answer a question.",
        },
      ],
    },
  };
}

function build(recipe: unknown) {
  const imported = importRecipePayload(JSON.stringify(recipe));
  assert.deepEqual(imported.errors, []);
  assert.ok(imported.snapshot);
  const s = imported.snapshot;
  return buildRecipePayload(
    s.configs,
    s.nodes,
    s.edges,
    s.processors,
    s.layoutDirection,
    s.auxNodePositions,
  );
}

function loadHook(status: Record<string, unknown> | Error) {
  return loadWithStubs<{
    fillBlankLocalModels: (payload: unknown) => Promise<{
      payload: { recipe: { model_configs: Record<string, unknown>[] } };
      error: string | null;
    }>;
  }>(
    new URL(
      "../src/features/recipe-studio/hooks/use-recipe-executions.ts",
      import.meta.url,
    ),
    {
      "@/features/chat": {
        getInferenceStatus: () =>
          status instanceof Error ? Promise.reject(status) : Promise.resolve(status),
      },
      "@/features/hub/lib/model-identity": {},
      "@/features/chat/presets/preset-policy": {},
      "@/config/env": {},
      "@/features/model-picker": {},
      "@/lib/toast": {},
      "@/shared/toast": {},
      react: {},
      "zustand/react/shallow": {},
      "../api": {},
      "../data/executions-db": {},
      "../executions/execution-helpers": {},
      "../executions/hydration": {},
      "../executions/run-settings": {},
      "../executions/runtime": {},
      "../stores/recipe-executions": {},
      "../executions/tracker": {},
    },
  );
}

test("a Local model block left empty builds, and runs on the model Chat has loaded", async () => {
  const built = build(localRecipe(""));
  assert.deepEqual(built.errors, []);

  const { fillBlankLocalModels } = loadHook({
    // biome-ignore lint/style/useNamingConvention: api schema
    model_identifier: LOADED,
    // biome-ignore lint/style/useNamingConvention: api schema
    active_model: LOADED,
    // biome-ignore lint/style/useNamingConvention: api schema
    gguf_variant: "Q4_K_M",
  });
  const filled = await fillBlankLocalModels(built.payload);
  assert.equal(filled.error, null);
  const [config] = filled.payload.recipe.model_configs;
  assert.equal(config.model, LOADED);
  assert.equal(config.gguf_variant, "Q4_K_M");
  // The editor's own payload is left as it was.
  assert.equal(built.payload.recipe.model_configs[0].model, "");
});

// /status withholds a native-path GGUF's on-disk path, and the job's loaded-model check
// compares against that path, so the display label would be refused as a mismatch.
test("an empty Local model block runs on a GGUF Chat opened from a native file path", async () => {
  const { fillBlankLocalModels } = loadHook({
    // biome-ignore lint/style/useNamingConvention: api schema
    model_identifier: null,
    // biome-ignore lint/style/useNamingConvention: api schema
    active_model: "my-model-Q4_K_M.gguf",
    // biome-ignore lint/style/useNamingConvention: api schema
    is_gguf: true,
    // biome-ignore lint/style/useNamingConvention: api schema
    gguf_variant: null,
  });
  const filled = await fillBlankLocalModels(build(localRecipe("")).payload);
  assert.equal(filled.error, null);
  const [config] = filled.payload.recipe.model_configs;
  assert.equal(config.model, "local");
  assert.equal(config.gguf_variant, undefined);
});

test("an empty Local model block with nothing loaded in Chat says so", async () => {
  // biome-ignore lint/style/useNamingConvention: api schema
  const { fillBlankLocalModels } = loadHook({ active_model: null });
  const filled = await fillBlankLocalModels(build(localRecipe("")).payload);
  assert.match(String(filled.error), /writer: no model is loaded in Chat/);

  const failing = loadHook(new Error("offline"));
  const offline = await failing.fillBlankLocalModels(build(localRecipe("")).payload);
  assert.match(String(offline.error), /no model is loaded in Chat/);
});

test("a picked local model is never replaced by Chat's", async () => {
  const { fillBlankLocalModels } = loadHook({
    // biome-ignore lint/style/useNamingConvention: api schema
    model_identifier: LOADED,
  });
  const built = build(localRecipe("unsloth/Qwen3-14B-GGUF"));
  const filled = await fillBlankLocalModels(built.payload);
  assert.equal(filled.payload, built.payload);
});

test("the provider dialog copy matches what Run does", () => {
  const copy = readSrc(
    "features/recipe-studio/dialogs/models/model-provider-dialog.tsx",
  ).replace(/\s+/g, " ");
  assert.doesNotMatch(copy, /use whatever model is loaded in the Chat tab when you hit run/);
  assert.match(copy, /leave it empty to use the model loaded in the Chat tab/);
});

test("the model Chat has loaded is listed first in the local model picker", () => {
  const { loadedModelFirst } = loadWithStubs<{
    loadedModelFirst: (
      models: { id: string; model_id?: string; path: string }[],
      loadedId: string | null,
    ) => { id: string }[];
  }>(
    new URL(
      "../src/features/recipe-studio/dialogs/models/local-recipe-model-selector.tsx",
      import.meta.url,
    ),
    {
      "react/jsx-runtime": stubJsxRuntime(),
      react: { forwardRef: (fn: unknown) => fn },
      "@/components/ui/badge": {},
      "@/components/ui/button": {},
      "@/components/ui/input": {},
      "@/components/ui/popover": {},
      "@/components/ui/spinner": {},
      "@/features/chat": {},
      "@/lib/utils": {},
      "@/lib/refresh-icon": {},
      "@/lib/chevron-icons": {},
      "@hugeicons/react": {},
      "@tanstack/react-router": {},
      "lucide-react": {},
      "@/features/hub": {},
    },
  );
  const models = [
    { id: "ibm-granite/granite-4.1-30b", path: "/m/granite" },
    { id: "Qwen/Qwen3-14B", path: "/m/qwen" },
    { id: LOADED, path: "/m/gemma" },
  ];
  assert.deepEqual(
    loadedModelFirst(models, LOADED).map((model) => model.id),
    [LOADED, "ibm-granite/granite-4.1-30b", "Qwen/Qwen3-14B"],
  );
  assert.equal(loadedModelFirst(models, null), models);
});

test("a blank tool timeout is 30 s for Load tools and for the run", () => {
  const errors: string[] = [];
  const toolConfig = buildersLlm.buildLlmToolConfig(
    {
      id: "tools",
      // biome-ignore lint/style/useNamingConvention: api schema
      tool_alias: "docs-tools",
      providers: ["docs"],
      // biome-ignore lint/style/useNamingConvention: api schema
      timeout_sec: "",
    },
    errors,
  );
  assert.deepEqual(errors, []);
  assert.equal(toolConfig?.timeout_sec, 30);
  assert.equal(buildersLlm.resolveToolTimeoutSec("12"), 12);

  const dialog = readSrc(
    "features/recipe-studio/dialogs/tool-profile/tool-profile-dialog.tsx",
  );
  const loadTools = dialog.slice(
    dialog.indexOf("async function loadTools()"),
    dialog.indexOf("listMcpTools({"),
  );
  assert.match(loadTools, /resolveToolTimeoutSec\(config\.timeout_sec\)/);
});

test("a model timeout that isn't a whole number of seconds is an error, not a truncated 0", () => {
  const model = (timeout: string) => {
    const errors: string[] = [];
    const built = buildModelConfig(
      {
        id: "m",
        kind: "model_config",
        name: "writer",
        model: "gpt-4o-mini",
        provider: "openai",
        // biome-ignore lint/style/useNamingConvention: api schema
        inference_timeout: timeout,
      },
      errors,
    );
    const params = built.inference_parameters as Record<string, unknown> | undefined;
    return { errors, timeout: params?.timeout };
  };
  for (const bad of ["0.5", "-5", "0", "abc"]) {
    const result = model(bad);
    assert.equal(result.timeout, undefined, `timeout ${bad}`);
    assert.match(result.errors.join("\n"), /writer: Timeout must be a whole number/);
  }
  assert.deepEqual(model("30"), { errors: [], timeout: 30 });
  assert.deepEqual(model(""), { errors: [], timeout: undefined });
});
