// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  createRunSeed,
  findSameConfigVersion,
  formatVersionLabel,
  nextRunVersion,
  withExecutionDefaults,
} = await import("../src/features/recipe-studio/executions/execution-helpers.ts");
const { createBaseExecutionRecord } = await import(
  "../src/features/recipe-studio/executions/runtime.ts"
);
const { buildExecutionPayload } = await import(
  "../src/features/recipe-studio/executions/run-settings.ts"
);
const { summarizeQuality } = await import(
  "../src/features/recipe-studio/components/executions/executions-view-helpers.ts"
);
const { importRecipePayload } = await import(
  "../src/features/recipe-studio/utils/import/importer.ts"
);
const { buildRecipePayload } = await import(
  "../src/features/recipe-studio/utils/payload/build-payload.ts"
);

type Rec = Record<string, unknown>;
type ExecutionRecord = ReturnType<typeof withExecutionDefaults>;

function run(version: number | null, recipe: Rec | null): ExecutionRecord {
  return createBaseExecutionRecord({
    recipeId: "r",
    kind: "full",
    rows: 10,
    currentSignature: "",
    version,
    seed: 7,
    configSnapshot: recipe ? { recipe, run: { rows: 10 } } : null,
  });
}

test("run versions count up per recipe and skip legacy runs", () => {
  assert.equal(nextRunVersion([]), 1);
  assert.equal(nextRunVersion([run(null, null), run(2, null), run(1, null)]), 3);
  assert.equal(formatVersionLabel(run(4, null)), "v4");
  assert.equal(formatVersionLabel(run(null, null)), null);
});

test("a run points at the earliest older version with the same recipe", () => {
  const a = { columns: [{ name: "q" }] };
  const b = { columns: [{ name: "q2" }] };
  const runs = [run(1, a), run(2, b), run(3, a), run(4, a)];
  assert.equal(findSameConfigVersion(runs[3], runs), 1);
  assert.equal(findSameConfigVersion(runs[1], runs), null);
  assert.equal(findSameConfigVersion(runs[0], runs), null);
  assert.equal(findSameConfigVersion(run(5, null), runs), null);
});

test("saved runs from before versioning load with empty version fields", () => {
  const legacy = withExecutionDefaults({ id: "old", dataset: [] } as unknown as ExecutionRecord);
  assert.equal(legacy.version, null);
  assert.equal(legacy.seed, null);
  assert.equal(legacy.quality, null);
  assert.equal(legacy.config_snapshot, null);
  const junk = withExecutionDefaults({
    id: "junk",
    version: 1.5,
    seed: -3,
    quality: "bad",
  } as unknown as ExecutionRecord);
  assert.equal(junk.version, null);
  assert.equal(junk.seed, null);
  assert.equal(junk.quality, null);
});

test("the run seed is an unsigned 32-bit integer sent with the job", () => {
  for (let i = 0; i < 50; i += 1) {
    const seed = createRunSeed();
    assert.ok(Number.isInteger(seed) && seed >= 0 && seed < 2 ** 32, String(seed));
  }
  const payload = buildExecutionPayload({
    payload: {
      recipe: {
        // biome-ignore lint/style/useNamingConvention: api schema
        model_providers: [],
        // biome-ignore lint/style/useNamingConvention: api schema
        mcp_providers: [],
        // biome-ignore lint/style/useNamingConvention: api schema
        model_configs: [],
        // biome-ignore lint/style/useNamingConvention: api schema
        tool_configs: [],
        columns: [],
        processors: [],
      },
      // biome-ignore lint/style/useNamingConvention: api schema
      run: { rows: 5, preview: true, output_formats: ["jsonl"] },
      ui: { nodes: [], edges: [] },
    },
    kind: "full",
    rows: 20,
    settings: {
      batchSize: 1000,
      batchEnabled: false,
      mergeBatches: false,
      llmParallelRequests: null,
      nonInferenceWorkers: 4,
      maxConversationRestarts: 5,
      maxConversationCorrectionSteps: 0,
      disableEarlyShutdown: true,
      shutdownErrorRate: 0.5,
      shutdownErrorWindow: 10,
    },
    seed: 123,
  });
  assert.equal(payload.run.seed, 123);
  assert.equal(payload.run.rows, 20);
});

test("quality summary adds up empty values and format failures across columns", () => {
  const summary = summarizeQuality({
    // biome-ignore lint/style/useNamingConvention: backend schema
    rows_checked: 3,
    // biome-ignore lint/style/useNamingConvention: backend schema
    rows_total: 3,
    truncated: false,
    // biome-ignore lint/style/useNamingConvention: backend schema
    content_columns: ["a", "b"],
    // biome-ignore lint/style/useNamingConvention: backend schema
    exact_duplicates: 0,
    // biome-ignore lint/style/useNamingConvention: backend schema
    near_duplicates: 0,
    // biome-ignore lint/style/useNamingConvention: backend schema
    near_duplicates_checked: true,
    columns: [
      // biome-ignore lint/style/useNamingConvention: backend schema
      { name: "a", type: "llm-text", empty: 1, format_failures: 2, format_check: "leaked_tokens", min_chars: 1, median_chars: 2, max_chars: 3 },
      // biome-ignore lint/style/useNamingConvention: backend schema
      { name: "b", type: "llm-structured", empty: 0, format_failures: 1, format_check: "json", min_chars: 1, median_chars: 2, max_chars: 3 },
    ],
    validators: [],
    warnings: [],
    verdict: "ok",
  });
  assert.deepEqual(summary, { formatFailures: 3, emptyValues: 1 });
});

const NEW_TEMPLATES = ["tool-calls.json", "reasoning-traces.json"];
const DIR = new URL("../src/features/data-recipes/learning-recipes/", import.meta.url);

function loadTemplate(name: string): { text: string; recipe: Rec } {
  const text = readFileSync(new URL(name, DIR), "utf8");
  return { text, recipe: (JSON.parse(text) as { recipe: Rec }).recipe };
}

const list = (value: unknown): Rec[] => (Array.isArray(value) ? (value as Rec[]) : []);

test("the new templates run on the local model and pass the editor's Check", () => {
  for (const name of NEW_TEMPLATES) {
    const { text, recipe } = loadTemplate(name);
    const providers = list(recipe.model_providers);
    assert.ok(providers.length > 0 && providers.every((p) => p.is_local === true), name);
    for (const config of list(recipe.model_configs)) {
      assert.equal(config.model, "unsloth/gemma-4-E2B-it-GGUF", name);
    }

    const { errors, snapshot } = importRecipePayload(text);
    assert.ok(snapshot, `${name}: ${errors.join("; ")}`);
    const built = buildRecipePayload(
      snapshot.configs,
      snapshot.nodes,
      snapshot.edges,
      snapshot.processors,
      snapshot.layoutDirection,
      snapshot.auxNodePositions,
    );
    assert.deepEqual(built.errors, [], name);
    const names = list(built.payload.recipe.columns).map((c) => c.name);
    assert.deepEqual(
      names.sort(),
      list(recipe.columns)
        .map((c) => c.name)
        .sort(),
      name,
    );
  }
});

test("the tool-call template checks every call with the JSON validator", () => {
  const { recipe } = loadTemplate("tool-calls.json");
  const columns = list(recipe.columns);
  const validator = columns.find((c) => c.column_type === "validation");
  assert.deepEqual(validator?.target_columns, ["arguments"]);
  assert.equal(
    (validator?.validator_params as Rec | undefined)?.validation_function,
    "unsloth_json_validator",
  );
  // The schema list is a constant expression, so it must not open a Jinja tag.
  const tools = String(columns.find((c) => c.name === "tools")?.expr ?? "");
  assert.doesNotMatch(tools, /\{\{|\{%|\{#/);
  assert.equal((JSON.parse(tools) as unknown[]).length, 4);
});

test("the reasoning template ships a <think> trace built from structured output", () => {
  const { recipe } = loadTemplate("reasoning-traces.json");
  const columns = list(recipe.columns);
  const solution = columns.find((c) => c.name === "solution");
  assert.equal(solution?.column_type, "llm-structured");
  assert.deepEqual(
    ((solution?.output_format as Rec | undefined)?.required as string[] | undefined) ?? [],
    ["reasoning", "final_answer"],
  );
  assert.match(String(columns.find((c) => c.name === "assistant")?.expr), /^<think>\n\{\{ solution\.reasoning \}\}/);
});

test("both templates are in the gallery", () => {
  const index = readSrc("features/data-recipes/learning-recipes/index.ts");
  const page = readSrc("features/data-recipes/pages/data-recipes-page.tsx");
  for (const id of ["tool-calls", "reasoning-traces"]) {
    assert.match(index, new RegExp(`id: "${id}"`));
    assert.match(page, new RegExp(`learningRecipeId: "${id}"`));
  }
});

// Card heights as rendered in the editor (measured at 1440x900); notes are 400 wide like the rest.
const CARD_HEIGHT: Record<string, number> = {
  note: 290,
  provider: 84,
  model: 184,
  sampler: 81,
  expression: 189,
  "llm-text": 212,
  "llm-structured": 80,
  validation: 80,
};

test("the new templates lay out without overlaps and fit the editor at its minimum zoom", () => {
  for (const name of NEW_TEMPLATES) {
    const parsed = JSON.parse(loadTemplate(name).text) as {
      recipe: Rec;
      ui: { nodes: { id: string; x: number; y: number; width: number; node_type?: string }[] };
    };
    const kind = new Map<string, string>();
    for (const p of list(parsed.recipe.model_providers)) kind.set(String(p.name), "provider");
    for (const m of list(parsed.recipe.model_configs)) kind.set(String(m.alias), "model");
    for (const c of list(parsed.recipe.columns)) kind.set(String(c.name), String(c.column_type));
    const boxes = parsed.ui.nodes.map((n) => {
      const k = n.node_type === "markdown_note" ? "note" : kind.get(n.id);
      assert.ok(k && CARD_HEIGHT[k], `${name}: ${n.id}`);
      return { id: n.id, note: k === "note", x0: n.x, y0: n.y, x1: n.x + n.width, y1: n.y + CARD_HEIGHT[k] };
    });
    for (const [i, a] of boxes.entries()) {
      for (const b of boxes.slice(i + 1)) {
        const apart = a.x1 + 40 <= b.x0 || b.x1 + 40 <= a.x0 || a.y1 + 30 <= b.y0 || b.y1 + 30 <= a.y0;
        assert.ok(apart, `${name}: ${a.id} crowds ${b.id}`);
      }
    }
    // Fit ignores notes, so a note outside the cards' box gets cut off on load.
    const cards = boxes.filter((b) => !b.note);
    const minX = Math.min(...cards.map((b) => b.x0));
    const minY = Math.min(...cards.map((b) => b.y0));
    for (const note of boxes.filter((b) => b.note)) {
      assert.ok(note.x0 >= minX && note.y0 >= minY, `${name}: note sits outside the fitted area`);
    }
    // At the 0.5 zoom floor a 1440px window with the app sidebar open shows ~1900 units.
    assert.ok(Math.max(...cards.map((b) => b.x1)) - minX <= 1900, `${name} is too wide`);
  }
});
