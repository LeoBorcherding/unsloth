// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { importRecipePayload } = await import(
  "../src/features/recipe-studio/utils/import/importer.ts"
);
const { buildRecipePayload } = await import(
  "../src/features/recipe-studio/utils/payload/build-payload.ts"
);

// The saved source keeps only the resolved path, so a reopened recipe needs the picked subset and
// split from the UI block, or Load would resolve the default config over the right path.
test("a saved recipe reopens with the subset and split it was loaded with", () => {
  const recipe = {
    recipe: {
      seed_config: {
        source: { seed_type: "hf", path: "datasets/openai/gsm8k/socratic/test-*.parquet" },
      },
      columns: [{ column_type: "sampler", name: "id", sampler_type: "uuid", params: {} }],
    },
    ui: { seed_source_type: "hf" },
  };
  const first = importRecipePayload(JSON.stringify(recipe));
  assert.ok(first.snapshot, JSON.stringify(first.errors));
  const seed = Object.values(first.snapshot.configs).find((config) => config.kind === "seed");
  assert.ok(seed?.kind === "seed");
  seed.hf_subset = "socratic";
  seed.hf_split = "test";

  const { configs, nodes, edges, processors, layoutDirection, auxNodePositions } = first.snapshot;
  const built = buildRecipePayload(configs, nodes, edges, processors, layoutDirection, auxNodePositions);
  const reopened = importRecipePayload(JSON.stringify(built.payload));
  assert.ok(reopened.snapshot, JSON.stringify(reopened.errors));
  const again = Object.values(reopened.snapshot.configs).find((config) => config.kind === "seed");
  assert.ok(again?.kind === "seed");
  assert.equal(again.hf_subset, "socratic");
  assert.equal(again.hf_split, "test");
});
