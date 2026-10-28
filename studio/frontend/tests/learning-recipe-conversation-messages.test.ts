// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Multi-Turn Chat learning recipe shipped unregistered, and its chat sat in a dict
// ({conversation: [...]}) under `output_format`, which training mapped to one assistant
// turn. It is listed again and writes a flat `messages` list training reads as chat turns.

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { LEARNING_RECIPES } = await import(
  "../src/features/data-recipes/learning-recipes/index.ts"
);

const RECIPES_DIR = new URL(
  "../src/features/data-recipes/learning-recipes/",
  import.meta.url,
);
const PAGE = readFileSync(
  new URL(
    "../src/features/data-recipes/pages/data-recipes-page.tsx",
    import.meta.url,
  ),
  "utf8",
);

const registeredIds = new Set(
  LEARNING_RECIPES.map((recipe: { id: string }) => recipe.id),
);
const cardIds = new Set(
  [...PAGE.matchAll(/learningRecipeId: "([^"]+)"/g)].map((match) => match[1]),
);

test("every learning recipe file is registered and has a card", () => {
  const files = readdirSync(RECIPES_DIR)
    .filter((name) => name.endsWith(".json"))
    .map((name) => name.replace(/\.json$/, ""));
  for (const id of files) {
    assert.ok(registeredIds.has(id), `${id}.json is not in LEARNING_RECIPES`);
    assert.ok(cardIds.has(id), `${id} has no template card`);
  }
});

test("Multi-Turn Chat writes a flat messages list of role/content turns", () => {
  const payload = JSON.parse(
    readFileSync(new URL("conversation.json", RECIPES_DIR), "utf8"),
  );
  const structured = payload.recipe.columns.filter(
    (column: { column_type: string }) =>
      column.column_type === "llm-structured",
  );
  assert.equal(structured.length, 1);
  const [chat] = structured;
  assert.equal(chat.name, "messages");
  assert.equal(chat.drop, false);
  assert.equal(chat.output_format.type, "array");
  assert.deepEqual(chat.output_format.items.required, ["role", "content"]);
  assert.ok(
    payload.ui.nodes.some((node: { id: string }) => node.id === "messages"),
  );
});

test("Multi-Turn Chat runs on the local model, not a hardcoded endpoint", () => {
  const payload = JSON.parse(
    readFileSync(new URL("conversation.json", RECIPES_DIR), "utf8"),
  );
  for (const provider of payload.recipe.model_providers) {
    assert.equal(provider.is_local, true);
    assert.equal(provider.endpoint, "");
  }
});
