// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

// The gallery and tour promise these run as shipped, so a template that can't pass Check is a bug.

const DIR = new URL("../src/features/data-recipes/learning-recipes/", import.meta.url);
const SRC = new URL("../src/", import.meta.url);

type Rec = Record<string, unknown>;

const templates = readdirSync(DIR)
  .filter((name) => name.endsWith(".json") && name !== "conversation.json")
  .map((name) => ({
    name,
    recipe: (JSON.parse(readFileSync(new URL(name, DIR), "utf8")) as { recipe: Rec }).recipe,
  }));

const list = (value: unknown): Rec[] => (Array.isArray(value) ? (value as Rec[]) : []);
const text = (value: unknown): string => (typeof value === "string" ? value.trim() : "");

test("every learning template names a model and a provider that can reach it", () => {
  assert.equal(templates.length, 8);
  for (const { name, recipe } of templates) {
    const providers = new Map(list(recipe.model_providers).map((p) => [text(p.name), p]));
    for (const config of list(recipe.model_configs)) {
      assert.ok(text(config.model), `${name}: model config ${text(config.alias)} has no model`);
      const provider = providers.get(text(config.provider));
      assert.ok(provider, `${name}: provider ${text(config.provider)} not found`);
      assert.ok(
        provider.is_local === true || text(provider.endpoint),
        `${name}: provider ${text(provider.name)} is neither the local model nor an endpoint`,
      );
    }
  }
});

test("every model-only template runs on the local model", () => {
  const remote = templates
    .filter(({ recipe }) => list(recipe.model_providers).some((p) => p.is_local !== true))
    .map(({ name }) => name);
  assert.deepEqual(remote, ["ocr-document-extraction.json"]);
});

test("HF seeds ship the resolved dataset path, not the bare repo id", () => {
  for (const { name, recipe } of templates) {
    const source = (recipe.seed_config as Rec | undefined)?.source as Rec | undefined;
    if (source?.seed_type !== "hf") continue;
    assert.match(text(source.path), /^datasets\/[^/]+\/[^/]+\/.*\*/, `${name}: ${text(source.path)}`);
  }
});

test("sla_target renders without a trailing newline", () => {
  const recipe = templates.find(({ name }) => name === "structured-outputs-jinja.json")?.recipe;
  const column = list(recipe?.columns).find((c) => c.name === "sla_target");
  const outside = text(column?.expr).replace(/\{%.*?%\}/g, "|");
  assert.equal(outside.includes("\n"), false, JSON.stringify(column?.expr));
});

test("the MNIST template asks for a digit, not a document transcription", () => {
  const recipe = templates.find(({ name }) => name === "ocr-document-extraction.json")?.recipe;
  const prompt = text(list(recipe?.columns).find((c) => c.name === "ocr_text")?.prompt);
  assert.match(prompt, /digit/i);
  assert.doesNotMatch(prompt, /document/i);
});

test("tour copy sends PDFs to Document file and shows Esc as a key", () => {
  const editorSteps = readFileSync(new URL("features/recipe-studio/tour/steps.tsx", SRC), "utf8");
  assert.doesNotMatch(editorSteps, /Import\s+is where a PDF/);
  assert.match(editorSteps, /Document file/);
  const tour = readFileSync(new URL("features/tour/components/guided-tour.tsx", SRC), "utf8");
  assert.doesNotMatch(tour, /`Esc`/);
});

test("templates pass the editor's Check as shipped, short of a PDF upload", async () => {
  const { importRecipePayload } = await import("../src/features/recipe-studio/utils/import/importer.ts");
  const { buildRecipePayload } = await import("../src/features/recipe-studio/utils/payload/build-payload.ts");
  for (const name of readdirSync(DIR).filter((n) => n.endsWith(".json") && n !== "conversation.json")) {
    const { errors, snapshot } = importRecipePayload(readFileSync(new URL(name, DIR), "utf8"));
    assert.ok(snapshot, `${name}: ${errors.join("; ")}`);
    const built = buildRecipePayload(
      snapshot.configs,
      snapshot.nodes,
      snapshot.edges,
      snapshot.processors,
      snapshot.layoutDirection,
      snapshot.auxNodePositions,
    );
    // Document seeds need an upload before Check can pass.
    const expected = ["pdf-grounded-qa.json", "books-and-documents.json"].includes(name)
      ? ["seed: Load the source-data preview first."]
      : [];
    assert.deepEqual(built.errors, expected, name);
  }
});
