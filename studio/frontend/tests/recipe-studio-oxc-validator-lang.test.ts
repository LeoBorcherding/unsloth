// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { makeLlmConfig, makeValidatorConfig } = await import(
  "../src/features/recipe-studio/utils/config-factories.ts"
);
const { buildValidatorColumn } = await import(
  "../src/features/recipe-studio/utils/payload/builders-validator.ts"
);
const { getBlocksForKind } = await import(
  "../src/features/recipe-studio/blocks/definitions.ts"
);

function oxcMarker(targetLang: string): string {
  const target = {
    ...makeLlmConfig("src", "code", []),
    // biome-ignore lint/style/useNamingConvention: api schema
    code_lang: targetLang,
  };
  const validator = {
    ...makeValidatorConfig("v1", "oxc", "javascript", [target]),
    // biome-ignore lint/style/useNamingConvention: api schema
    target_columns: [target.name],
  };
  const column = buildValidatorColumn(validator, [], new Map([[target.name, target]]));
  return (column.validator_params as { validation_function: string }).validation_function;
}

test("a JS/TS check takes the target's language when OXC can parse it", () => {
  assert.match(oxcMarker("typescript"), /^unsloth_oxc_validator:typescript:/);
});

test("a JS/TS check never sends a language OXC can't parse", () => {
  // The editor already blocks this pair; an imported or hand-edited recipe used to send python.
  assert.match(oxcMarker("python"), /^unsloth_oxc_validator:javascript:/);
});

test("check blocks don't promise to remove failing rows", () => {
  for (const block of getBlocksForKind("validator")) {
    assert.doesNotMatch(block.description, /filter/i, block.title);
  }
});
