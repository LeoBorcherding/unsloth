// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The GitHub seed reads the first 30 comments and can't run without a token
// (its GraphQL API refuses anonymous requests), so the form must say both.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { buildSeedConfig } = await import(
  "../src/features/recipe-studio/utils/payload/builders-seed.ts"
);
const { makeSeedConfig } = await import(
  "../src/features/recipe-studio/utils/config-factories.ts"
);

type SeedConfig = Parameters<typeof buildSeedConfig>[0];

function githubSeed(maxComments: string): SeedConfig {
  return {
    ...makeSeedConfig("seed-1", [], "github_repo"),
    github_repo_slug: "octo/repo",
    github_max_comments_per_item: maxComments,
  } as SeedConfig;
}

test("max comments per item stops at the 30 the scraper reads", () => {
  for (const [value, ok] of [["30", true], ["31", false], ["200", false]] as const) {
    const errors: string[] = [];
    buildSeedConfig(githubSeed(value), errors);
    assert.equal(errors.length === 0, ok, `${value}: ${errors.join(" ")}`);
  }
});

test("the token field is not labelled optional when the server has no token", () => {
  const dialog = readFileSync(
    new URL(
      "../src/features/recipe-studio/dialogs/seed/seed-dialog.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.doesNotMatch(dialog, /label="GitHub token \(optional\)"/);
  assert.match(dialog, /"GitHub token \(required\)"/);
  assert.match(dialog, /max=\{30\}/);
});
