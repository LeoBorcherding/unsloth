// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A scheduled task sends a text prompt, so its Model list must leave out cached image and speech repos.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const FILTERED_LOOP = /isChattableCachedRepo\(repo\)/;

const { isChattableCachedRepo } = await import(
  "../src/features/chat/lib/chattable-cached-repo.ts"
);

// Shapes as /api/hub/cached-gguf returns them on a box with these repos cached.
function repo(repoId: string, task: string | null, canChat: boolean) {
  return {
    // biome-ignore lint/style/useNamingConvention: API schema
    repo_id: repoId,
    task,
    capabilities: {
      // biome-ignore lint/style/useNamingConvention: API schema
      can_chat: canChat,
    },
  };
}

const CACHED = [
  repo("unsloth/Qwen3-1.7B-GGUF", "text-generation", true),
  repo("unsloth/gemma-4-E2B-it-GGUF", "text-generation", true),
  repo("unsloth/Z-Image-GGUF", "text-to-image", true),
  repo("unsloth/FLUX.2-klein-9B-GGUF", "text-to-image", true),
  repo("audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF", "text-to-speech", false),
  repo(
    "audio-cpp/audio.cpp-gguf/Parakeet-TDT-0.6B-v3-GGUF",
    "automatic-speech-recognition",
    false,
  ),
  repo("unslothai/Qwen3-ASR-0.6B-GGUF", "text-generation", false),
];

test("only chat models among cached GGUF repos are task models", () => {
  assert.deepEqual(
    CACHED.filter(isChattableCachedRepo).map((row) => row.repo_id),
    ["unsloth/Qwen3-1.7B-GGUF", "unsloth/gemma-4-E2B-it-GGUF"],
  );
});

test("the task dialog filters cached GGUF repos through the chat predicate", () => {
  const source = readFileSync(
    new URL("../src/features/scheduled-tasks/task-dialog.tsx", import.meta.url),
    "utf8",
  );
  const loop = source.slice(
    source.indexOf("for (const repo of inventory.cachedGguf.rows)"),
  );
  assert.match(loop.slice(0, 200), FILTERED_LOOP);
});
