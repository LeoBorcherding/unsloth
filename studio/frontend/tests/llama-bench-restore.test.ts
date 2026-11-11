// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  ownsRestore,
  shouldPutBack,
} from "../src/features/benchmarks/llama-bench/llama-bench-restore.ts";

const chat = { model: "unsloth/Qwen3-4B-GGUF", variant: "Q4_K_M" };
const picked = { model: "unsloth/gemma-3-1b-it-GGUF", variant: "Q8_0" };
const none = { model: null, variant: null };

test("a persisted restore only resumes for the run it was saved with", () => {
  const run = (jobId: string, clientRef: string | null = null) => ({ jobId, clientRef });
  assert.equal(ownsRestore({ jobId: "llama-bench-a", clientRef: "t1" }, run("llama-bench-a")), true);
  // A later run, or none after a restart: the old model must not load over the user's.
  assert.equal(
    ownsRestore({ jobId: "llama-bench-a", clientRef: "t1" }, run("llama-bench-b", "t1")),
    false,
  );
  assert.equal(ownsRestore({ jobId: "llama-bench-a" }, undefined), false);
  assert.equal(ownsRestore(null, run("llama-bench-a")), false);
  // Saved before the server answered, or the answer was lost: the start's tag finds its run.
  assert.equal(ownsRestore({ jobId: null, clientRef: "t1" }, run("llama-bench-a", "t1")), true);
  // Not another tab's run, and not a run with no tag.
  assert.equal(ownsRestore({ jobId: null, clientRef: "t1" }, run("llama-bench-a", "t2")), false);
  assert.equal(ownsRestore({ jobId: null, clientRef: "t1" }, run("llama-bench-a")), false);
  assert.equal(ownsRestore({ jobId: null }, run("llama-bench-a")), false);
});

test("chat's model goes back only over an empty slot or the benched model", () => {
  assert.equal(shouldPutBack(none, chat, picked), true);
  assert.equal(shouldPutBack(picked, chat, picked), true);
  // Another tab already put it back, or the run was refused before the unload.
  assert.equal(shouldPutBack(chat, chat, chat), false);
  assert.equal(shouldPutBack(chat, chat, picked), false);
  // The user loaded something else since.
  assert.equal(
    shouldPutBack({ model: "other", variant: null }, chat, picked),
    false,
  );
});
