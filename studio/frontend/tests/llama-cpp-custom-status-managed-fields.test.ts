// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

installLocalStorageFake();
register("./store-settings-resolver.mjs", import.meta.url);

const store = await import("../src/features/chat/stores/chat-runtime-store.ts");
const { applyActiveModelStatusToStore } = await import(
  "../src/features/chat/lib/apply-inference-status-to-store.ts"
);
const { useChatRuntimeStore } = store;

const MODEL = "unsloth/Qwen3-1.7B-GGUF";
const custom = {
  version: 1,
  mode: "custom",
  ini: "[*]\n",
  section: null,
} as const;
// What a custom load echoes: the INI's tuning, which must not become the managed choice.
const echo = {
  is_gguf: true,
  gpu_memory_mode: "manual",
  gpu_layers: 0,
  n_layers: 28,
  cache_type_kv: "f16",
  speculative_type: "none",
  tensor_parallel: true,
} as const;

test("a custom load's tuning stays out of the managed controls", () => {
  const resp = { ...echo, requested_llama_cpp_config: custom };
  assert.deepEqual(store.managedGpuMemoryFields(resp), {
    ggufLayerCount: 28,
    moeLayerCount: null,
  });
  assert.deepEqual(store.managedKvCacheFields(resp), {});
  assert.deepEqual(store.managedSpeculativeSettings(resp), {});
  assert.deepEqual(store.managedTensorParallelFields(resp), {});
  assert.equal(store.managedTensorParallelFields(echo).tensorParallel, true);

  useChatRuntimeStore.setState({
    modelLoading: false,
    gpuMemoryMode: "auto",
    gpuLayers: -1,
    customContextLength: null,
    kvCacheDtype: null,
    speculativeType: "auto",
    tensorParallel: false,
    params: { ...useChatRuntimeStore.getState().params, checkpoint: MODEL },
  });
  applyActiveModelStatusToStore(
    {
      ...echo,
      active_model: MODEL,
      model_identifier: MODEL,
      requested_context_length: 3072,
      requested_llama_cpp_config: custom,
    } as never,
    { previousCheckpoint: MODEL },
  );
  const s = useChatRuntimeStore.getState();
  assert.equal(s.llamaCppConfig?.mode, "custom");
  assert.equal(s.gpuMemoryMode, "auto");
  assert.equal(s.gpuLayers, -1);
  assert.equal(s.customContextLength, null);
  assert.equal(s.kvCacheDtype, null);
  assert.equal(s.speculativeType, "auto");
  assert.equal(s.tensorParallel, false);
});
