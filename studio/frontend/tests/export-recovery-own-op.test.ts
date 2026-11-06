// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./helpers/export-store-resolver.mjs", import.meta.url);

const { recoverViaStatus, useExportRuntimeStore } = await import(
  "../src/features/export/stores/export-runtime-store.ts"
);
const stub = await import("./helpers/export-api-stub.mjs");

const done = (seq: number, kind: string, op: string | null, path: string) => ({
  is_export_active: false,
  last_op_seq: seq,
  last_op_kind: kind,
  last_op_id: op,
  last_op_status: "success",
  last_op_output_path: path,
});
const active = (kind: string, op: string | null) => ({
  is_export_active: true,
  active_op_kind: kind,
  active_op_id: op,
});

async function recover(statuses: object[], opId?: string) {
  stub.resetStub();
  stub.responses.set("getExportStatus", () =>
    statuses.length > 1 ? statuses.shift() : statuses[0],
  );
  return recoverViaStatus(3, () => true, opId);
}

test("a merge settles on its own record, not an export it queued behind", async () => {
  const result = await recover(
    [
      active("export_gguf", null),
      done(4, "export_gguf", null, "/out/gguf"),
      active("merge", "mine"),
      done(5, "merge", "mine", "/out/merged"),
    ],
    "mine",
  );
  assert.deepEqual(result, { outputPath: "/out/merged" });
});

test("a merge does not take another tab's merge as its own", async () => {
  const result = await recover(
    [
      active("merge", "other-tab"),
      done(4, "merge", "other-tab", "/out/theirs"),
      active("merge", "mine"),
      done(5, "merge", "mine", "/out/mine"),
    ],
    "mine",
  );
  assert.deepEqual(result, { outputPath: "/out/mine" });
});

test("an export recovered without an op id waits out a merge ahead of it", async () => {
  const result = await recover([
    active("merge", "someone"),
    done(4, "merge", "someone", "/out/merged"),
    active("export_gguf", null),
    done(5, "export_gguf", null, "/out/gguf"),
  ]);
  assert.deepEqual(result, { outputPath: "/out/gguf" });
});

test("a running merge is not adopted as an export run, nor settled as one", () => {
  const store = useExportRuntimeStore.getState();
  store.reset();
  store.applyBackendStatus({
    is_export_active: true,
    active_op_kind: "merge",
  } as Parameters<typeof store.applyBackendStatus>[0]);
  assert.equal(useExportRuntimeStore.getState().isExporting, false);
  useExportRuntimeStore.setState({ isExporting: true, ownsRun: false });
  useExportRuntimeStore.getState().applyBackendStatus({
    is_export_active: false,
    last_op_kind: "merge",
    last_op_status: "success",
    last_op_output_path: "/out/merged",
  } as Parameters<typeof store.applyBackendStatus>[0]);
  assert.notEqual(useExportRuntimeStore.getState().phase, "success");
  useExportRuntimeStore.getState().reset();
});

test("an export settles on its own record even after a merge finished over it", async () => {
  stub.resetStub();
  // Polls pass the op id, so the server answers with this export's kept record.
  stub.responses.set("getExportStatus", (opId?: string) =>
    opId === "export-1"
      ? done(4, "export_gguf", "export-1", "/out/gguf")
      : done(5, "merge", "merge-1", "/out/merged"),
  );
  const result = await recoverViaStatus(3, () => true, "export-1");
  assert.deepEqual(result, { outputPath: "/out/gguf" });
  assert.ok(stub.calls.every((c) => c.args[0] === "export-1"));
});

test("a kept record settles recovery while a queued merge already runs", async () => {
  stub.resetStub();
  stub.responses.set("getExportStatus", () => ({
    ...active("merge", "merge-1"),
    last_op_seq: 4,
    last_op_kind: "export_gguf",
    last_op_id: "export-1",
    last_op_status: "success",
    last_op_output_path: "/out/gguf",
  }));
  const result = await recoverViaStatus(3, () => true, "export-1");
  assert.deepEqual(result, { outputPath: "/out/gguf" });
});

test("Cancel during an export stops only that export's op", async () => {
  stub.resetStub();
  const store = useExportRuntimeStore.getState();
  store.reset();
  stub.responses.set("loadCheckpoint", async () => {
    await useExportRuntimeStore.getState().requestCancel();
    return { success: true, message: "ok", details: null };
  });
  await store.runExport({
    sourceMode: "model",
    checkpointPath: null,
    source: "unsloth/Qwen3-0.6B",
    modelSource: "hf",
    trustRemoteCode: false,
    exportMethod: "gguf",
    isAdapter: false,
    quantLevels: ["q4_k_m"],
    saveDirectory: "out",
    destination: "local",
    privateRepo: false,
    summary: {},
  } as unknown as Parameters<typeof store.runExport>[0]);
  const load = stub.calls.find((c) => c.name === "loadCheckpoint");
  const cancel = stub.calls.find((c) => c.name === "cancelExport");
  assert.ok(load && cancel);
  assert.equal(cancel.args[0], (load.args[0] as { op_id: string }).op_id);
  useExportRuntimeStore.getState().reset();
});
