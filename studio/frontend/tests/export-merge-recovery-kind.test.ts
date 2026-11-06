// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./helpers/export-store-resolver.mjs", import.meta.url);

const { recoverViaStatus } = await import(
  "../src/features/export/stores/export-runtime-store.ts"
);
const stub = await import("./helpers/export-api-stub.mjs");

test("a merge queued behind an export is not settled by that export's result", async () => {
  stub.resetStub();
  const statuses = [
    { is_export_active: true, active_op_kind: "export" },
    {
      is_export_active: false,
      last_op_seq: 4,
      last_op_kind: "export",
      last_op_status: "success",
      last_op_output_path: "/out/gguf",
    },
    { is_export_active: true, active_op_kind: "merge" },
    {
      is_export_active: false,
      last_op_seq: 5,
      last_op_kind: "merge",
      last_op_status: "success",
      last_op_output_path: "/out/merged",
    },
  ];
  stub.responses.set("getExportStatus", () => statuses.shift() ?? statuses.at(-1));
  const result = await recoverViaStatus(3, () => true, "merge");
  assert.deepEqual(result, { outputPath: "/out/merged" });
});
