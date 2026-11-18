// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings > System lists the other vendor's cards on an NVIDIA + AMD host (#12248).

import assert from "node:assert/strict";
import test from "node:test";

import {
  backendLabel,
  otherVendorGpuRows,
} from "../src/features/settings/lib/other-vendor-gpus.ts";
import { readSrcAsync } from "./helpers/kit.ts";

const { en } = await import("../src/i18n/locales/en.ts");

test("a single-vendor host has no other-vendor rows", () => {
  assert.deepEqual(otherVendorGpuRows(undefined), []);
  assert.deepEqual(otherVendorGpuRows([]), []);
});

test("a nameless sysfs AMD row falls back to its vendor and keeps its qualified id", () => {
  const [row] = otherVendorGpuRows([
    {
      vendor: "amd",
      backend: "rocm",
      index: 0,
      qualified_id: "rocm:0",
      name: null,
      memory_total_gb: 31.86,
      trainable: false,
    },
  ]);
  assert.deepEqual(row, {
    key: "rocm:0",
    name: null,
    vendorLabel: "AMD",
    qualifiedId: "rocm:0",
    memoryTotalGb: 31.86,
  });
});

test("an NVIDIA row beside ROCm torch keeps its name; unknown capacity stays unknown", () => {
  const [row] = otherVendorGpuRows([
    {
      vendor: "nvidia",
      index: 0,
      qualified_id: "cuda:0",
      name: "NVIDIA GeForce RTX 4090",
      memory_total_gb: null,
    },
  ]);
  assert.equal(row.name, "NVIDIA GeForce RTX 4090");
  assert.equal(row.vendorLabel, "NVIDIA");
  assert.equal(row.memoryTotalGb, null);
});

test("the note names the installed torch's backend", () => {
  assert.equal(backendLabel("cuda"), "CUDA");
  assert.equal(backendLabel("rocm"), "ROCm");
  assert.match(en.settings.resources.gpu.otherVendorNote, /\{backend\} PyTorch/);
});

test("the System tab renders other_vendor_devices from the training inventory", async () => {
  const tab = await readSrcAsync("features/settings/tabs/resources-tab.tsx");
  assert.match(tab, /otherVendorGpuRows\(gpuInventory\?\.other_vendor_devices\)/);
  assert.match(tab, /settings\.resources\.gpu\.otherVendorNote/);
});
