// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { hfModelFitsDevice } from "../src/features/model-picker/components/model-selector/recommended-fit.ts";

// unsloth/Qwen3.8-Flash-Next-GGUF as the listing reports it (`expand=gguf`). 51.2B of the total
// is `per_layer_token_embd`, the 160 x 320,001,536 ngram table, read from the UD-IQ1_M header.
const FLASH_NEXT = {
  id: "unsloth/Qwen3.8-Flash-Next-GGUF",
  isGguf: true,
  totalParams: 176_943_899_520,
  ggufArchitecture: "qwen4exp",
};

const gpu = (memoryTotalGb: number, systemRamAvailableGb: number) => ({
  memoryTotalGb,
  systemRamAvailableGb,
  budgetKnown: true,
});

test("the qwen4exp ngram table is not priced as dense weights", () => {
  // The Discord report: 24 GiB card + 63 GiB RAM, 54.8 GiB of budget. Counting the table needed
  // 76.9 GiB; the measured UD-IQ1_S without it needs 47.9.
  assert.equal(hfModelFitsDevice(FLASH_NEXT, gpu(24, 63)), true);
  // A box short of even the dense part stays hidden.
  assert.equal(hfModelFitsDevice(FLASH_NEXT, gpu(16, 32)), false);
});

test("the override only applies to that architecture and total", () => {
  const sameTotalOtherArch = { ...FLASH_NEXT, ggufArchitecture: "qwen3moe" };
  assert.equal(hfModelFitsDevice(sameTotalOtherArch, gpu(24, 63)), false);
  const otherSize = { ...FLASH_NEXT, totalParams: 180_000_000_000 };
  assert.equal(hfModelFitsDevice(otherSize, gpu(24, 63)), false);
});
