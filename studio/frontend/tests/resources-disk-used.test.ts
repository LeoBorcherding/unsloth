// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { diskUsedGb } from "../src/features/settings/disk-used.ts";
import { readSrc } from "./helpers/kit.ts";

const resourcesTab = readSrc("features/settings/tabs/resources-tab.tsx");

test("used space comes from the backend, not total - free", () => {
  // 2 TB ext4 disk: 89 GB of files, 98 GB held back for root. total - free read 187 GB used.
  assert.equal(diskUsedGb({ total_gb: 1967.3, free_gb: 1779.8, used_gb: 89.4 }), 89.4);
});

test("older backends without used_gb still get total - free", () => {
  assert.equal(diskUsedGb({ total_gb: 100, free_gb: 40 }), 60);
  assert.equal(diskUsedGb(null), 0);
});

test("Resources keeps the percent and the bar on the total it shows", () => {
  assert.match(resourcesTab, /diskUsedGb\(systemInfo\.disk\)/);
  assert.match(resourcesTab, /diskUsedGb\(modelsDisk\)/);
  // The tile reads "used / total" beside the percent, so both use the same denominator.
  assert.match(resourcesTab, /diskPercent = diskTotal > 0 \? \(diskUsed \/ diskTotal\) \* 100/);
  assert.match(resourcesTab, /\(modelsDiskUsed \/ modelsDisk\.total_gb\) \* 100/);
  assert.match(resourcesTab, /totalBytes=\{metrics\.diskTotal\}/);
  assert.doesNotMatch(resourcesTab, /Math\.max\(0, diskTotal - diskFree\)/);
});
