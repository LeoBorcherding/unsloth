// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type DiskUsageReading = {
  total_gb?: number | null;
  free_gb?: number | null;
  used_gb?: number | null;
};

/** Space files take. total - free also counts blocks the filesystem keeps back for root (5% of an ext4
 * disk), so it is only the fallback for a backend that sends no used_gb. */
export function diskUsedGb(disk: DiskUsageReading | null | undefined): number {
  if (disk?.used_gb != null) return disk.used_gb;
  return Math.max(0, (disk?.total_gb ?? 0) - (disk?.free_gb ?? 0));
}
