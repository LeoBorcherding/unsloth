// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { SeedBlockType } from "../../blocks/definitions";
import type { SeedConfig, SeedSourceType } from "../../types";

const SOURCE_BY_BLOCK: Record<SeedBlockType, SeedSourceType> = {
  seed_hf: "hf",
  seed_local: "local",
  seed_unstructured: "unstructured",
  seed_github: "github_repo",
};

export function seedSourceTypeForBlock(type: SeedBlockType): SeedSourceType {
  return SOURCE_BY_BLOCK[type];
}

/** True when switching the recipe's one seed to `type` would clear something the user set up. */
export function seedSwitchLosesData(
  existing: SeedConfig,
  type: SeedBlockType,
): boolean {
  if (seedSourceTypeForBlock(type) === existing.seed_source_type) {
    return false;
  }
  return Boolean(
    existing.hf_repo_id.trim() ||
      existing.hf_path.trim() ||
      existing.local_file_name?.trim() ||
      existing.unstructured_file_ids?.length ||
      existing.github_repo_slug?.trim() ||
      existing.resolved_paths?.length ||
      existing.seed_columns?.length,
  );
}
