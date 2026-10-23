// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type RewardRecord, useRlWorkspaceStore } from "@/features/training";
import { useEffect, useMemo, useState } from "react";
import { getRecipeJobDataset } from "../api";
import type { RecipeExecutionRecord } from "../execution-types";
import type { TrainRewardInfo } from "../utils/train-card";

/** The reward library the Train page uses, loaded once a GRPO card needs it. */
export function useRewardLibrary(enabled: boolean): {
  library: RewardRecord[];
  libraryError: string | null;
  rewardInfo: TrainRewardInfo[];
} {
  const library = useRlWorkspaceStore((state) => state.library);
  const libraryError = useRlWorkspaceStore((state) => state.libraryError);
  const refreshLibrary = useRlWorkspaceStore((state) => state.refreshLibrary);
  useEffect(() => {
    if (enabled && library.length === 0) {
      refreshLibrary().catch(() => undefined);
    }
  }, [enabled, library.length, refreshLibrary]);
  const rewardInfo = useMemo(
    () =>
      library
        .filter((reward) => reward.valid && !reward.shadowed)
        .map((reward) => ({
          name: reward.name,
          compareTo:
            typeof reward.rule?.compare_to === "string"
              ? reward.rule.compare_to
              : null,
        })),
    [library],
  );
  return { library, libraryError, rewardInfo };
}

/** Row 1 of the run the card trains on, for the reward preview. */
export function useTrainSampleRow(
  source: RecipeExecutionRecord | null,
  enabled: boolean,
): Record<string, unknown> | null {
  const loaded = source?.dataset[0] ?? null;
  const [fetched, setFetched] = useState<Record<string, unknown> | null>(null);
  const jobId = source?.jobId ?? null;
  useEffect(() => {
    setFetched(null);
    if (!(enabled && jobId) || loaded) {
      return;
    }
    let cancelled = false;
    getRecipeJobDataset(jobId, { limit: 1 })
      .then((response) => {
        const row = response.dataset?.[0];
        if (!cancelled && row && typeof row === "object") {
          setFetched(row as Record<string, unknown>);
        }
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [enabled, jobId, loaded]);
  return loaded ?? fetched;
}
