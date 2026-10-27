// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import type { RecipeExecutionRecord } from "../execution-types";
import { createLoopRecord, resumeLoopRecord, runLoop } from "../loop/controller";
import { listRecipeLoops, saveRecipeLoop } from "../loop/loop-api";
import { createStudioLoopSteps } from "../loop/studio-steps";
import type { LoopRecord, LoopSettings } from "../loop/types";
import { useRecipeStudioStore } from "../stores/recipe-studio";
import { loopRecipeColumns } from "../utils/loop-card";
import { loopRunPayload } from "../loop/run-payload";
import type { RecipePayload } from "../utils/payload/types";

export type RecipeLoopView = {
  latest: LoopRecord | null;
  // A saved "running" loop nothing in this tab is driving (reload, closed tab, restart).
  interrupted: boolean;
  running: boolean;
  start: (settings: LoopSettings) => void;
  resume: (settings: LoopSettings) => void;
  stop: () => void;
};

export function useRecipeLoop(input: {
  recipeId: string;
  workflowName: string;
  runFullForLoop: (
    rows: number,
    runName: string,
    payload: RecipePayload,
  ) => Promise<RecipeExecutionRecord | null>;
}): RecipeLoopView {
  const { recipeId } = input;
  const [loops, setLoops] = useState<LoopRecord[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  // Rebuilt whenever the graph changes; a loop always runs the current recipe.
  const runRef = useRef(input.runFullForLoop);
  runRef.current = input.runFullForLoop;
  const nameRef = useRef(input.workflowName);
  nameRef.current = input.workflowName;

  useEffect(() => {
    let cancelled = false;
    setLoops([]);
    listRecipeLoops(recipeId)
      .then((records) => {
        if (!cancelled) {
          setLoops(records);
        }
      })
      .catch((error) => {
        // biome-ignore lint/suspicious/noConsole: loop history is optional on load
        console.error("Load recipe loops failed:", error);
      });
    return () => {
      cancelled = true;
      abortRef.current?.abort();
    };
  }, [recipeId]);

  const upsert = useCallback((record: LoopRecord): void => {
    setLoops((prev) => [record, ...prev.filter((loop) => loop.id !== record.id)]);
    saveRecipeLoop(record);
  }, []);

  const drive = useCallback(
    async (record: LoopRecord): Promise<void> => {
      if (abortRef.current) {
        return;
      }
      const controller = new AbortController();
      abortRef.current = controller;
      setActiveId(record.id);
      const store = useRecipeStudioStore.getState;
      upsert(record);
      try {
        await runLoop({
          record,
          steps: createStudioLoopSteps({
            tag: record.id.slice(0, 6),
            recipeName: () => nameRef.current,
            configs: () => store().configs,
            updateConfig: (id, patch) => store().updateConfig(id, patch),
            runRecipe: (rows, runName) => runRef.current(rows, runName, loopRunPayload(store())),
          }),
          recipeColumns: loopRecipeColumns(store().configs),
          signal: controller.signal,
          save: upsert,
        });
      } finally {
        abortRef.current = null;
        setActiveId(null);
      }
    },
    [upsert],
  );

  const latest = loops[0] ?? null;

  const start = useCallback(
    (settings: LoopSettings) => {
      void drive(
        createLoopRecord({
          id: crypto.randomUUID(),
          recipeId,
          settings,
          now: Date.now(),
        }),
      );
    },
    [drive, recipeId],
  );

  const resume = useCallback(
    (settings: LoopSettings) => {
      // Current settings apply, so a finished loop can go on with a higher round cap.
      if (latest) {
        void drive(resumeLoopRecord(latest, settings, Date.now()));
      }
    },
    [drive, latest],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return {
    latest,
    interrupted: Boolean(latest && latest.status === "running" && activeId !== latest.id),
    running: activeId !== null,
    start,
    resume,
    stop,
  };
}
