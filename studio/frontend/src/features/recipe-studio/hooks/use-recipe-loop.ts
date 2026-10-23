// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import type { RecipeExecutionRecord } from "../execution-types";
import {
  DEFAULT_LOOP_CONFIG,
  EVAL_LOSS_METRIC,
  createLoopRecord,
  resumeLoopRecord,
  runLoop,
} from "../loop/controller";
import { listRecipeLoops, saveRecipeLoop } from "../loop/loop-api";
import { createStudioLoopSteps } from "../loop/studio-steps";
import { datasetVersionFromArtifact } from "../loop/training-request";
import type { LoopConfig, LoopRecord, LoopSteps } from "../loop/types";

export type RecipeLoopView = {
  latest: LoopRecord | null;
  // A saved "running" loop nothing in this tab is driving (reload, closed tab, restart).
  interrupted: boolean;
  running: boolean;
  config: LoopConfig;
  setConfig: (patch: Partial<LoopConfig>) => void;
  start: () => void;
  resume: () => void;
  stop: () => void;
};

function abortError(): Error {
  const error = new Error("Stopped.");
  error.name = "AbortError";
  return error;
}

function untilAborted(signal: AbortSignal): Promise<never> {
  return new Promise((_, reject) => {
    if (signal.aborted) reject(abortError());
    signal.addEventListener("abort", () => reject(abortError()), {
      once: true,
    });
  });
}

export function useRecipeLoop(input: {
  recipeId: string;
  workflowName: string;
  runFullForLoop: (
    rows: number,
    runName: string,
  ) => Promise<RecipeExecutionRecord | null>;
}): RecipeLoopView {
  const { recipeId } = input;
  const [loops, setLoops] = useState<LoopRecord[]>([]);
  const [config, setConfigState] = useState<LoopConfig>(DEFAULT_LOOP_CONFIG);
  const [activeId, setActiveId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  // The run callback is rebuilt whenever the graph changes; a loop always uses the current one.
  const runRef = useRef(input.runFullForLoop);
  runRef.current = input.runFullForLoop;
  const nameRef = useRef(input.workflowName);
  nameRef.current = input.workflowName;

  useEffect(() => {
    let cancelled = false;
    setLoops([]);
    setConfigState(DEFAULT_LOOP_CONFIG);
    listRecipeLoops(recipeId)
      .then((records) => {
        if (cancelled) return;
        setLoops(records);
        if (records[0]) {
          setConfigState({ ...DEFAULT_LOOP_CONFIG, ...records[0].config });
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
    setLoops((prev) => [record, ...prev.filter((l) => l.id !== record.id)]);
    saveRecipeLoop(record);
  }, []);

  const drive = useCallback(
    async (record: LoopRecord): Promise<void> => {
      if (abortRef.current) return;
      const controller = new AbortController();
      abortRef.current = controller;
      setActiveId(record.id);
      const tag = record.id.slice(0, 6);
      const generate: LoopSteps["generate"] = async ({
        iteration,
        rows,
        purpose,
        signal,
      }) => {
        const base = nameRef.current.trim() || "Recipe";
        const name =
          purpose === "holdout"
            ? `${base} loop ${tag} holdout`
            : `${base} loop ${tag} v${iteration}`;
        const execution = await Promise.race([
          runRef.current(rows, name),
          untilAborted(signal),
        ]);
        if (!execution) {
          throw new Error("Generating the dataset failed. The Runs tab has the details.");
        }
        const version = datasetVersionFromArtifact(execution.artifact_path);
        if (!version) {
          throw new Error("The recipe run finished without saving a dataset.");
        }
        return {
          version,
          rows: execution.datasetTotal ?? execution.rows,
          executionId: execution.id,
        };
      };
      upsert(record);
      try {
        await runLoop({
          record,
          steps: createStudioLoopSteps(generate),
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

  const start = useCallback(() => {
    void drive(
      createLoopRecord({
        id: crypto.randomUUID(),
        recipeId,
        config,
        metric: EVAL_LOSS_METRIC,
        now: Date.now(),
      }),
    );
  }, [config, drive, recipeId]);

  const resume = useCallback(() => {
    // Current settings apply, so a finished loop can continue with a higher iteration cap.
    if (latest) void drive(resumeLoopRecord({ ...latest, config }, Date.now()));
  }, [config, drive, latest]);

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const setConfig = useCallback((patch: Partial<LoopConfig>) => {
    setConfigState((prev) => ({ ...prev, ...patch }));
  }, []);

  const running = activeId !== null;
  return {
    latest,
    interrupted: Boolean(
      latest && latest.status === "running" && activeId !== latest.id,
    ),
    running,
    config,
    setConfig,
    start,
    resume,
    stop,
  };
}
