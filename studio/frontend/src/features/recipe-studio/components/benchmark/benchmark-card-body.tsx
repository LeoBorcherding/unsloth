// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import { type ReactElement, useEffect, useRef, useState } from "react";
import {
  cancelBenchmarkJob,
  getBenchmarkJob,
  getRecipeJobDataset,
  startBenchmarkJob,
} from "../../api";
import { useRecipeExecutionsStore } from "../../stores/recipe-executions";
import { useRecipeStudioStore } from "../../stores/recipe-studio";
import type { BenchmarkConfig, BenchmarkResult } from "../../types";
import {
  buildBenchmarkRequest,
  failedExamplesToJsonl,
  formatBenchmarkDelta,
  formatBenchmarkScore,
  getBenchmarkHoldout,
  pickBenchmarkExecution,
} from "../../utils/benchmark";

const POLL_MS = 1000;

const SCORER_SHORT: Record<BenchmarkConfig["scorer"], string> = {
  reference: "reference match",
  judge: "LLM judge",
  json: "valid JSON",
  markdown: "valid markdown",
  python: "valid Python",
};

function downloadFailed(name: string, result: BenchmarkResult): void {
  const blob = new Blob([failedExamplesToJsonl(result)], {
    type: "application/x-ndjson",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${name}-failed.jsonl`;
  anchor.click();
  URL.revokeObjectURL(url);
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "up" | "down";
}): ReactElement {
  return (
    <div className="corner-squircle rounded-md border border-border/60 px-2 py-1.5">
      <p className="text-ui-10 uppercase text-muted-foreground">{label}</p>
      <p
        className={cn(
          "text-sm font-semibold tabular-nums",
          tone === "up" && "text-emerald-600 dark:text-emerald-400",
          tone === "down" && "text-rose-600 dark:text-rose-400",
        )}
      >
        {value}
      </p>
    </div>
  );
}

export function BenchmarkCardBody({
  config,
  disabled,
}: {
  config: BenchmarkConfig;
  disabled: boolean;
}): ReactElement {
  const executions = useRecipeExecutionsStore((state) => state.executions);
  const updateConfig = useRecipeStudioStore((state) => state.updateConfig);
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const executionIdRef = useRef<string>("");
  const rowsRef = useRef(0);

  useEffect(() => {
    if (!jobId) {
      return;
    }
    let stopped = false;
    const timer = window.setInterval(async () => {
      try {
        const status = await getBenchmarkJob(jobId);
        if (stopped) {
          return;
        }
        setProgress(status.progress);
        if (status.status === "running") {
          return;
        }
        window.clearInterval(timer);
        setJobId(null);
        if (status.status === "error") {
          setError(status.error ?? "Benchmark failed.");
          return;
        }
        if (status.result) {
          updateConfig(config.id, {
            // biome-ignore lint/style/useNamingConvention: ui schema
            last_result: {
              ...status.result,
              // biome-ignore lint/style/useNamingConvention: ui schema
              finished_at: Date.now(),
              // biome-ignore lint/style/useNamingConvention: ui schema
              execution_id: executionIdRef.current,
              rows: rowsRef.current,
            },
          });
        }
      } catch (err) {
        window.clearInterval(timer);
        setJobId(null);
        setError(err instanceof Error ? err.message : String(err));
      }
    }, POLL_MS);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [jobId, config.id, updateConfig]);

  const execution = pickBenchmarkExecution(executions);
  const total = execution ? execution.datasetTotal ?? execution.dataset.length : 0;
  const holdout = getBenchmarkHoldout(total, config.holdout_percent, config.max_rows);
  const running = Boolean(jobId) || starting;

  const run = async () => {
    setError(null);
    if (!execution) {
      setError("Run the recipe first: the benchmark scores rows from the last finished run.");
      return;
    }
    setStarting(true);
    try {
      let rows: Record<string, unknown>[];
      if (execution.dataset.length >= total) {
        rows = execution.dataset.slice(holdout.offset, holdout.offset + holdout.limit);
      } else if (execution.jobId) {
        const page = await getRecipeJobDataset(execution.jobId, {
          offset: holdout.offset,
          limit: holdout.limit,
        });
        rows = (page.dataset ?? []) as Record<string, unknown>[];
      } else {
        rows = [];
      }
      const allConfigs = Object.values(useRecipeStudioStore.getState().configs);
      const { request, errors } = buildBenchmarkRequest(config, allConfigs, rows);
      if (!request) {
        setError(errors.join(" "));
        return;
      }
      executionIdRef.current = execution.id;
      rowsRef.current = rows.length;
      const created = await startBenchmarkJob(request);
      setProgress({ done: 0, total: rows.length });
      setJobId(created.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setStarting(false);
    }
  };

  const result = config.last_result;
  const delta = result?.delta;
  const percent =
    progress && progress.total > 0 ? Math.round((progress.done / progress.total) * 100) : 0;

  return (
    <div className="nodrag space-y-2">
      <p className="text-xs text-muted-foreground">
        {config.model_alias || "No model"} · {SCORER_SHORT[config.scorer]}
        {execution
          ? ` · ${holdout.limit} of ${holdout.holdout} held-out rows`
          : " · run the recipe first"}
      </p>
      {result && !running && (
        <div className="space-y-1.5">
          <div className={cn("grid gap-1.5", result.baseline ? "grid-cols-3" : "grid-cols-1")}>
            <Stat label="Score" value={formatBenchmarkScore(result.candidate.score)} />
            {result.baseline && (
              <>
                <Stat label="Baseline" value={formatBenchmarkScore(result.baseline.score)} />
                <Stat
                  label="Delta"
                  value={formatBenchmarkDelta(delta)}
                  tone={
                    typeof delta === "number" && delta !== 0
                      ? delta > 0
                        ? "up"
                        : "down"
                      : undefined
                  }
                />
              </>
            )}
          </div>
          <p className="text-ui-11 text-muted-foreground">
            {result.candidate.passed}/{result.candidate.scored} passed
            {result.candidate.errors > 0 ? `, ${result.candidate.errors} errored` : ""}
            {result.cancelled ? " (cancelled)" : ""}
          </p>
          {result.failed_examples[0] && (
            <div className="corner-squircle space-y-0.5 rounded-md bg-muted/40 px-2 py-1.5 text-ui-11">
              <p className="line-clamp-2">
                <span className="font-semibold">Failed:</span>{" "}
                {result.failed_examples[0].prompt}
              </p>
              <p className="line-clamp-2 text-muted-foreground">
                Got: {result.failed_examples[0].output ?? "(no answer)"}
                {result.failed_examples[0].reference
                  ? ` · expected ${result.failed_examples[0].reference}`
                  : ""}
              </p>
            </div>
          )}
        </div>
      )}
      {running && (
        <div className="space-y-1">
          <Progress value={percent} />
          <p className="text-ui-11 tabular-nums text-muted-foreground">
            Scoring {progress?.done ?? 0}/{progress?.total ?? holdout.limit} answers
          </p>
        </div>
      )}
      {error && <p className="text-ui-11 text-destructive">{error}</p>}
      <div className="flex items-center gap-1.5">
        {running ? (
          <Button
            type="button"
            size="xs"
            variant="outline"
            disabled={!jobId}
            onClick={() => jobId && void cancelBenchmarkJob(jobId)}
          >
            Stop
          </Button>
        ) : (
          <Button type="button" size="xs" disabled={disabled} onClick={() => void run()}>
            {result ? "Run again" : "Run benchmark"}
          </Button>
        )}
        {result && result.failed_total > 0 && !running && (
          <Button
            type="button"
            size="xs"
            variant="ghost"
            onClick={() => downloadFailed(config.name, result)}
          >
            {result.failed_total} failed (JSONL)
          </Button>
        )}
      </div>
    </div>
  );
}
