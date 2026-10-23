// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import { type ReactElement, useEffect, useMemo, useRef, useState } from "react";
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
  findTrainedModelInput,
  formatBenchmarkDelta,
  formatBenchmarkScore,
  getBenchmarkHoldout,
  pickBenchmarkExecution,
} from "../../utils/benchmark";

const POLL_MS = 1000;

const SCORER_SHORT: Record<BenchmarkConfig["scorer"], string> = {
  reference: "Reference match",
  judge: "LLM judge",
  json: "Valid JSON",
  markdown: "Valid markdown",
  python: "Valid Python",
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

function FailedExamplesDialog({
  name,
  result,
  open,
  onOpenChange,
}: {
  name: string;
  result: BenchmarkResult;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}): ReactElement {
  const shown = result.failed_examples.length;
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Failed examples</DialogTitle>
          <DialogDescription>
            {result.failed_total} of {result.candidate.total} held-out rows failed
            {shown < result.failed_total ? `; the first ${shown} are kept.` : "."}
          </DialogDescription>
        </DialogHeader>
        <div className="max-h-[60vh] space-y-2 overflow-y-auto pr-1">
          {result.failed_examples.map((example, index) => (
            <div
              // biome-ignore lint/suspicious/noArrayIndexKey: static list
              key={index}
              className="corner-squircle space-y-1 rounded-lg border border-border/60 p-3 text-sm"
            >
              <p className="font-medium">{example.prompt}</p>
              <p className="text-muted-foreground">
                <span className="text-foreground">Got:</span>{" "}
                {example.output || "(no answer)"}
              </p>
              {example.reference && (
                <p className="text-muted-foreground">
                  <span className="text-foreground">Expected:</span> {example.reference}
                </p>
              )}
              {example.reason && (
                <p className="text-xs text-muted-foreground">{example.reason}</p>
              )}
            </div>
          ))}
        </div>
        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={() => downloadFailed(name, result)}
          >
            Download JSONL
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
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
  const edges = useRecipeStudioStore((state) => state.edges);
  const configs = useRecipeStudioStore((state) => state.configs);
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [failedOpen, setFailedOpen] = useState(false);
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

  const trained = useMemo(
    () => findTrainedModelInput(config.id, edges, configs),
    [config.id, edges, configs],
  );
  const execution = pickBenchmarkExecution(executions);
  const total = execution ? execution.datasetTotal ?? execution.dataset.length : 0;
  const holdout = getBenchmarkHoldout(total, config.holdout_percent, config.max_rows);
  const running = Boolean(jobId) || starting;
  const sourceLabel = trained
    ? `${trained.holdoutRows} rows ${trained.trainName} held out`
    : execution
      ? `${holdout.limit} held-out rows`
      : "Run the recipe first";

  const run = async () => {
    setError(null);
    if (!(trained || execution)) {
      setError("Run the recipe first: the benchmark scores rows from the last finished run.");
      return;
    }
    setStarting(true);
    try {
      let rows: Record<string, unknown>[] = [];
      if (!trained && execution) {
        if (execution.dataset.length >= total) {
          rows = execution.dataset.slice(holdout.offset, holdout.offset + holdout.limit);
        } else if (execution.jobId) {
          const page = await getRecipeJobDataset(execution.jobId, {
            offset: holdout.offset,
            limit: holdout.limit,
          });
          rows = (page.dataset ?? []) as Record<string, unknown>[];
        }
      }
      const allConfigs = Object.values(useRecipeStudioStore.getState().configs);
      const { request, errors } = buildBenchmarkRequest(config, allConfigs, rows, trained);
      if (!request) {
        setError(errors.join(" "));
        return;
      }
      executionIdRef.current = execution?.id ?? "";
      rowsRef.current = trained ? request.max_rows ?? 0 : rows.length;
      const created = await startBenchmarkJob(request);
      setProgress(null);
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
    <div className="nodrag space-y-3">
      <p className="text-xs text-muted-foreground">
        {config.model_alias || "No model"} · {SCORER_SHORT[config.scorer]} · {sourceLabel}
      </p>
      {result && !running && (
        <div className="space-y-2">
          <div className="flex items-end justify-between gap-3">
            <div>
              <p className="text-3xl font-semibold leading-none tabular-nums">
                {formatBenchmarkScore(result.candidate.score)}
              </p>
              <p className="mt-1 text-xs text-muted-foreground tabular-nums">
                {result.candidate.passed} of {result.candidate.scored} passed
                {result.candidate.errors > 0 ? ` · ${result.candidate.errors} errored` : ""}
                {result.cancelled ? " · stopped early" : ""}
              </p>
            </div>
            {result.baseline && (
              <div className="text-right">
                <p
                  className={cn(
                    "text-lg font-semibold leading-none tabular-nums",
                    typeof delta === "number" && delta > 0 && "text-emerald-600 dark:text-emerald-400",
                    typeof delta === "number" && delta < 0 && "text-rose-600 dark:text-rose-400",
                  )}
                >
                  {formatBenchmarkDelta(delta)}
                </p>
                <p className="mt-1 text-xs text-muted-foreground tabular-nums">
                  vs baseline {formatBenchmarkScore(result.baseline.score)}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
      {running && (
        <div className="space-y-1.5">
          <Progress value={percent} />
          <p className="text-xs tabular-nums text-muted-foreground">
            {progress ? `Scoring ${progress.done} of ${progress.total} answers` : "Starting..."}
          </p>
        </div>
      )}
      {error && <p className="text-xs text-destructive">{error}</p>}
      <div className="flex items-center gap-1.5">
        {running ? (
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={!jobId}
            onClick={() => jobId && void cancelBenchmarkJob(jobId)}
          >
            Stop
          </Button>
        ) : (
          <Button type="button" size="sm" disabled={disabled} onClick={() => void run()}>
            {result ? "Run again" : "Run benchmark"}
          </Button>
        )}
        {result && result.failed_total > 0 && !running && (
          <Button
            type="button"
            size="sm"
            variant="ghost"
            onClick={() => setFailedOpen(true)}
          >
            {result.failed_total} failed
          </Button>
        )}
      </div>
      {result && (
        <FailedExamplesDialog
          name={config.name}
          result={result}
          open={failedOpen}
          onOpenChange={setFailedOpen}
        />
      )}
    </div>
  );
}
