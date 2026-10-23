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
import {
  cancelBenchmark,
  getBenchmarkRunDetail,
  listBenchmarkRuns,
  runBenchmark,
  streamBenchmarkLogs,
} from "@/features/benchmark/api/benchmark-api";
import { cn } from "@/lib/utils";
import { Link } from "@tanstack/react-router";
import { type ReactElement, useMemo, useState } from "react";
import { fetchLoadedModelStatus, getRecipeJobDataset } from "../../api";
import { useRecipeExecutionsStore } from "../../stores/recipe-executions";
import { useRecipeStudioStore } from "../../stores/recipe-studio";
import type { BenchmarkConfig, BenchmarkResult } from "../../types";
import {
  type LmEvalRunDetail,
  buildBenchmarkRequest,
  failedExamplesToJsonl,
  findTrainedModelInput,
  formatBenchmarkDelta,
  formatBenchmarkScore,
  getBenchmarkHoldout,
  isRecipeTask,
  lmEvalBlockedReason,
  lmEvalRunToResult,
  pickBenchmarkExecution,
} from "../../utils/benchmark";


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
            {result.failed_total} of {result.candidate.total} held-out rows
            failed
            {shown < result.failed_total
              ? `; the first ${shown} are kept.`
              : "."}
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
                  <span className="text-foreground">Expected:</span>{" "}
                  {example.reference}
                </p>
              )}
              {example.reason && (
                <p className="text-xs text-muted-foreground">
                  {example.reason}
                </p>
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
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<{
    done: number;
    total: number;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [failedOpen, setFailedOpen] = useState(false);

  const recipeTask = isRecipeTask(config.task);
  const trained = useMemo(
    () => findTrainedModelInput(config.id, edges, configs),
    [config.id, edges, configs],
  );
  const execution = pickBenchmarkExecution(executions);
  const total = execution
    ? (execution.datasetTotal ?? execution.dataset.length)
    : 0;
  const holdout = getBenchmarkHoldout(
    total,
    config.holdout_percent,
    config.max_rows,
  );
  const taskLabel = recipeTask ? SCORER_SHORT[config.scorer] : config.task;
  const sourceLabel = !recipeTask
    ? `first ${config.limit || "all"} samples · loaded GGUF`
    : trained
      ? `${trained.holdoutRows} rows ${trained.trainName} held out`
      : execution
        ? `${holdout.limit} held-out rows`
        : "Run the recipe first";

  const recipeSpec = async (): Promise<Record<string, unknown> | null> => {
    if (!(trained || execution)) {
      setError(
        "Run the recipe first: the benchmark scores rows from the last finished run.",
      );
      return null;
    }
    let rows: Record<string, unknown>[] = [];
    if (!trained && execution) {
      if (execution.dataset.length >= total) {
        rows = execution.dataset.slice(
          holdout.offset,
          holdout.offset + holdout.limit,
        );
      } else if (execution.jobId) {
        const page = await getRecipeJobDataset(execution.jobId, {
          offset: holdout.offset,
          limit: holdout.limit,
        });
        rows = (page.dataset ?? []) as Record<string, unknown>[];
      }
    }
    const allConfigs = Object.values(useRecipeStudioStore.getState().configs);
    const { request, errors } = buildBenchmarkRequest(
      config,
      allConfigs,
      rows,
      trained,
    );
    if (!request) {
      setError(errors.join(" "));
      return null;
    }
    return request;
  };

  const run = async () => {
    setError(null);
    const status = await fetchLoadedModelStatus();
    let params: Parameters<typeof runBenchmark>[0];
    if (recipeTask) {
      const spec = await recipeSpec();
      if (!spec) {
        return;
      }
      params = {
        // biome-ignore lint/style/useNamingConvention: api schema
        checkpoint_path: status?.active_model ?? "",
        // biome-ignore lint/style/useNamingConvention: api schema
        model_source: "local",
        task: config.task,
        recipe: spec,
      };
    } else {
      const blocked = lmEvalBlockedReason(status);
      if (blocked || !status?.active_model) {
        setError(blocked);
        return;
      }
      const limit = Number.parseInt(config.limit ?? "", 10);
      params = {
        // biome-ignore lint/style/useNamingConvention: api schema
        checkpoint_path: status.active_model,
        // biome-ignore lint/style/useNamingConvention: api schema
        model_source: "local",
        task: config.task,
        limit: Number.isFinite(limit) && limit > 0 ? limit : null,
      };
    }
    setRunning(true);
    setProgress(null);
    const logs = new AbortController();
    void streamBenchmarkLogs({
      signal: logs.signal,
      onEvent: (event) => {
        if (event.entry?.stream !== "progress") {
          return;
        }
        try {
          const value = JSON.parse(event.entry.line) as {
            current?: number;
            total?: number;
          };
          if (
            typeof value.current === "number" &&
            typeof value.total === "number"
          ) {
            setProgress({ done: value.current, total: value.total });
          }
        } catch {
          // not a progress payload
        }
      },
    }).catch(() => undefined);
    try {
      const response = await runBenchmark(params);
      if (!response.success) {
        setError(response.message);
        return;
      }
      let result: BenchmarkResult;
      let runIds: Record<string, string>;
      if (recipeTask) {
        const recipe = response.details?.recipe as BenchmarkResult & {
          // biome-ignore lint/style/useNamingConvention: api schema
          run_ids: Record<string, string>;
        };
        result = recipe;
        runIds = recipe.run_ids;
      } else {
        const runs = await listBenchmarkRuns();
        const latest = runs.runs.find((item) => item.task === config.task);
        if (!latest) {
          setError("The run finished but its result wasn't stored.");
          return;
        }
        const detail = await getBenchmarkRunDetail(latest.id);
        result = lmEvalRunToResult({
          ...detail,
          // biome-ignore lint/style/useNamingConvention: api schema
          default_metric: latest.default_metric,
        } as LmEvalRunDetail);
        runIds = { candidate: latest.id };
      }
      updateConfig(config.id, {
        // biome-ignore lint/style/useNamingConvention: ui schema
        last_result: {
          ...result,
          task: config.task,
          // biome-ignore lint/style/useNamingConvention: ui schema
          run_ids: runIds,
          // biome-ignore lint/style/useNamingConvention: ui schema
          finished_at: Date.now(),
          // biome-ignore lint/style/useNamingConvention: ui schema
          execution_id: execution?.id ?? "",
          rows: result.candidate.total,
        },
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      logs.abort();
      setRunning(false);
    }
  };

  const result =
    config.last_result?.task === config.task ||
    (!config.last_result?.task && recipeTask)
      ? config.last_result
      : undefined;
  const delta = result?.delta;
  const percent =
    progress && progress.total > 0
      ? Math.round((progress.done / progress.total) * 100)
      : 0;

  return (
    <div className="nodrag space-y-3">
      <p className="text-xs text-muted-foreground">
        {recipeTask ? config.model_alias || "No model" : "lm-eval"} ·{" "}
        {taskLabel} · {sourceLabel}
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
                {result.candidate.errors > 0
                  ? ` · ${result.candidate.errors} errored`
                  : ""}
                {result.cancelled ? " · stopped early" : ""}
              </p>
            </div>
            {result.baseline && (
              <div className="text-right">
                <p
                  className={cn(
                    "text-lg font-semibold leading-none tabular-nums",
                    typeof delta === "number" &&
                      delta > 0 &&
                      "text-emerald-600 dark:text-emerald-400",
                    typeof delta === "number" &&
                      delta < 0 &&
                      "text-rose-600 dark:text-rose-400",
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
            {progress
              ? `Scoring ${progress.done} of ${progress.total}`
              : "Starting..."}
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
            onClick={() => void cancelBenchmark()}
          >
            Stop
          </Button>
        ) : (
          <Button
            type="button"
            size="sm"
            disabled={disabled}
            onClick={() => void run()}
          >
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
        {result && !running && (
          <Button type="button" size="sm" variant="ghost" asChild={true}>
            <Link to="/benchmarks">History</Link>
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
