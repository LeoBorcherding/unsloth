// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ModelSelector } from "@/features/model-picker/components/model-selector";
import type {
  LoraModelOption,
  ModelOption,
  ModelSelectorChangeMeta,
} from "@/features/model-picker/components/model-selector/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Checkbox } from "@/components/ui/checkbox";
import { Spinner } from "@/components/ui/spinner";
import { prepareHfTokenForUse } from "@/features/hf-auth";
import {
  type LocalModelInfo,
  listLocalModels,
} from "@/features/training";
import { useHfTokenStore } from "@/features/hub";
import {
  DOWNLOAD_KIND,
  downloadManager,
  subscribeJobListeners,
} from "@/features/hub/download-manager";
import { useT } from "@/i18n";
import { AlertCircleIcon, Rocket01Icon } from "@hugeicons/core-free-icons";
import { cn } from "@/lib/utils";
import { BENCH_CARD, RUN_BUTTON } from "@/features/benchmarks/components/bench-ui";
import { CountInput, Field } from "@/features/benchmarks/components/setup-panel";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useDebouncedValue } from "@/hooks";
import type { BenchmarkTaskConfig, BenchmarkTaskInfo, ModelCheckpoints } from "./api/benchmark-api";
import { fetchBenchmarkTaskConfig, fetchBenchmarkTasks, fetchCheckpoints } from "./api/benchmark-api";
import { BenchmarkRunPanel } from "./components/benchmark-run-panel";
import { BenchmarkHistoryPanel } from "./components/benchmark-history-panel";
import {
  isBenchmarkPanelActive,
  useBenchmarkRuntimeStore,
} from "./stores/benchmark-runtime-store";
import { useChatRuntimeStore, useChatModelRuntime } from "@/features/chat";

/** The Evals tab of the Benchmarks page: lm-eval tasks on a picked model. */
export function BenchmarkPage() {
  const t = useT();
  const hfToken = useHfTokenStore((s) => s.token);

  const [trainingModels, setTrainingModels] = useState<ModelCheckpoints[]>([]);
  const [loadingCheckpoints, setLoadingCheckpoints] = useState(true);
  const [checkpointError, setCheckpointError] = useState<string | null>(null);

  const [localModels, setLocalModels] = useState<LocalModelInfo[]>([]);
  const [isLoadingLocalModels, setIsLoadingLocalModels] = useState(true);
  const [localModelsError, setLocalModelsError] = useState<string | null>(null);

  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [selectedModelSource, setSelectedModelSource] = useState<"hub" | "lora" | "exported" | "local" | "external" | null>(null);
  const [selectedGgufVariant, setSelectedGgufVariant] = useState<string | null>(null);
  const [selectedModelIsDownloaded, setSelectedModelIsDownloaded] = useState<boolean | null>(null);
  const [downloadingForBenchmark, setDownloadingForBenchmark] = useState(false);

  const [batchSize, setBatchSize] = useState(0); // 0 = auto
  const [logSamples, setLogSamples] = useState(true);
  const [numFewshot, setNumFewshot] = useState(0);
  const [maxTokens, setMaxTokens] = useState(32768);
  const [outputPath, setOutputPath] = useState("");

  const { selectModel, loadingModel, refresh } = useChatModelRuntime();
  const inferenceParams = useChatRuntimeStore((s) => s.params);
  const activeGgufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const modelLoading = useChatRuntimeStore((s) => s.modelLoading);

  useEffect(() => {
    if (inferenceParams.checkpoint && inferenceParams.checkpoint !== selectedModel) {
      setSelectedModel(inferenceParams.checkpoint);
      setSelectedGgufVariant(activeGgufVariant);
    } else if (!inferenceParams.checkpoint && selectedModel) {
      setSelectedModel(null);
      setSelectedGgufVariant(null);
      setDownloadingForBenchmark(false);
    }
  }, [inferenceParams.checkpoint, activeGgufVariant, selectedModel]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const [tasks, setTasks] = useState<BenchmarkTaskInfo[]>([]);
  const [loadingTasks, setLoadingTasks] = useState(true);
  const [selectedTask, setSelectedTask] = useState("mmlu");
  const [taskConfig, setTaskConfig] = useState<BenchmarkTaskConfig | null>(null);
  const fewshotSupported = taskConfig?.num_fewshot !== 0;

  // Fetch task config when selected task changes
  useEffect(() => {
    let cancelled = false;
    setTaskConfig(null);
    fetchBenchmarkTaskConfig(selectedTask)
      .then((cfg) => {
        if (!cancelled) {
          setTaskConfig(cfg);
          if (cfg.num_fewshot != null && cfg.num_fewshot > 0) {
            setNumFewshot(cfg.num_fewshot);
          }
        }
      })
      .catch(() => {
        if (!cancelled) setTaskConfig(null);
      });
    return () => { cancelled = true; };
  }, [selectedTask]);
  const [taskSearch, setTaskSearch] = useState("");
  const taskAnchorRef = useRef<HTMLDivElement>(null);
  const selectingTaskRef = useRef(false);

  const [panelOpen, setPanelOpen] = useState(false);
  const downloadUnsubRef = useRef<(() => void) | null>(null);

  const runBenchmark = useBenchmarkRuntimeStore((s) => s.run);
  const resetBenchmarkRun = useBenchmarkRuntimeStore((s) => s.reset);
  const isRunning = useBenchmarkRuntimeStore((s) => s.isRunning);
  const panelActive = useBenchmarkRuntimeStore(isBenchmarkPanelActive);

  useEffect(() => {
    let cancelled = false;
    setLoadingCheckpoints(true);
    setCheckpointError(null);
    fetchCheckpoints()
      .then((data) => {
        if (!cancelled) {
          setTrainingModels(data.models);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setCheckpointError(
            err instanceof Error ? err.message : "Failed to load checkpoints",
          );
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingCheckpoints(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void listLocalModels(controller.signal)
      .then((models) => {
        if (controller.signal.aborted) return;
        setLocalModels(models);
      })
      .catch((error) => {
        if (controller.signal.aborted) return;
        setLocalModelsError(
          error instanceof Error
            ? error.message
            : t("benchmark.failedToLoadLocalModels"),
        );
      })
      .finally(() => {
        if (controller.signal.aborted) return;
        setIsLoadingLocalModels(false);
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoadingTasks(true);
    fetchBenchmarkTasks()
      .then((data) => {
        if (!cancelled) {
          setTasks(data.tasks);
        }
      })
      .catch(() => {
        // silently ignore — the dropdown will just be empty
      })
      .finally(() => {
        if (!cancelled) setLoadingTasks(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const models = useMemo<ModelOption[]>(() => {
    const seen = new Set<string>();
    const result = localModels
      .filter((m) => {
        if (seen.has(m.id)) return false;
        seen.add(m.id);
        return true;
      })
      .map((m) => ({
        id: m.id,
        name: m.display_name ?? m.id,
        description:
          m.source === "hf_cache"
            ? t("benchmark.hfCache")
            : m.source === "custom"
              ? t("benchmark.customFolders")
              : t("benchmark.localDir"),
      }));
    return result;
  }, [localModels]);

  const debouncedTaskSearch = useDebouncedValue(taskSearch, 150);

  // Pre-compute lowercased lookup and curated/non-curated split (stable across searches)
  const taskIndex = useMemo(() => {
    const curated = new Set([
      "mmlu", "mmlu_pro", "gpqa_main_cot_zeroshot", "hellaswag",
      "arc_challenge", "winogrande", "gsm8k", "hendrycks_math",
      "ifeval", "humaneval", "truthfulqa_mc1", "longbench",
      "bbh", "bbh_fewshot", "bbh_zeroshot",
    ]);
    const curatedIds: string[] = [];
    const rest: Array<{ id: string; lowerId: string; lowerName: string }> = [];
    for (const t of tasks) {
      const entry = { id: t.id, lowerId: t.id.toLowerCase(), lowerName: t.name.toLowerCase() };
      if (curated.has(t.id)) {
        curatedIds.push(t.id);
      } else {
        rest.push(entry);
      }
    }
    return { curatedIds, rest };
  }, [tasks]);

  const taskItems = useMemo(() => {
    const q = debouncedTaskSearch.toLowerCase().trim();
    if (!q) {
      return [...taskIndex.curatedIds, ...taskIndex.rest.map((e) => e.id)].slice(0, 50);
    }

    const words = q.split(/\s+/).filter(Boolean);
    const starts: string[] = [];
    const contains: string[] = [];
    for (const { id, lowerId, lowerName } of taskIndex.rest) {
      const matchesAll = words.every((w) => lowerId.includes(w) || lowerName.includes(w));
      if (!matchesAll) continue;
      if (lowerId.startsWith(q) || lowerName.startsWith(q)) {
        starts.push(id);
      } else {
        contains.push(id);
      }
    }
    // Also search curated tasks
    for (const id of taskIndex.curatedIds) {
      const lowerId = id.toLowerCase();
      const matchesAll = words.every((w) => lowerId.includes(w));
      if (!matchesAll) continue;
      if (lowerId.startsWith(q)) {
        starts.unshift(id);
      } else {
        starts.push(id);
      }
    }
    return [...starts, ...contains].slice(0, 50);
  }, [taskIndex, debouncedTaskSearch]);

  const loraModels = useMemo<LoraModelOption[]>(() => {
    const result: LoraModelOption[] = [];
    for (const run of trainingModels) {
      const tsMatch = run.name.match(/_(\d{10,})$/);
      const displayName = tsMatch
        ? run.name.slice(0, tsMatch.index)
        : run.name;
      const timeStr = tsMatch
        ? new Date(Number(tsMatch[1]) * 1000).toLocaleString(undefined, {
          dateStyle: "medium",
          timeStyle: "short",
        })
        : null;

      for (const cp of run.checkpoints) {
        const label = timeStr ? `${displayName} · ${cp.display_name} · ${timeStr}` : `${displayName} · ${cp.display_name}`;
        result.push({
          id: cp.path,
          name: label,
          description: run.base_model ?? undefined,
          baseModel: run.base_model ?? undefined,
          source: "training",
        });
      }
    }
    return result;
  }, [trainingModels]);

  const handleModelChange = useCallback(
    (value: string, meta: ModelSelectorChangeMeta) => {
      if (value !== selectedModel) {
        downloadUnsubRef.current?.();
        downloadUnsubRef.current = null;
        setPanelOpen(false);
      }
      setSelectedModel(value);
      setSelectedModelSource(meta.isLora ? "lora" : meta.source);
      setSelectedGgufVariant(meta.ggufVariant ?? null);
      setSelectedModelIsDownloaded(meta.isDownloaded ?? null);
      setDownloadingForBenchmark(false);
      if (value && value !== (inferenceParams.checkpoint ?? undefined)) {
        void selectModel({
          id: value,
          source: meta.source,
          isLora: meta.isLora,
          ggufVariant: meta.ggufVariant,
          isDownloaded: meta.isDownloaded,
          expectedBytes: meta.expectedBytes,
          isGguf: meta.isGguf,
          forceReload: true,
          throwOnError: true,
        });
      }
    },
    [selectedModel, inferenceParams.checkpoint, selectModel],
  );

  const handleStartAndOpenPanel = useCallback(() => {
    if (!selectedModel) return;

    const resolvedSource = selectedModelSource === "lora" ? "checkpoint" : "local";

    if (!isRunning) {
      resetBenchmarkRun();
    }
    setPanelOpen(false);

    const extraParams = {
      batch_size: batchSize > 0 ? String(batchSize) : "auto",
      log_samples: logSamples,
      num_fewshot: numFewshot > 0 ? numFewshot : null,
      max_tokens: maxTokens,
      output_path: outputPath || null,
    };

    const needsDownload =
      selectedModelSource === "hub" &&
      selectedModelIsDownloaded === false &&
      !downloadingForBenchmark;
    if (needsDownload) {
      setDownloadingForBenchmark(true);
      downloadUnsubRef.current = subscribeJobListeners(DOWNLOAD_KIND.MODEL, selectedModel, {
        onComplete: () => {
          downloadUnsubRef.current?.();
          downloadUnsubRef.current = null;
          setDownloadingForBenchmark(false);
          setSelectedModelIsDownloaded(true);
          setPanelOpen(true);
          prepareHfTokenForUse(hfToken, { allowAnonymous: true }).then((preparedToken) => {
            if (preparedToken.proceed) {
              void runBenchmark(selectedModel, resolvedSource, selectedGgufVariant, selectedTask, extraParams);
            }
          });
        },
        onError: () => {
          downloadUnsubRef.current?.();
          downloadUnsubRef.current = null;
          setDownloadingForBenchmark(false);
          setPanelOpen(false);
        },
        onCancelled: () => {
          downloadUnsubRef.current?.();
          downloadUnsubRef.current = null;
          setDownloadingForBenchmark(false);
          setPanelOpen(false);
        },
      });
      void downloadManager.requestStart({
        kind: DOWNLOAD_KIND.MODEL,
        repoId: selectedModel,
        variant: selectedGgufVariant,
        expectedBytes: 0,
      });
      return;
    }

    setPanelOpen(true);
    prepareHfTokenForUse(hfToken, { allowAnonymous: true }).then((preparedToken) => {
      if (preparedToken.proceed) {
        void runBenchmark(selectedModel, resolvedSource, selectedGgufVariant, selectedTask, extraParams);
      }
    });
  }, [selectedModel, selectedModelSource, selectedGgufVariant, selectedModelIsDownloaded, downloadingForBenchmark, selectedTask, hfToken, runBenchmark, isRunning, resetBenchmarkRun, batchSize, logSamples, numFewshot, maxTokens, outputPath]);

  const handleClosePanel = useCallback(() => {
    resetBenchmarkRun();
    setPanelOpen(false);
  }, [resetBenchmarkRun]);

  const showPanel = panelOpen || panelActive;

  const panelEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!showPanel) return;
    const id = window.setTimeout(() => {
      panelEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    }, 60);
    return () => window.clearTimeout(id);
  }, [showPanel]);

  return (
    <div className="@container/evals">
      <div className="grid grid-cols-1 items-start gap-6 @3xl/evals:grid-cols-[calc(264px*var(--ui-space-scale,1))_minmax(0,1fr)]">
        <div
          className={cn(
            BENCH_CARD,
            "flex min-w-0 flex-col gap-6 px-5 pb-5 pt-4 @3xl/evals:sticky @3xl/evals:top-6",
          )}
        >
          <span className="text-ui-11 font-medium tracking-nav text-muted-foreground">
            Setup
          </span>

          {loadingCheckpoints && (
            <div className="flex items-center gap-2 text-ui-12 text-muted-foreground">
              <Spinner className="size-4" />
              {t("benchmark.loadingCheckpoints")}
            </div>
          )}
          {checkpointError && (
            <div className="flex items-center gap-2 text-ui-12 text-destructive">
              <HugeiconsIcon icon={AlertCircleIcon} className="size-4" />
              {checkpointError}
            </div>
          )}

          {!loadingCheckpoints && !checkpointError && (
            <>
              <Field label={t("benchmark.modelLabel")}>
                <ModelSelector
                  models={models}
                  loraModels={loraModels}
                  externalModels={[]}
                  value={selectedModel ?? undefined}
                  activeGgufVariant={activeGgufVariant}
                  onValueChange={handleModelChange}
                  variant="muted"
                  className="w-full bg-accent!"
                />
                {isLoadingLocalModels && (
                  <span className="text-ui-11 text-muted-foreground">
                    {t("benchmark.scanningLocalModels")}
                  </span>
                )}
                {localModelsError && (
                  <span className="text-ui-11 text-destructive">
                    {localModelsError}
                  </span>
                )}
              </Field>

              <Field label={t("benchmark.taskLabel")}>
                {loadingTasks ? (
                  <span className="flex h-9 items-center gap-2 text-ui-12 text-muted-foreground">
                    <Spinner className="size-3.5" />
                    {t("benchmark.loadingTasks")}
                  </span>
                ) : (
                  <div ref={taskAnchorRef}>
                    <Combobox
                      items={taskItems}
                      filteredItems={taskItems}
                      filter={null}
                      value={selectedTask}
                      onValueChange={(next) => {
                        if (next) setSelectedTask(next);
                      }}
                      onInputValueChange={(next) => {
                        if (selectingTaskRef.current) {
                          selectingTaskRef.current = false;
                          return;
                        }
                        setTaskSearch(next);
                      }}
                      itemToStringValue={(item) => {
                        const info = tasks.find((x) => x.id === item);
                        return info ? info.name : item;
                      }}
                      autoHighlight={true}
                    >
                      <ComboboxInput
                        placeholder={t("benchmark.searchTaskPlaceholder")}
                        className="w-full"
                        showClear={true}
                      />
                      <ComboboxContent anchor={taskAnchorRef}>
                        <ComboboxEmpty>{t("benchmark.noTasksFound")}</ComboboxEmpty>
                        <ComboboxList>
                          {taskItems.map((id) => {
                            const info = tasks.find((x) => x.id === id);
                            const tl = info?.task_type;
                            const typeLabel = !tl
                              ? null
                              : tl.includes("log")
                                ? "log likelihood"
                                : tl === "greedy_until"
                                  ? "generation"
                                  : tl;
                            return (
                              <ComboboxItem
                                key={id}
                                value={id}
                                onPointerDown={() => {
                                  selectingTaskRef.current = true;
                                }}
                              >
                                <span className="truncate font-medium">
                                  {info?.name ?? id}
                                </span>
                                {typeLabel && (
                                  <span className="ml-2 shrink-0 text-xs text-muted-foreground">
                                    {typeLabel}
                                  </span>
                                )}
                              </ComboboxItem>
                            );
                          })}
                        </ComboboxList>
                      </ComboboxContent>
                    </Combobox>
                  </div>
                )}
              </Field>

              <div className="grid grid-cols-2 gap-2">
                <Field label={t("benchmark.numFewshotLabel")}>
                  {fewshotSupported ? (
                    <CountInput
                      value={numFewshot}
                      min={0}
                      max={20}
                      step={1}
                      onCommit={setNumFewshot}
                    />
                  ) : (
                    <span className="flex h-9 items-center text-ui-11 text-muted-foreground/60">
                      {t("benchmark.fewshotNotSupported")}
                    </span>
                  )}
                </Field>
                <Field label={t("benchmark.batchSizeLabel")}>
                  <Input
                    type="number"
                    value={batchSize > 0 ? String(batchSize) : ""}
                    onChange={(e) => {
                      const v = e.target.value;
                      setBatchSize(v === "" ? 0 : Math.max(0, Number(v) || 0));
                    }}
                    placeholder="auto"
                    min={0}
                    max={512}
                    step={1}
                    className="text-center font-mono tabular-nums"
                  />
                </Field>
              </div>

              <Field label={t("benchmark.maxTokensLabel")}>
                <CountInput
                  value={maxTokens}
                  min={1024}
                  max={262144}
                  step={1024}
                  onCommit={setMaxTokens}
                />
              </Field>

              <Field label={t("benchmark.outputPathLabel")}>
                <Input
                  type="text"
                  value={outputPath}
                  onChange={(e) => setOutputPath(e.target.value)}
                  placeholder={t("benchmark.outputPathPlaceholder")}
                />
                <label className="flex items-center gap-2.5 text-ui-12p5 text-foreground">
                  <Checkbox
                    checked={logSamples}
                    onCheckedChange={(checked) => setLogSamples(checked === true)}
                  />
                  {t("benchmark.logSamplesLabel")}
                </label>
              </Field>

              {!showPanel && (
                <Button
                  size="lg"
                  className={RUN_BUTTON}
                  disabled={
                    !selectedModel ||
                    modelLoading ||
                    !!loadingModel ||
                    downloadingForBenchmark
                  }
                  onClick={handleStartAndOpenPanel}
                >
                  <HugeiconsIcon
                    icon={Rocket01Icon}
                    strokeWidth={1.75}
                    className="size-4"
                  />
                  {loadingModel
                    ? t("benchmark.loadingModel")
                    : downloadingForBenchmark
                      ? t("benchmark.downloadingModel")
                      : t("benchmark.runButton")}
                </Button>
              )}
            </>
          )}
        </div>

        <div className="flex min-w-0 flex-col gap-4">
          {showPanel && (
            <section className={cn(BENCH_CARD, "p-4 sm:p-5")}>
              <BenchmarkRunPanel onClose={handleClosePanel} />
              <div ref={panelEndRef} aria-hidden="true" className="h-px w-full" />
            </section>
          )}
          <section className={cn(BENCH_CARD, "p-4 sm:p-5")}>
            <BenchmarkHistoryPanel />
          </section>
        </div>
      </div>
    </div>
  );
}
