// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ModelSelector } from "@/features/model-picker/components/model-selector";
import type {
  LoraModelOption,
  ModelOption,
  ModelSelectorChangeMeta,
} from "@/features/model-picker/components/model-selector/types";
import { SectionCard } from "@/components/section-card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Checkbox } from "@/components/ui/checkbox";
import { Spinner } from "@/components/ui/spinner";
import { prepareHfTokenForUse } from "@/features/hf-auth";
import {
  type LocalModelInfo,
  listLocalModels,
  useTrainingConfigStore,
} from "@/features/training";
import {
  DOWNLOAD_KIND,
  downloadManager,
  subscribeJobListeners,
} from "@/features/hub/download-manager";
import { useT } from "@/i18n";
import { AlertCircleIcon, WorkHistoryIcon, PackageIcon } from "@hugeicons/core-free-icons";
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

export function BenchmarkPage() {
  const t = useT();
  const hfToken = useTrainingConfigStore((s) => s.hfToken);

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
            : t("studio.model.failedToLoadLocalModels"),
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
    <div className="min-h-[calc(100dvh-var(--studio-titlebar-height,0px))] bg-background">
      <main className="mx-auto max-w-7xl px-5 py-8 sm:px-9">
        <div className="mb-8 flex flex-col gap-0.5">
          <h1 className="text-[30px] font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-[34px]">
            {t("benchmark.pageTitle")}
          </h1>
          <p className="text-sm text-muted-foreground">
            {t("benchmark.pageDescription")}
          </p>
        </div>

        <SectionCard
          icon={<HugeiconsIcon icon={PackageIcon} className="size-5" />}
          title={t("benchmark.configSectionTitle")}
          description={t("benchmark.configSectionDescription")}
          accent="emerald"
          featured={true}
          className="ring-0 shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:shadow-none"
        >
          {loadingCheckpoints && (
            <div className="flex items-center gap-2 py-6 justify-center text-sm text-muted-foreground">
              <Spinner className="size-4" />
              {t("benchmark.loadingCheckpoints")}
            </div>
          )}

          {checkpointError && (
            <div className="flex items-center gap-2 py-6 justify-center text-sm text-destructive">
              <HugeiconsIcon icon={AlertCircleIcon} className="size-4" />
              {checkpointError}
            </div>
          )}

          {!loadingCheckpoints && !checkpointError && (
            <>
              <div className="grid grid-cols-3 gap-3">
                <div className="flex flex-col gap-2">
                  <Label className="text-xs font-medium text-muted-foreground">
                    {t("benchmark.modelLabel")}
                  </Label>
                  <ModelSelector
                    models={models}
                    loraModels={loraModels}
                    externalModels={[]}
                    value={selectedModel ?? undefined}
                    activeGgufVariant={activeGgufVariant}
                    onValueChange={handleModelChange}
                    variant="muted" className="bg-accent!"
                  />
                </div>
                <div className="flex flex-col gap-2">
                  <Label className="text-xs font-medium text-muted-foreground">
                    {t("benchmark.taskLabel")}
                  </Label>
                  {loadingTasks ? (
                    <p className="text-xs text-muted-foreground">
                      {t("benchmark.loadingTasks")}
                    </p>
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
                          const t = tasks.find((t) => t.id === item);
                          return t ? t.name : item;
                        }}
                        autoHighlight={true}
                      >
                        <ComboboxInput
                          placeholder={t("benchmark.searchTaskPlaceholder")}
                          className="w-full"
                          showClear={true}
                        />
                        <ComboboxContent anchor={taskAnchorRef}>
                          <ComboboxEmpty>
                            {t("benchmark.noTasksFound")}
                          </ComboboxEmpty>
                          <ComboboxList>
                            {taskItems.map((id) => {
                              const info = tasks.find((t) => t.id === id);
                              const tl = info?.task_type;
                              const typeLabel = !tl ? null
                                : tl.includes("log") ? "log likelihood"
                                  : tl === "greedy_until" ? "generation"
                                    : tl;
                              return (
                                <ComboboxItem
                                  key={id}
                                  value={id}
                                  onPointerDown={() => { selectingTaskRef.current = true; }}
                                >
                                  <span className="truncate font-medium">
                                    {info?.name ?? id}
                                  </span>
                                  {typeLabel && (
                                    <span className="text-muted-foreground shrink-0 text-xs ml-2">
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
                </div>
                <div className="flex flex-col gap-2">
                  <Label className="text-xs font-medium text-muted-foreground">
                    {t("benchmark.batchSizeLabel")}
                  </Label>
                  <div className="flex h-9 items-center gap-3 px-3">
                    <Slider
                      value={[batchSize === 0 ? 0 : Math.round(Math.log2(batchSize)) + 1]}
                      onValueChange={([v]) => setBatchSize(v === 0 ? 0 : Math.pow(2, v - 1))}
                      min={0}
                      max={8}
                      step={1}
                      className="flex-1"
                    />
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
                      className="w-14 text-right font-mono text-xs font-medium h-7 px-1.5 [&+span]:hidden"
                    />
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-3">
                <div className="flex flex-col gap-2">
                  <Label className="text-xs font-medium text-muted-foreground">
                    {t("benchmark.outputPathLabel")}
                  </Label>
                  <Input
                    type="text"
                    value={outputPath}
                    onChange={(e) => setOutputPath(e.target.value)}
                    placeholder={t("benchmark.outputPathPlaceholder")}
                  />
                </div>
                <div className="flex flex-col gap-2">
                  <Label className="text-xs font-medium text-muted-foreground">
                    {t("benchmark.numFewshotLabel")}
                  </Label>
                  {fewshotSupported ? (
                    <div className="flex h-9 items-center gap-3 px-3">
                      <Slider
                        value={[numFewshot]}
                        onValueChange={([v]) => setNumFewshot(v)}
                        min={0}
                        max={20}
                        step={1}
                        className="flex-1"
                      />
                      <Input
                        type="number"
                        value={numFewshot}
                        onChange={(e) => setNumFewshot(Number(e.target.value))}
                        min={0}
                        max={20}
                        step={1}
                        className="w-12 text-right font-mono text-xs font-medium h-7 px-1.5 [&+span]:hidden"
                      />
                    </div>
                  ) : (
                    <div className="flex h-9 items-center px-3 text-xs text-muted-foreground/50">
                      {t("benchmark.fewshotNotSupported")}
                    </div>
                  )}
                </div>
                <div className="flex flex-col gap-2">
                  <div className="flex items-center gap-2 h-9 mt-auto">
                    <Checkbox
                      id="log-samples"
                      checked={logSamples}
                      onCheckedChange={(checked) => setLogSamples(checked === true)}
                      className="data-[state=checked]:bg-emerald-600 data-[state=checked]:border-emerald-600"
                    />
                    <Label htmlFor="log-samples" className="text-sm text-muted-foreground cursor-pointer select-none">
                      {t("benchmark.logSamplesLabel")}
                    </Label>
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-3">
                <div className="flex flex-col gap-2">
                  <Label className="text-xs font-medium text-muted-foreground">
                    {t("benchmark.maxTokensLabel")}
                  </Label>
                  <div className="flex h-9 items-center gap-3 px-3">
                    <Slider
                      value={[Math.round(Math.log2(maxTokens / 1024))]}
                      onValueChange={([v]) => setMaxTokens(1024 * Math.pow(2, v))}
                      min={0}
                      max={8}
                      step={1}
                      className="flex-1"
                    />
                    <span className="text-xs font-mono text-muted-foreground w-14 text-right tabular-nums">{maxTokens.toLocaleString()}</span>
                  </div>
                </div>
              </div>
              {isLoadingLocalModels && (
                <p className="text-[10px] text-muted-foreground">
                  {t("benchmark.scanningLocalModels")}
                </p>
              )}
              {localModelsError && (
                <p className="text-xs text-destructive">
                  {localModelsError}
                </p>
              )}

              <Separator />
              {showPanel && (
                <BenchmarkRunPanel onClose={handleClosePanel} />
              )}
              {showPanel && (
                <div ref={panelEndRef} aria-hidden="true" className="h-px w-full" />
              )}
              {!showPanel && (
                <div className="flex items-center justify-end">
                  <Button disabled={!selectedModel || modelLoading || !!loadingModel || downloadingForBenchmark} onClick={handleStartAndOpenPanel}>
                    {loadingModel ? t("benchmark.loadingModel") : downloadingForBenchmark ? t("benchmark.downloadingModel") : t("benchmark.runButton")}
                  </Button>
                </div>
              )}
            </>
          )}
        </SectionCard>

        <SectionCard
          icon={<HugeiconsIcon icon={WorkHistoryIcon} className="size-5" />}
          title={t("benchmark.historySectionTitle")}
          description={t("benchmark.historySectionDescription")}
          accent="emerald"
          className="mt-6 ring-0 shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:shadow-none"
        >
          <BenchmarkHistoryPanel />
        </SectionCard>
      </main>
    </div>
  );
}
