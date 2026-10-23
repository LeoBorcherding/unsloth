// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Spinner } from "@/components/ui/spinner";
import {
  isTrainingStartPending,
  useTrainingActions,
  useTrainingRuntimeLifecycle,
  useTrainingRuntimeStore,
} from "@/features/training";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  Database02Icon,
  Rocket01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { NodeResizer, Position, useUpdateNodeInternals } from "@xyflow/react";
import { type ReactElement, useEffect, useMemo, useState } from "react";
import { MAX_NODE_WIDTH, MIN_NODE_WIDTH } from "../constants";
import { useRecipeExecutionsStore } from "../stores/recipe-executions";
import { useRecipeStudioStore } from "../stores/recipe-studio";
import type { RecipeNode as RecipeGraphNodeType, TrainCardRun } from "../types";
import { NODE_HANDLE_CLASS } from "../utils/handle-layout";
import { HANDLE_IDS } from "../utils/handles";
import {
  describeTrainPhase,
  formatHoldoutSummary,
  isTrainRunActive,
  listTrainableColumns,
  parseTrainCardSettings,
  pickTrainSourceExecution,
} from "../utils/train-card";
import { launchTrainCardRun } from "../utils/train-card-launch";
import { RECIPE_STUDIO_NODE_TONES } from "../utils/ui-tones";
import {
  BaseNode,
  BaseNodeContent,
  BaseNodeHeader,
  BaseNodeHeaderTitle,
} from "./rf-ui/base-node";
import { LabeledHandle } from "./rf-ui/labeled-handle";

/** The Train page keeps the run's status in sync; mounted here while a Train card is on the canvas. */
export function TrainingRuntimeSync(): null {
  useTrainingRuntimeLifecycle();
  return null;
}

type RecipeTrainNodeProps = {
  id: string;
  data: RecipeGraphNodeType["data"];
  selected: boolean;
};

function StatusRow({
  tone,
  icon,
  title,
  detail,
}: {
  tone: "ready" | "waiting" | "error";
  icon: typeof Rocket01Icon;
  title: string;
  detail?: string;
}): ReactElement {
  return (
    <div
      className={cn(
        "corner-squircle flex items-start gap-2 rounded-lg border px-3 py-2",
        tone === "ready" && "border-emerald-500/40 bg-emerald-500/5",
        tone === "waiting" && "border-amber-400/50 bg-amber-500/5",
        tone === "error" && "border-rose-500/40 bg-rose-500/5",
      )}
    >
      <HugeiconsIcon
        icon={icon}
        className={cn(
          "mt-0.5 size-4 shrink-0",
          tone === "ready" && "text-emerald-600 dark:text-emerald-400",
          tone === "waiting" && "text-amber-600 dark:text-amber-400",
          tone === "error" && "text-rose-600 dark:text-rose-400",
        )}
      />
      <div className="min-w-0">
        <p className="break-words text-xs font-medium text-foreground">
          {title}
        </p>
        {detail && (
          <p className="break-all text-ui-11 text-muted-foreground">{detail}</p>
        )}
      </div>
    </div>
  );
}

export function RecipeTrainNode({
  id,
  data,
  selected,
}: RecipeTrainNodeProps): ReactElement | null {
  const config = useRecipeStudioStore((state) => state.configs[id]);
  const configs = useRecipeStudioStore((state) => state.configs);
  const openConfig = useRecipeStudioStore((state) => state.openConfig);
  const updateConfig = useRecipeStudioStore((state) => state.updateConfig);
  const executions = useRecipeExecutionsStore((state) => state.executions);
  const { startTrainingRun } = useTrainingActions();
  const runtimeBusy = useTrainingRuntimeStore(isTrainingStartPending);
  const runtimeJobId = useTrainingRuntimeStore((state) => state.jobId);
  const runtimePhase = useTrainingRuntimeStore((state) => state.phase);
  const runtimeError = useTrainingRuntimeStore((state) => state.error);
  const outputDir = useTrainingRuntimeStore((state) => state.outputDir);
  const currentStep = useTrainingRuntimeStore((state) => state.currentStep);
  const totalSteps = useTrainingRuntimeStore((state) => state.totalSteps);
  const currentLoss = useTrainingRuntimeStore((state) => state.currentLoss);
  const navigate = useNavigate();
  const updateNodeInternals = useUpdateNodeInternals();
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const executionLocked = Boolean(data.executionLocked);

  // biome-ignore lint/correctness/useExhaustiveDependencies: re-measure handles when the graph orientation flips
  useEffect(() => {
    updateNodeInternals(id);
  }, [id, data.layoutDirection, updateNodeInternals]);

  const source = useMemo(() => pickTrainSourceExecution(executions), [executions]);
  const columns = useMemo(() => listTrainableColumns(configs), [configs]);
  const lastRun = config?.kind === "train" ? (config.lastRun ?? null) : null;
  const tracksRuntime = Boolean(
    lastRun?.jobId && runtimeJobId && lastRun.jobId === runtimeJobId,
  );

  useEffect(() => {
    if (!(tracksRuntime && lastRun) || runtimePhase === "idle") {
      return;
    }
    const nextOutputDir = outputDir ?? lastRun.outputDir;
    if (lastRun.phase === runtimePhase && lastRun.outputDir === nextOutputDir) {
      return;
    }
    updateConfig(id, {
      lastRun: { ...lastRun, phase: runtimePhase, outputDir: nextOutputDir },
    });
  }, [id, lastRun, outputDir, runtimePhase, tracksRuntime, updateConfig]);

  if (config?.kind !== "train") {
    return null;
  }

  const { settings, errors } = parseTrainCardSettings(config, columns);
  const active = launching || (tracksRuntime && isTrainRunActive(lastRun));
  const canRun =
    !(executionLocked || active || runtimeBusy) &&
    errors.length === 0 &&
    source !== null;

  const onRun = async () => {
    if (!(canRun && source)) {
      return;
    }
    setLaunching(true);
    setLaunchError(null);
    try {
      const run: TrainCardRun = await launchTrainCardRun(
        settings,
        source,
        startTrainingRun,
      );
      updateConfig(id, { lastRun: run });
    } catch (error) {
      setLaunchError(
        error instanceof Error ? error.message : "Training did not start.",
      );
    } finally {
      setLaunching(false);
    }
  };

  const stats = [
    { label: "Method", value: config.method === "qlora" ? "QLoRA" : "LoRA" },
    { label: "Steps", value: config.maxSteps || "?" },
    { label: "Rank", value: config.loraRank || "?" },
    { label: "LR", value: config.learningRate || "?" },
    { label: "Hold out", value: `${config.holdoutPercent || 0}%` },
  ];
  const sourceLabel = source
    ? `${source.run_name?.trim() || "Full run"} · ${source.rows} rows`
    : "Run the full recipe first";
  const showProgress = active && !launching;
  const progressValue =
    totalSteps > 0 ? Math.min(100, (currentStep / totalSteps) * 100) : 0;
  const runFailed =
    lastRun?.phase === "error" ? (tracksRuntime ? runtimeError : null) : null;
  let buttonLabel = "Train on this dataset";
  if (launching) {
    buttonLabel = "Preparing...";
  } else if (active) {
    buttonLabel = "Training...";
  } else if (lastRun?.phase === "completed") {
    buttonLabel = "Train again";
  }

  return (
    <BaseNode
      className={cn(
        "corner-squircle relative w-full min-w-0 overflow-visible rounded-xl border-border/60 shadow-sm",
        active && "border-ring-strong shadow-md",
        lastRun?.phase === "completed" &&
          !active &&
          "border-emerald-500/60 ring-1 ring-emerald-500/20",
      )}
    >
      <NodeResizer
        isVisible={selected}
        minWidth={MIN_NODE_WIDTH}
        minHeight={160}
        maxWidth={MAX_NODE_WIDTH}
        maxHeight={720}
        color="var(--primary)"
        lineClassName="!border-transparent !shadow-none"
        lineStyle={{ opacity: 0 }}
        handleClassName="!h-3 !w-3 !border-transparent !bg-transparent"
        handleStyle={{ opacity: 0 }}
      />
      <BaseNodeHeader className="border-b border-border/50 px-3 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <div
            className={cn(
              "corner-squircle flex size-7 items-center justify-center rounded-sm border",
              RECIPE_STUDIO_NODE_TONES.train,
            )}
          >
            <HugeiconsIcon icon={Rocket01Icon} className="size-3.5" />
          </div>
          <div className="min-w-0">
            <BaseNodeHeaderTitle className="truncate text-sm">
              {config.name}
            </BaseNodeHeaderTitle>
            <p className="truncate text-ui-11 text-muted-foreground">
              {data.subtype} · {settings.baseModel || "No base model yet"}
            </p>
          </div>
        </div>
        <Button
          type="button"
          size="xs"
          variant="ghost"
          className="nodrag"
          disabled={executionLocked}
          onClick={(event) => {
            event.preventDefault();
            event.stopPropagation();
            openConfig(id);
          }}
        >
          Configure
        </Button>
      </BaseNodeHeader>

      <BaseNodeContent
        className={cn(
          "gap-2 px-3 py-2",
          executionLocked && "pointer-events-none opacity-85",
        )}
      >
        <div className="flex flex-wrap gap-1.5">
          {stats.map((stat) => (
            <Badge
              key={stat.label}
              variant="secondary"
              className="corner-squircle font-mono text-ui-11"
            >
              <span className="text-muted-foreground">{stat.label}</span>
              <span className="ml-1 text-foreground">{stat.value}</span>
            </Badge>
          ))}
        </div>

        <StatusRow
          tone={source ? "ready" : "waiting"}
          icon={source ? Tick02Icon : Database02Icon}
          title={sourceLabel}
          detail={
            settings.promptColumn && settings.responseColumn
              ? `${settings.promptColumn} → ${settings.responseColumn}`
              : "Pick prompt and response columns in Configure"
          }
        />

        {showProgress && (
          <div className="grid gap-1.5">
            <div className="flex items-center justify-between text-ui-11 text-muted-foreground">
              <span>{describeTrainPhase(lastRun?.phase ?? "")}</span>
              <span className="font-mono">
                {totalSteps > 0 ? `step ${currentStep}/${totalSteps}` : ""}
                {currentLoss ? ` · loss ${currentLoss.toFixed(3)}` : ""}
              </span>
            </div>
            <Progress
              value={progressValue}
              indeterminate={totalSteps === 0}
              className="h-1.5"
            />
          </div>
        )}

        {lastRun?.phase === "completed" && (
          <StatusRow
            tone="ready"
            icon={Tick02Icon}
            title={`Adapter ready · ${formatHoldoutSummary(lastRun)}`}
            detail={lastRun.outputDir ?? lastRun.jobId ?? undefined}
          />
        )}
        {lastRun?.phase === "stopped" && !active && (
          <StatusRow tone="waiting" icon={Alert02Icon} title="Run stopped" />
        )}
        {(launchError || lastRun?.phase === "error") && (
          <StatusRow
            tone="error"
            icon={Alert02Icon}
            title={launchError ? "Training did not start" : "Training failed"}
            detail={launchError ?? runFailed ?? undefined}
          />
        )}
        {errors.length > 0 && !active && (
          <p className="text-ui-11 text-muted-foreground">{errors[0]}</p>
        )}

        <div className="flex gap-2">
          <Button
            type="button"
            size="sm"
            className="nodrag flex-1"
            disabled={!canRun}
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              onRun().catch(() => undefined);
            }}
          >
            {active ? (
              <Spinner className="size-4" />
            ) : (
              <HugeiconsIcon icon={Rocket01Icon} className="size-4" />
            )}
            {buttonLabel}
          </Button>
          {lastRun?.jobId && (
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="nodrag"
              onClick={(event) => {
                event.preventDefault();
                event.stopPropagation();
                navigate({ to: "/studio" }).catch(() => undefined);
              }}
            >
              Open run
            </Button>
          )}
        </div>
      </BaseNodeContent>

      <LabeledHandle
        id={HANDLE_IDS.dataIn}
        title="Dataset"
        type="target"
        position={Position.Left}
        className="absolute inset-0 pointer-events-none"
        labelClassName="sr-only"
        handleClassName={NODE_HANDLE_CLASS}
      />
      <LabeledHandle
        id={HANDLE_IDS.dataInTop}
        title="Dataset"
        type="target"
        position={Position.Top}
        className="absolute inset-0 pointer-events-none"
        labelClassName="sr-only"
        handleClassName={NODE_HANDLE_CLASS}
      />
      <LabeledHandle
        id={HANDLE_IDS.dataOut}
        title="Trained adapter"
        type="source"
        position={Position.Right}
        className="absolute inset-0 pointer-events-none"
        labelClassName="sr-only"
        handleClassName={NODE_HANDLE_CLASS}
      />
      <LabeledHandle
        id={HANDLE_IDS.dataOutBottom}
        title="Trained adapter"
        type="source"
        position={Position.Bottom}
        className="absolute inset-0 pointer-events-none"
        labelClassName="sr-only"
        handleClassName={NODE_HANDLE_CLASS}
      />
    </BaseNode>
  );
}
