// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import { Alert02Icon, RepeatIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { NodeResizer, Position, useUpdateNodeInternals } from "@xyflow/react";
import { type ReactElement, useEffect, useMemo, useState } from "react";
import { MAX_NODE_WIDTH, MIN_NODE_WIDTH } from "../../constants";
import { bestRound, loopStopReason } from "../../loop/controller";
import { useRecipeStudioStore } from "../../stores/recipe-studio";
import type { RecipeNode as RecipeGraphNodeType } from "../../types";
import { NODE_HANDLE_CLASS } from "../../utils/handle-layout";
import { HANDLE_IDS } from "../../utils/handles";
import { findLoopParts, loopSetupIssues, parseLoopSettings } from "../../utils/loop-card";
import { RECIPE_STUDIO_NODE_TONES } from "../../utils/ui-tones";
import {
  BaseNode,
  BaseNodeContent,
  BaseNodeHeader,
  BaseNodeHeaderTitle,
} from "../rf-ui/base-node";
import { LabeledHandle } from "../rf-ui/labeled-handle";
import { useRecipeLoopView } from "./loop-context";
import { LoopHistoryDialog } from "./loop-history-dialog";
import { LoopScoreChart } from "./loop-score-chart";
import { describeLoopStatus } from "./loop-status";

const FEEDBACK_ISSUE = "Add the Benchmark failures seed.";

type RecipeLoopNodeProps = {
  id: string;
  data: RecipeGraphNodeType["data"];
  selected: boolean;
};

export function RecipeLoopNode({ id, data, selected }: RecipeLoopNodeProps): ReactElement | null {
  const config = useRecipeStudioStore((state) => state.configs[id]);
  const configs = useRecipeStudioStore((state) => state.configs);
  const openConfig = useRecipeStudioStore((state) => state.openConfig);
  const addLoopFeedbackSeed = useRecipeStudioStore((state) => state.addLoopFeedbackSeed);
  const loop = useRecipeLoopView();
  const updateNodeInternals = useUpdateNodeInternals();
  const [historyOpen, setHistoryOpen] = useState(false);
  const executionLocked = Boolean(data.executionLocked);

  // biome-ignore lint/correctness/useExhaustiveDependencies: re-measure handles when the graph orientation flips
  useEffect(() => {
    updateNodeInternals(id);
  }, [id, data.layoutDirection, updateNodeInternals]);

  const issues = useMemo(() => loopSetupIssues(findLoopParts(configs), configs), [configs]);

  if (config?.kind !== "loop") {
    return null;
  }

  const { settings, errors } = parseLoopSettings(config);
  const record = loop?.latest ?? null;
  const running = Boolean(loop?.running);
  const interrupted = Boolean(loop?.interrupted);
  const best = record ? bestRound(record.rounds) : null;
  const blockers = [...errors, ...issues];
  const canStart = Boolean(loop) && !running && !executionLocked && blockers.length === 0;
  const canResume =
    canStart &&
    record !== null &&
    record.status !== "target_reached" &&
    (interrupted ||
      record.status === "stopped" ||
      record.status === "error" ||
      loopStopReason(record.rounds, settings) === null);

  const chips = [
    `Target ${config.targetScore || "?"}%`,
    `Up to ${config.maxRounds || "?"} rounds`,
    config.stopOnPlateau ? "Stops on plateau" : null,
    `${config.failureShare || 0}% rows from failures`,
  ].filter((chip): chip is string => chip !== null);

  return (
    <BaseNode
      className={cn(
        "corner-squircle relative w-full min-w-0 overflow-visible rounded-xl border-border/60 shadow-sm",
        running && "border-ring-strong shadow-md",
        record?.status === "target_reached" &&
          !running &&
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
              RECIPE_STUDIO_NODE_TONES.loop,
            )}
          >
            <HugeiconsIcon icon={RepeatIcon} className="size-3.5" />
          </div>
          <div className="min-w-0">
            <BaseNodeHeaderTitle className="truncate text-sm">{config.name}</BaseNodeHeaderTitle>
            <p className="truncate text-ui-11 text-muted-foreground">{data.subtype}</p>
          </div>
        </div>
        <Button
          type="button"
          size="xs"
          variant="ghost"
          className="nodrag"
          disabled={executionLocked || running}
          onClick={(event) => {
            event.preventDefault();
            event.stopPropagation();
            openConfig(id);
          }}
        >
          Configure
        </Button>
      </BaseNodeHeader>

      <BaseNodeContent className="gap-2.5 px-3 py-2.5">
        <div className="flex flex-wrap gap-1.5">
          {chips.map((chip) => (
            <Badge key={chip} variant="secondary" className="corner-squircle text-ui-11">
              {chip}
            </Badge>
          ))}
        </div>

        {record && record.rounds.length > 0 && (
          <div className="corner-squircle space-y-1 rounded-lg border border-border/60 px-2.5 pt-2 pb-1.5">
            <div className="flex items-baseline justify-between gap-2">
              <span className="text-ui-11 text-muted-foreground">Benchmark score per round</span>
              {best?.result && (
                <span className="text-sm font-semibold tabular-nums">
                  {Math.round(best.result.score * 100)}%
                  <span className="ml-1 text-ui-11 font-normal text-muted-foreground">
                    best, round {best.index}
                  </span>
                </span>
              )}
            </div>
            <LoopScoreChart
              rounds={record.rounds}
              target={record.settings.targetScore}
              maxRounds={record.settings.maxRounds}
            />
          </div>
        )}

        {record && (
          <p
            className={cn(
              "flex items-center gap-1.5 text-xs",
              record.status === "error" && !running ? "text-destructive" : "text-muted-foreground",
            )}
          >
            {running && <Spinner className="size-3.5" />}
            {describeLoopStatus(record, interrupted)}
          </p>
        )}

        {!running && blockers.length > 0 && (
          <div className="corner-squircle space-y-1.5 rounded-lg border border-amber-400/50 bg-amber-500/5 px-3 py-2">
            {blockers.slice(0, 3).map((issue) => (
              <p key={issue} className="flex items-start gap-1.5 text-xs text-foreground">
                <HugeiconsIcon
                  icon={Alert02Icon}
                  className="mt-0.5 size-3.5 shrink-0 text-amber-600 dark:text-amber-400"
                />
                <span>{issue}</span>
              </p>
            ))}
            {issues.includes(FEEDBACK_ISSUE) && (
              <Button
                type="button"
                size="xs"
                variant="outline"
                className="nodrag"
                disabled={executionLocked}
                onClick={(event) => {
                  event.preventDefault();
                  event.stopPropagation();
                  addLoopFeedbackSeed();
                }}
              >
                Add failures seed
              </Button>
            )}
          </div>
        )}

        <div className="flex gap-2">
          {running ? (
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="nodrag flex-1"
              onClick={(event) => {
                event.preventDefault();
                event.stopPropagation();
                loop?.stop();
              }}
            >
              Stop loop
            </Button>
          ) : (
            <Button
              type="button"
              size="sm"
              className="nodrag flex-1"
              disabled={!canStart}
              onClick={(event) => {
                event.preventDefault();
                event.stopPropagation();
                if (canResume) {
                  loop?.resume(settings);
                } else {
                  loop?.start(settings);
                }
              }}
            >
              <HugeiconsIcon icon={RepeatIcon} className="size-4" />
              {canResume ? "Resume loop" : record ? "Start new loop" : "Start loop"}
            </Button>
          )}
          {record && record.rounds.length > 0 && (
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="nodrag"
              onClick={(event) => {
                event.preventDefault();
                event.stopPropagation();
                setHistoryOpen(true);
              }}
            >
              History
            </Button>
          )}
        </div>
      </BaseNodeContent>

      {record && (
        <LoopHistoryDialog record={record} open={historyOpen} onOpenChange={setHistoryOpen} />
      )}

      <LabeledHandle
        id={HANDLE_IDS.dataIn}
        title="Benchmark score"
        type="target"
        position={Position.Left}
        className="absolute inset-0 pointer-events-none"
        labelClassName="sr-only"
        handleClassName={NODE_HANDLE_CLASS}
      />
      <LabeledHandle
        id={HANDLE_IDS.dataInTop}
        title="Benchmark score"
        type="target"
        position={Position.Top}
        className="absolute inset-0 pointer-events-none"
        labelClassName="sr-only"
        handleClassName={NODE_HANDLE_CLASS}
      />
    </BaseNode>
  );
}
