// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import type { ReactElement } from "react";
import { bestRound } from "../../loop/controller";
import type { LoopRecord, LoopRound } from "../../loop/types";
import { describeLoopStatus, roundPhaseLabel } from "./loop-status";
import { LoopScoreChart } from "./loop-score-chart";

function leaf(path: string | null): string {
  if (!path) {
    return "base model";
  }
  const parts = path.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] ?? path;
}

function percent(score: number | null | undefined): string {
  return typeof score === "number" ? `${Math.round(score * 100)}%` : "n/a";
}

function RoundRow({
  round,
  previous,
  best,
}: {
  round: LoopRound;
  previous: LoopRound | null;
  best: boolean;
}): ReactElement {
  const dataset = round.dataset;
  const score = round.result?.score ?? null;
  const before = previous?.result?.score ?? null;
  const delta = score !== null && before !== null ? Math.round((score - before) * 100) : null;
  return (
    <div className={cn("space-y-1.5 px-3 py-2.5", best && "bg-primary/5")}>
      <div className="grid grid-cols-[3.5rem_1fr_1fr_5rem] items-baseline gap-3 text-sm">
        <span className="font-medium">Round {round.index}</span>
        <span className="text-muted-foreground">
          {dataset ? (
            <>
              {dataset.rows} rows
              {dataset.fromFailures > 0 && (
                <span className="text-foreground"> · {dataset.fromFailures} from failures</span>
              )}
            </>
          ) : (
            "..."
          )}
        </span>
        <span className="truncate text-muted-foreground" title={round.adapter?.outputDir}>
          {round.adapter
            ? `from ${round.adapter.continuedFrom ? leaf(round.adapter.continuedFrom) : "base model"}`
            : round.status === "error"
              ? ""
              : roundPhaseLabel(round.status)}
        </span>
        <span className="text-right font-semibold tabular-nums">
          {round.status === "done" ? percent(score) : round.status === "error" ? "failed" : "..."}
          {delta !== null && delta !== 0 && (
            <span
              className={cn(
                "ml-1 text-xs font-normal",
                delta > 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400",
              )}
            >
              {delta > 0 ? "+" : ""}
              {delta}
            </span>
          )}
        </span>
      </div>
      {round.result && (
        <p className="text-xs text-muted-foreground">
          {round.result.passed}/{round.result.scored} held-out rows passed
          {round.result.baseline !== null ? ` · base model ${percent(round.result.baseline)}` : ""}
          {round.result.failedTotal > 0 ? ` · ${round.result.failedTotal} failed, sent to the next round` : ""}
        </p>
      )}
      {round.error && <p className="text-xs text-destructive">{round.error}</p>}
      {dataset && dataset.examples.length > 0 && (
        <ul className="space-y-1">
          {dataset.examples.map((example, i) => (
            <li
              // biome-ignore lint/suspicious/noArrayIndexKey: fixed list
              key={i}
              className="corner-squircle grid grid-cols-[auto_1fr] gap-x-2 rounded-md bg-muted/40 px-2 py-1.5 text-xs"
            >
              <span className="text-muted-foreground">Failed</span>
              <span className="line-clamp-1">{example.failed}</span>
              <span className="text-muted-foreground">Wrote</span>
              <span className="line-clamp-2">{example.written}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function LoopHistoryDialog({
  record,
  open,
  onOpenChange,
}: {
  record: LoopRecord;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}): ReactElement {
  const best = bestRound(record.rounds);
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Loop history</DialogTitle>
          <DialogDescription>
            {describeLoopStatus(record, false)}
            {record.holdout
              ? ` Every round is scored on the same ${record.holdout.rows} held-out rows from round 1.`
              : ""}
          </DialogDescription>
        </DialogHeader>
        <div className="corner-squircle rounded-lg border border-border/60 px-3 pt-2 pb-1">
          <LoopScoreChart
            rounds={record.rounds}
            target={record.settings.targetScore}
            maxRounds={record.settings.maxRounds}
            height={120}
            labels={true}
          />
        </div>
        <div className="corner-squircle max-h-[50vh] divide-y divide-border/60 overflow-y-auto rounded-lg border border-border/60">
          {record.rounds.map((round, i) => (
            <RoundRow
              key={round.index}
              round={round}
              previous={record.rounds[i - 1] ?? null}
              best={best?.index === round.index}
            />
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}
