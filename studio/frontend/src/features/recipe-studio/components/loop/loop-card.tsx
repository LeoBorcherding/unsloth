// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import {
  ChevronDownStandardIcon,
  ChevronUpStandardIcon,
} from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import { RepeatIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useState } from "react";
import type { RecipeLoopView } from "../../hooks/use-recipe-loop";
import { bestIteration } from "../../loop/controller";
import type {
  LoopConfig,
  LoopIteration,
  LoopRecord,
} from "../../loop/types";

type NumberKey = {
  [K in keyof LoopConfig]: LoopConfig[K] extends number ? K : never;
}[keyof LoopConfig];

const NUMBER_FIELDS: Array<{
  key: NumberKey;
  label: string;
  step: number;
  min: number;
}> = [
  { key: "target", label: "Target eval loss", step: 0.05, min: 0 },
  { key: "maxIterations", label: "Max iterations", step: 1, min: 1 },
  { key: "plateauPatience", label: "Plateau after (0 off)", step: 1, min: 0 },
  { key: "minImprovement", label: "Min improvement", step: 0.01, min: 0 },
  { key: "baseRows", label: "Rows, first iteration", step: 50, min: 1 },
  { key: "rowsPerIteration", label: "Rows added each time", step: 50, min: 1 },
  { key: "holdoutRows", label: "Held-out rows", step: 16, min: 0 },
  { key: "epochs", label: "Epochs", step: 1, min: 1 },
];

function formatScore(value: number | null): string {
  return value === null ? "--" : value.toFixed(4);
}

function iterationState(it: LoopIteration): string {
  if (it.status === "done") return formatScore(it.score);
  if (it.status === "error") return "failed";
  return it.status;
}

function statusLine(
  record: LoopRecord | null,
  interrupted: boolean,
): string {
  if (!record) {
    return "Generate, train and score, repeated until the target or a stop rule.";
  }
  const best = bestIteration(record);
  const bestText = best
    ? `best ${record.metric.label.toLowerCase()} ${formatScore(best.score)} (iteration ${best.index})`
    : "no score yet";
  if (interrupted) {
    return `Interrupted at iteration ${record.iterations.length}, ${bestText}. Resume picks up from the last finished iteration.`;
  }
  const current = record.iterations[record.iterations.length - 1];
  switch (record.status) {
    case "running":
      return current
        ? `Iteration ${current.index} of ${record.config.maxIterations}: ${current.status}. ${bestText}.`
        : "Generating the held-out rows.";
    case "target_reached":
      return `Target ${record.config.target} reached, ${bestText}.`;
    case "plateau":
      return `Stopped on a plateau: no gain of ${record.config.minImprovement} in ${record.config.plateauPatience} iterations, ${bestText}.`;
    case "max_iterations":
      return `Ran ${record.config.maxIterations} iterations, ${bestText}.`;
    case "stopped":
      return `Stopped, ${bestText}.`;
    case "error":
      return `Failed: ${record.error ?? "unknown error"}`;
  }
}

export function LoopCard({
  view,
  disabled,
}: {
  view: RecipeLoopView;
  disabled: boolean;
}): ReactElement {
  const [open, setOpen] = useState(true);
  const { latest, interrupted, running, config, setConfig } = view;
  const best = latest ? bestIteration(latest) : null;
  const canResume =
    !running &&
    latest !== null &&
    (interrupted || latest.status === "stopped" || latest.status === "error" ||
      latest.status === "max_iterations" || latest.status === "plateau");

  return (
    <div
      className="w-[clamp(18rem,30vw,24rem)] max-w-[calc(100vw-1rem)] rounded-xl border bg-card/96 shadow-sm backdrop-blur-sm"
      data-testid="recipe-loop-card"
    >
      <div className="flex items-center justify-between gap-2 px-3 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <HugeiconsIcon icon={RepeatIcon} className="size-3.5 text-primary" />
          <p className="text-xs font-semibold text-foreground">Loop</p>
          {running && <Spinner className="size-3.5 text-muted-foreground" />}
        </div>
        <button
          type="button"
          onClick={() => setOpen(!open)}
          className="btn-fill inline-flex size-7 items-center justify-center rounded-full text-muted-foreground"
          aria-label={open ? "Collapse loop" : "Expand loop"}
        >
          <HugeiconsIcon
            icon={open ? ChevronDownStandardIcon : ChevronUpStandardIcon}
            className="size-3.5"
          />
        </button>
      </div>
      {open && (
        <div className="space-y-3 px-3 pb-3">
          <p className="text-ui-11 text-muted-foreground" aria-live="polite">
            {statusLine(latest, interrupted)}
          </p>
          <div className="grid grid-cols-2 gap-x-2 gap-y-1.5">
            {NUMBER_FIELDS.map((field) => (
              <label key={field.key} className="space-y-0.5">
                <span className="text-ui-11 text-muted-foreground">
                  {field.label}
                </span>
                <Input
                  type="number"
                  className="h-7 text-xs"
                  min={field.min}
                  step={field.step}
                  disabled={running}
                  value={config[field.key]}
                  onChange={(event) => {
                    const value = Number(event.target.value);
                    if (Number.isFinite(value) && value >= field.min) {
                      setConfig({ [field.key]: value });
                    }
                  }}
                />
              </label>
            ))}
            <label className="col-span-2 space-y-0.5">
              <span className="text-ui-11 text-muted-foreground">
                Model (LoRA, eval loss on the held-out rows)
              </span>
              <Input
                className="h-7 text-xs"
                disabled={running}
                value={config.modelName}
                onChange={(event) =>
                  setConfig({ modelName: event.target.value })
                }
              />
            </label>
          </div>
          <div className="flex gap-2">
            {running ? (
              <Button size="sm" variant="outline" onClick={view.stop}>
                Stop loop
              </Button>
            ) : (
              <Button
                size="sm"
                disabled={disabled || !config.modelName.trim()}
                onClick={view.start}
              >
                Run loop
              </Button>
            )}
            {canResume && (
              <Button
                size="sm"
                variant="outline"
                disabled={disabled}
                onClick={view.resume}
              >
                Resume
              </Button>
            )}
          </div>
          {latest && latest.iterations.length > 0 && (
            <table className="w-full text-ui-11" data-testid="recipe-loop-history">
              <thead className="text-muted-foreground">
                <tr className="text-left">
                  <th className="font-medium">#</th>
                  <th className="font-medium">Dataset</th>
                  <th className="font-medium">Rows</th>
                  <th className="font-medium">Adapter</th>
                  <th className="text-right font-medium">
                    {latest.metric.label}
                  </th>
                </tr>
              </thead>
              <tbody>
                {latest.iterations.map((it) => (
                  <tr
                    key={it.index}
                    className={cn(
                      "border-t border-border/60",
                      best?.index === it.index &&
                        "font-semibold text-emerald-700 dark:text-emerald-300",
                    )}
                    title={it.error ?? undefined}
                  >
                    <td>{it.index}</td>
                    <td className="max-w-28 truncate" title={it.dataset?.version}>
                      {it.dataset?.version ?? "--"}
                    </td>
                    <td>{it.dataset?.rows ?? "--"}</td>
                    <td className="max-w-24 truncate" title={it.run?.adapter ?? it.run?.runId}>
                      {it.run?.adapter ?? it.run?.runId ?? "--"}
                    </td>
                    <td className="text-right tabular-nums">
                      {iterationState(it)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {latest?.holdout && (
            <p className="truncate text-ui-11 text-muted-foreground">
              Scored on {latest.holdout.rows ?? "--"} held-out rows from{" "}
              {latest.holdout.version}, generated once for every iteration.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
