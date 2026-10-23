// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactElement } from "react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { RecipeExecutionRecord } from "../../execution-types";
import {
  executionLabel,
  formatVersionLabel,
  isExecutionInProgress,
  normalizeRunName,
} from "../../executions/execution-helpers";
import {
  formatStatus,
  formatTimestamp,
  statusRightBorder,
  statusTone,
} from "./executions-view-helpers";

type ExecutionSidebarProps = {
  executions: RecipeExecutionRecord[];
  selectedExecutionId: string | null;
  onSelectExecution: (id: string) => void;
};

export function ExecutionSidebar({
  executions,
  selectedExecutionId,
  onSelectExecution,
}: ExecutionSidebarProps): ReactElement {
  return (
    <aside className="w-72 shrink-0 overflow-y-auto border-r border-border/60 bg-card/20 max-md:max-h-48 max-md:w-full max-md:border-r-0 max-md:border-b">
      <div className="flex items-center justify-between  border-border/60 px-3 py-2">
        <p className="text-xs font-semibold uppercase text-muted-foreground">
          Runs
        </p>
      </div>
      <div className="h-[calc(100%-45px)] space-y-2 overflow-auto p-2">
        {executions.length === 0 ? (
          <div className="rounded-lg border border-dashed border-border/60 p-3 text-xs text-muted-foreground">
            No runs yet.
          </div>
        ) : (
          executions.map((execution) => {
            const title =
              execution.kind === "full"
                ? (normalizeRunName(execution.run_name) ??
                  executionLabel(execution.kind))
                : executionLabel(execution.kind);
            const version = formatVersionLabel(execution);
            return (
              <button
                key={execution.id}
                type="button"
                onClick={() => onSelectExecution(execution.id)}
                className={cn(
                  "w-full rounded-lg corner-squircle border border-r-2 border-border/60 bg-card/60 p-3 text-left transition-colors",
                  selectedExecutionId === execution.id
                    ? "border-ring-strong bg-primary/[0.045]"
                    : "hover:bg-muted/25",
                  statusRightBorder(execution.status),
                )}
              >
                <div className="mb-2 flex items-center justify-between gap-2">
                  <p className="flex min-w-0 items-center gap-1.5 text-sm font-medium">
                    {version && (
                      <span className="shrink-0 font-mono text-xs text-muted-foreground">
                        {version}
                      </span>
                    )}
                    <span className="truncate">{title}</span>
                  </p>
                  <Badge
                    variant="outline"
                    className={cn("capitalize text-ui-11", statusTone(execution.status))}
                  >
                    {formatStatus(execution.status)}
                  </Badge>
                </div>
                <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <span>{execution.rows} rows</span>
                  {execution.status === "completed" &&
                    execution.quality?.verdict === "warn" && (
                      <Badge
                        variant="outline"
                        className="border-amber-500/30 text-ui-11 text-amber-700 dark:text-amber-300"
                      >
                        Quality warning
                      </Badge>
                    )}
                </p>
                {isExecutionInProgress(execution.status) &&
                  typeof execution.batch?.total === "number" &&
                  execution.batch.total > 1 && (
                    <p className="text-xs text-muted-foreground">
                      Batch {execution.batch.idx ?? "--"}/{execution.batch.total}
                    </p>
                  )}
                <p className="text-xs text-muted-foreground">
                  {formatTimestamp(execution.createdAt)}
                </p>
              </button>
            );
          })
        )}
      </div>
    </aside>
  );
}
