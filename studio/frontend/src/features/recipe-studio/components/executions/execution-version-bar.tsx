// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactElement } from "react";
import { Copy01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toastError, toastSuccess } from "@/shared/toast";
import type { RecipeExecutionRecord } from "../../execution-types";
import {
  findSameConfigVersion,
  formatVersionLabel,
} from "../../executions/execution-helpers";
import { formatMetricValue } from "./executions-view-helpers";

export function ExecutionVersionBar({
  execution,
  executions,
}: {
  execution: RecipeExecutionRecord;
  executions: RecipeExecutionRecord[];
}): ReactElement | null {
  const label = formatVersionLabel(execution);
  if (!label) {
    return null;
  }
  const sameAs = findSameConfigVersion(execution, executions);
  const snapshot = execution.config_snapshot;

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-xl border border-border/60 bg-card/40 px-3 py-2 text-xs">
      <Badge variant="outline" className="font-mono text-ui-11">
        {label}
      </Badge>
      <span className="text-muted-foreground">
        Rows <span className="font-semibold text-foreground">{formatMetricValue(execution.rows)}</span>
      </span>
      <span className="text-muted-foreground">
        Seed{" "}
        <span className="font-mono font-semibold text-foreground">
          {execution.seed ?? "--"}
        </span>
      </span>
      {snapshot && (
        <span className="text-muted-foreground">
          {sameAs ? `Same recipe as v${sameAs}` : "Recipe snapshot saved"}
        </span>
      )}
      {snapshot && (
        <Button
          type="button"
          size="sm"
          variant="ghost"
          className="ml-auto h-7 px-2 text-xs"
          onClick={() => {
            copyToClipboard(JSON.stringify(snapshot, null, 2)).then((ok) => {
              if (ok) {
                toastSuccess(`Copied the ${label} config`);
              } else {
                toastError("Copy failed", "The clipboard is not available.");
              }
            });
          }}
        >
          <HugeiconsIcon icon={Copy01Icon} className="size-3.5" />
          Copy config
        </Button>
      )}
    </div>
  );
}
