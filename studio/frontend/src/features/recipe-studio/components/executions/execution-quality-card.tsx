// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactElement } from "react";
import {
  Alert02Icon,
  CheckmarkCircle02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import type { RecipeExecutionQuality } from "../../execution-types";
import { formatMetricValue, summarizeQuality } from "./executions-view-helpers";

function Stat({
  label,
  value,
  flagged,
}: {
  label: string;
  value: string;
  flagged: boolean;
}): ReactElement {
  return (
    <div className="rounded-lg border border-border/60 bg-card/50 px-3 py-2">
      <p className="text-ui-11 text-muted-foreground">{label}</p>
      <p
        className={cn(
          "text-sm font-semibold",
          flagged && "text-amber-700 dark:text-amber-300",
        )}
      >
        {value}
      </p>
    </div>
  );
}

export function ExecutionQualityCard({
  quality,
}: {
  quality: RecipeExecutionQuality;
}): ReactElement {
  const summary = summarizeQuality(quality);
  const warn = quality.verdict === "warn";

  return (
    <div className="space-y-3 rounded-2xl border border-border/60 bg-card/55 p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <HugeiconsIcon
            icon={warn ? Alert02Icon : CheckmarkCircle02Icon}
            className={cn(
              "size-4",
              warn
                ? "text-amber-700 dark:text-amber-300"
                : "text-emerald-700 dark:text-emerald-300",
            )}
          />
          <p className="text-sm font-semibold">Quality pass</p>
        </div>
        <Badge
          variant="outline"
          className={cn(
            "text-ui-11",
            warn
              ? "border-amber-500/30 text-amber-700 dark:text-amber-300"
              : "border-emerald-500/30 text-emerald-700 dark:text-emerald-300",
          )}
        >
          {warn ? "Check before training" : "Looks clean"}
        </Badge>
      </div>

      {warn && (
        <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-3">
          <p className="text-sm font-semibold text-amber-800 dark:text-amber-200">
            This dataset looks risky to train on
          </p>
          <ul className="mt-1 list-disc space-y-0.5 pl-4 text-xs text-amber-800 dark:text-amber-200">
            {quality.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="grid gap-2 sm:grid-cols-3 lg:grid-cols-5">
        <Stat
          label="Rows checked"
          value={
            quality.truncated
              ? `${formatMetricValue(quality.rows_checked)} of ${formatMetricValue(quality.rows_total)}`
              : formatMetricValue(quality.rows_checked)
          }
          flagged={false}
        />
        <Stat
          label="Exact duplicates"
          value={formatMetricValue(quality.exact_duplicates)}
          flagged={quality.exact_duplicates > 0}
        />
        <Stat
          label={quality.near_duplicates_checked ? "Near duplicates" : "Near duplicates (text match)"}
          value={formatMetricValue(quality.near_duplicates)}
          flagged={quality.near_duplicates > 0}
        />
        <Stat
          label="Format failures"
          value={formatMetricValue(summary.formatFailures)}
          flagged={summary.formatFailures > 0}
        />
        <Stat
          label="Empty values"
          value={formatMetricValue(summary.emptyValues)}
          flagged={summary.emptyValues > 0}
        />
      </div>

      {quality.columns.length > 0 && (
        <div className="overflow-hidden rounded-md border border-border/60 bg-card/50">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Column</TableHead>
                <TableHead>Format check</TableHead>
                <TableHead className="text-right">Empty</TableHead>
                <TableHead className="text-right">Format fails</TableHead>
                <TableHead className="text-right">Chars (min / median / max)</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {quality.columns.map((column) => (
                <TableRow key={column.name}>
                  <TableCell className="font-medium">{column.name}</TableCell>
                  <TableCell className="text-muted-foreground">
                    {column.format_check === "json" ? "Valid JSON" : "No leaked chat tokens"}
                  </TableCell>
                  <TableCell className="text-right">{formatMetricValue(column.empty)}</TableCell>
                  <TableCell className="text-right">
                    {formatMetricValue(column.format_failures)}
                  </TableCell>
                  <TableCell className="text-right">
                    {formatMetricValue(column.min_chars)} / {formatMetricValue(column.median_chars)} /{" "}
                    {formatMetricValue(column.max_chars)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      {quality.validators.length > 0 && (
        <div className="space-y-1.5 text-xs">
          {quality.validators.map((validator) => (
            <p key={validator.name} className="flex items-center justify-between gap-3">
              <span className="text-muted-foreground">Validator {validator.name}</span>
              <span className="font-semibold">
                {validator.pass_rate === null
                  ? "--"
                  : `${(validator.pass_rate * 100).toFixed(0)}% pass (${validator.passed}/${validator.checked})`}
              </span>
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
