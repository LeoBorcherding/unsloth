// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import type { ReactElement } from "react";
import type { SeedConfig } from "../../types";
import { RECIPE_STUDIO_REFERENCE_BADGE_TONES } from "../../utils/ui-tones";
import { useRecipeLoopView } from "./loop-context";

const SHOWN_COLUMNS = 5;

export function FeedbackSeedBody({ config }: { config: SeedConfig }): ReactElement {
  const loop = useRecipeLoopView();
  const latest = loop?.latest?.rounds.at(-1)?.dataset ?? null;
  const columns = config.seed_columns ?? [];
  return (
    <div className="space-y-2">
      <p className="text-xs text-muted-foreground">
        {latest
          ? `Round ${loop?.latest?.rounds.at(-1)?.index}: ${latest.fromFailures} rows from failed examples, ${latest.rows - latest.fromFailures} fresh.`
          : "The Loop card fills this each round with the Benchmark card's failed examples, plus blank rows for fresh data."}
      </p>
      {columns.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {columns.slice(0, SHOWN_COLUMNS).map((column) => (
            <Badge
              key={column}
              variant="outline"
              className={RECIPE_STUDIO_REFERENCE_BADGE_TONES.seed}
            >
              {column}
            </Badge>
          ))}
          {columns.length > SHOWN_COLUMNS && (
            <span className="text-ui-11 text-muted-foreground">
              +{columns.length - SHOWN_COLUMNS}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
