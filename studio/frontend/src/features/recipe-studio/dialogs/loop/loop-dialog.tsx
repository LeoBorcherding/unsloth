// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { type ReactElement, useState } from "react";
import type { LoopCardConfig } from "../../types";
import { parseLoopSettings } from "../../utils/loop-card";
import { CollapsibleSectionTriggerButton } from "../shared/collapsible-section-trigger";
import { FieldLabel } from "../shared/field-label";
import { NameField } from "../shared/name-field";

type TextField = {
  key: keyof Pick<
    LoopCardConfig,
    "targetScore" | "maxRounds" | "plateauRounds" | "rowsPerRound" | "failureShare"
  >;
  label: string;
  hint: string;
};

const MAIN_FIELDS: TextField[] = [
  {
    key: "targetScore",
    label: "Target score %",
    hint: "Stop once the Benchmark card's pass rate on the held-out rows reaches this.",
  },
  {
    key: "maxRounds",
    label: "Max rounds",
    hint: "Each round generates rows, trains and benchmarks once.",
  },
];

const ADVANCED_FIELDS: TextField[] = [
  {
    key: "rowsPerRound",
    label: "Rows per round",
    hint: "Rows the recipe generates each round. Round 1 also gives up the Train card's hold out.",
  },
  {
    key: "failureShare",
    label: "Rows from failures %",
    hint: "From round 2, this share of the rows is written from the last round's failed examples; the rest are fresh.",
  },
];

type LoopDialogProps = {
  config: LoopCardConfig;
  onUpdate: (patch: Partial<LoopCardConfig>) => void;
};

export function LoopDialog({ config, onUpdate }: LoopDialogProps): ReactElement {
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const { errors } = parseLoopSettings(config);

  const textInput = (field: TextField) => {
    const inputId = `${config.id}-${field.key}`;
    return (
      <div key={field.key} className="grid gap-1.5">
        <FieldLabel label={field.label} htmlFor={inputId} hint={field.hint} />
        <Input
          id={inputId}
          className="nodrag"
          inputMode="numeric"
          value={config[field.key]}
          onChange={(event) => onUpdate({ [field.key]: event.target.value })}
        />
      </div>
    );
  };

  return (
    <div className="space-y-4">
      <p className="text-xs text-muted-foreground">
        Each round runs the recipe, continues training the Train card's adapter on the new rows,
        and scores it with the Benchmark card on the same held-out rows. Failed examples go back
        into the next round through the Benchmark failures seed.
      </p>
      <div className="grid gap-3 sm:grid-cols-2">{MAIN_FIELDS.map(textInput)}</div>
      <div className="flex items-center justify-between gap-3 rounded-lg border border-border/60 px-3 py-2">
        <FieldLabel
          label="Stop on plateau"
          htmlFor={`${config.id}-plateau`}
          hint="Stop when the score has not gained a point for this many rounds."
        />
        <div className="flex items-center gap-2">
          {config.stopOnPlateau && (
            <Input
              aria-label="Plateau rounds"
              className="nodrag h-8 w-16"
              inputMode="numeric"
              value={config.plateauRounds}
              onChange={(event) => onUpdate({ plateauRounds: event.target.value })}
            />
          )}
          <Switch
            id={`${config.id}-plateau`}
            checked={config.stopOnPlateau}
            onCheckedChange={(checked) => onUpdate({ stopOnPlateau: checked })}
          />
        </div>
      </div>
      <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
        <CollapsibleTrigger asChild={true}>
          <CollapsibleSectionTriggerButton
            label="Advanced"
            open={advancedOpen}
            summary={`${config.rowsPerRound} rows a round, ${config.failureShare}% from failures`}
          />
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-3 space-y-3">
          <div className="grid gap-3 sm:grid-cols-2">
            {ADVANCED_FIELDS.map(textInput)}
            <NameField
              label="Card name"
              hint="Shown on the canvas."
              value={config.name}
              onChange={(value) => onUpdate({ name: value })}
            />
          </div>
        </CollapsibleContent>
      </Collapsible>
      {errors.length > 0 && (
        <p className="text-xs text-amber-600">
          <span className="font-semibold">Needs attention: </span>
          {errors.join(" ")}
        </p>
      )}
    </div>
  );
}
