// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { type ReactElement, useMemo, useState } from "react";
import { useRecipeExecutionsStore } from "../../stores/recipe-executions";
import { useRecipeStudioStore } from "../../stores/recipe-studio";
import type { TrainCardConfig, TrainCardMethod } from "../../types";
import {
  listTrainableColumns,
  parseTrainCardSettings,
  pickTrainSourceExecution,
  summarizeAdvancedTrainSettings,
} from "../../utils/train-card";
import { CollapsibleSectionTriggerButton } from "../shared/collapsible-section-trigger";
import { FieldLabel } from "../shared/field-label";
import { NameField } from "../shared/name-field";

const SUGGESTED_MODELS = [
  "unsloth/Qwen2.5-0.5B-Instruct",
  "unsloth/Llama-3.2-1B-Instruct",
  "unsloth/gemma-3-1b-it",
  "unsloth/Qwen3-4B-Instruct-2507",
];

type NumberField = {
  key: keyof Pick<
    TrainCardConfig,
    | "maxSteps"
    | "learningRate"
    | "loraRank"
    | "loraAlpha"
    | "batchSize"
    | "contextLength"
    | "holdoutPercent"
    | "seed"
  >;
  label: string;
  hint: string;
};

const STEPS_FIELD: NumberField = {
  key: "maxSteps",
  label: "Steps",
  hint: "Optimizer steps to run.",
};

const HOLDOUT_FIELD: NumberField = {
  key: "holdoutPercent",
  label: "Hold out %",
  hint: "Rows kept out of training (0 to 50) so a Benchmark card can score the adapter on data it never saw.",
};

const ADVANCED_FIELDS: NumberField[] = [
  { key: "learningRate", label: "Learning rate", hint: "For example 2e-4." },
  { key: "loraRank", label: "LoRA rank", hint: "Adapter size." },
  { key: "loraAlpha", label: "LoRA alpha", hint: "Adapter scaling." },
  { key: "batchSize", label: "Batch size", hint: "Rows per device step." },
  {
    key: "contextLength",
    label: "Max seq length",
    hint: "Longer rows are truncated.",
  },
];

const SEED_FIELD: NumberField = {
  key: "seed",
  label: "Seed",
  hint: "Picks which rows are held out, and seeds training.",
};

type TrainDialogProps = {
  config: TrainCardConfig;
  onUpdate: (patch: Partial<TrainCardConfig>) => void;
};

export function TrainDialog({
  config,
  onUpdate,
}: TrainDialogProps): ReactElement {
  const configs = useRecipeStudioStore((state) => state.configs);
  const executions = useRecipeExecutionsStore((state) => state.executions);
  const columns = useMemo(() => listTrainableColumns(configs), [configs]);
  const source = useMemo(() => pickTrainSourceExecution(executions), [executions]);
  const { errors } = parseTrainCardSettings(config, columns);
  const modelId = `${config.id}-base-model`;
  const modelListId = `${config.id}-base-model-list`;
  const methodId = `${config.id}-method`;
  const promptId = `${config.id}-prompt`;
  const responseId = `${config.id}-response`;
  const outputId = `${config.id}-output`;
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const columnSelect = (
    id: string,
    value: string,
    onChange: (value: string) => void,
  ) => (
    <Select value={value || undefined} onValueChange={onChange}>
      <SelectTrigger className="nodrag w-full" id={id}>
        <SelectValue placeholder="Choose a column" />
      </SelectTrigger>
      <SelectContent>
        {columns.map((column) => (
          <SelectItem key={column} value={column}>
            {column}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );

  const numberInput = (field: NumberField) => {
    const inputId = `${config.id}-${field.key}`;
    return (
      <div key={field.key} className="grid gap-1.5">
        <FieldLabel label={field.label} htmlFor={inputId} hint={field.hint} />
        <Input
          id={inputId}
          className="nodrag"
          inputMode="decimal"
          value={config[field.key]}
          onChange={(event) => onUpdate({ [field.key]: event.target.value })}
        />
      </div>
    );
  };

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground">
        {source
          ? `Trains on the latest full run (${source.run_name?.trim() || source.id}, ${source.rows} rows).`
          : "Run the full recipe once; the card trains on its latest completed full run."}
      </p>
      <div className="grid gap-1.5">
        <FieldLabel
          label="Base model"
          htmlFor={modelId}
          hint="A Hugging Face model id or a model already on this machine."
        />
        <Input
          id={modelId}
          className="nodrag"
          list={modelListId}
          placeholder={SUGGESTED_MODELS[0]}
          value={config.baseModel}
          onChange={(event) => onUpdate({ baseModel: event.target.value })}
        />
        <datalist id={modelListId}>
          {SUGGESTED_MODELS.map((model) => (
            <option key={model} value={model} />
          ))}
        </datalist>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="grid gap-1.5">
          <FieldLabel
            label="Prompt column"
            htmlFor={promptId}
            hint="Becomes the user turn."
          />
          {columnSelect(promptId, config.promptColumn, (value) =>
            onUpdate({ promptColumn: value }),
          )}
        </div>
        <div className="grid gap-1.5">
          <FieldLabel
            label="Response column"
            htmlFor={responseId}
            hint="Becomes the assistant turn the model learns to write."
          />
          {columnSelect(responseId, config.responseColumn, (value) =>
            onUpdate({ responseColumn: value }),
          )}
        </div>
        {numberInput(STEPS_FIELD)}
        {numberInput(HOLDOUT_FIELD)}
      </div>
      <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
        <CollapsibleTrigger asChild={true}>
          <CollapsibleSectionTriggerButton
            label="Advanced"
            open={advancedOpen}
            summary={summarizeAdvancedTrainSettings(config)}
          />
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-3 space-y-3">
          <div className="grid gap-3 sm:grid-cols-3">
            <NameField
              label="Card name"
              hint="Shown on the canvas."
              value={config.name}
              onChange={(value) => onUpdate({ name: value })}
            />
            <div className="grid gap-1.5">
              <FieldLabel
                label="Run name"
                htmlFor={outputId}
                hint="Names the training run and its saved adapter."
              />
              <Input
                id={outputId}
                className="nodrag"
                value={config.outputName}
                onChange={(event) =>
                  onUpdate({ outputName: event.target.value })
                }
              />
            </div>
            {numberInput(SEED_FIELD)}
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="grid gap-1.5">
              <FieldLabel
                label="Method"
                htmlFor={methodId}
                hint="QLoRA loads the base model in 4-bit and fits smaller GPUs."
              />
              <Select
                value={config.method}
                onValueChange={(value) =>
                  onUpdate({ method: value as TrainCardMethod })
                }
              >
                <SelectTrigger className="nodrag w-full" id={methodId}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="qlora">QLoRA (4-bit)</SelectItem>
                  <SelectItem value="lora">LoRA (16-bit)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            {ADVANCED_FIELDS.map(numberInput)}
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
