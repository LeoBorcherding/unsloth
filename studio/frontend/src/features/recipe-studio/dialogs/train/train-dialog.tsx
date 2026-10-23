// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SegmentedTabsList } from "@/components/segmented-tabs";
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
import { Tabs } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { usePlatformStore } from "@/config/env";
import { type ReactElement, useMemo, useState } from "react";
import { useRewardLibrary, useTrainSampleRow } from "../../hooks/use-train-card-inputs";
import { useRecipeExecutionsStore } from "../../stores/recipe-executions";
import { useRecipeStudioStore } from "../../stores/recipe-studio";
import type {
  TrainCardConfig,
  TrainCardMethod,
  TrainCardObjective,
  TrainCardPairRule,
} from "../../types";
import {
  TRAIN_OBJECTIVE_LABELS,
  isPreferenceTrainObjective,
  listTrainableColumns,
  parseTrainCardSettings,
  pickTrainSourceExecution,
  summarizeAdvancedTrainSettings,
  switchTrainObjective,
} from "../../utils/train-card";
import { CollapsibleSectionTriggerButton } from "../shared/collapsible-section-trigger";
import { FieldLabel } from "../shared/field-label";
import { NameField } from "../shared/name-field";
import { TrainRewardsField } from "./train-rewards-field";

const SUGGESTED_MODELS = [
  "unsloth/Qwen2.5-0.5B-Instruct",
  "unsloth/Qwen3-0.6B",
  "unsloth/Llama-3.2-1B-Instruct",
  "unsloth/gemma-3-1b-it",
  "unsloth/Qwen3-4B-Instruct-2507",
];

const OBJECTIVE_HINTS: Record<TrainCardObjective, string> = {
  sft: "Learn to write the response column.",
  grpo: "Write several answers per prompt and learn from the ones the rewards score higher.",
  dpo: "Learn to prefer the better of two answers. No generation, no rewards.",
  orpo: "Like DPO with an SFT term in the same pass, so it also works from a base model.",
};

const PAIR_RULES: { value: TrainCardPairRule; label: string }[] = [
  { value: "a_better", label: "Answer A is always better" },
  { value: "winner", label: "A column says A or B" },
  { value: "scores", label: "Higher score wins" },
];

type TextField = keyof Pick<
  TrainCardConfig,
  | "maxSteps"
  | "learningRate"
  | "loraRank"
  | "loraAlpha"
  | "batchSize"
  | "contextLength"
  | "holdoutPercent"
  | "seed"
  | "numGenerations"
  | "maxCompletionLength"
  | "beta"
>;

type NumberField = { key: TextField; label: string; hint: string };

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
  {
    key: "seed",
    label: "Seed",
    hint: "Picks which rows are held out, and seeds training.",
  },
];

const GRPO_FIELDS: NumberField[] = [
  {
    key: "numGenerations",
    label: "Generations",
    hint: "Answers written per prompt and compared with each other.",
  },
  {
    key: "maxCompletionLength",
    label: "Max answer tokens",
    hint: "Longer answers are cut off.",
  },
  {
    key: "beta",
    label: "Beta",
    hint: "KL penalty toward the base model. Blank uses 0, as TRL does.",
  },
];

const BETA_FIELD: NumberField = {
  key: "beta",
  label: "Beta",
  hint: "How far the adapter may drift from the base model. Blank uses 0.1.",
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
  const isMac = usePlatformStore((state) => state.deviceType) === "mac";
  const columns = useMemo(() => listTrainableColumns(configs), [configs]);
  const source = useMemo(() => pickTrainSourceExecution(executions), [executions]);
  const objective = config.objective;
  const grpo = objective === "grpo";
  const preference = isPreferenceTrainObjective(objective);
  const { library, libraryError, rewardInfo } = useRewardLibrary(grpo);
  const sampleRow = useTrainSampleRow(source, grpo);
  const { errors } = parseTrainCardSettings(config, columns, rewardInfo);
  const modelId = `${config.id}-base-model`;
  const modelListId = `${config.id}-base-model-list`;
  const methodId = `${config.id}-method`;
  const outputId = `${config.id}-output`;
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const columnField = (
    key: keyof Pick<
      TrainCardConfig,
      | "promptColumn"
      | "responseColumn"
      | "pairA"
      | "pairB"
      | "winnerColumn"
      | "scoreAColumn"
      | "scoreBColumn"
    >,
    label: string,
    hint: string,
  ) => {
    const inputId = `${config.id}-${key}`;
    return (
      <div className="grid gap-1.5">
        <FieldLabel label={label} htmlFor={inputId} hint={hint} />
        <Select
          value={config[key] || undefined}
          onValueChange={(value) => onUpdate({ [key]: value })}
        >
          <SelectTrigger className="nodrag w-full" id={inputId}>
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
      </div>
    );
  };

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
    <div className="space-y-4">
      <p className="text-xs text-muted-foreground">
        {source
          ? `Trains on the latest full run (${source.run_name?.trim() || source.id}, ${source.rows} rows).`
          : "Run the full recipe once; the card trains on its latest completed full run."}
      </p>
      <div className="grid gap-1.5">
        <FieldLabel
          label="Objective"
          hint="The same objectives as the Train page. RL runs use its reward library and live charts."
        />
        <Tabs
          value={objective}
          onValueChange={(value) =>
            onUpdate(switchTrainObjective(config, value as TrainCardObjective))
          }
          className="contents"
        >
          <SegmentedTabsList
            value={objective}
            ariaLabel="Training objective"
            size="compact"
            options={[
              { value: "sft", label: TRAIN_OBJECTIVE_LABELS.sft },
              { value: "grpo", label: TRAIN_OBJECTIVE_LABELS.grpo, disabled: isMac },
              { value: "dpo", label: TRAIN_OBJECTIVE_LABELS.dpo, disabled: isMac },
              { value: "orpo", label: TRAIN_OBJECTIVE_LABELS.orpo, disabled: isMac },
            ]}
          />
        </Tabs>
        <p className="text-xs text-muted-foreground">
          {isMac
            ? "Preference and RL training are not available on Apple Silicon yet."
            : OBJECTIVE_HINTS[objective]}
        </p>
      </div>
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
        {columnField(
          "promptColumn",
          "Prompt column",
          preference || grpo
            ? "The question the model answers."
            : "Becomes the user turn.",
        )}
        {objective === "sft" &&
          columnField(
            "responseColumn",
            "Response column",
            "Becomes the assistant turn the model learns to write.",
          )}
        {grpo &&
          columnField(
            "responseColumn",
            "Answer column",
            "The reference answer. Rewards like exact-answer compare the model's answer to it.",
          )}
        {preference && (
          <>
            {columnField("pairA", "Answer A", "One answer to the prompt.")}
            {columnField("pairB", "Answer B", "Another answer to the same prompt.")}
            <div className="grid gap-1.5">
              <FieldLabel
                label="Better answer"
                htmlFor={`${config.id}-pair-rule`}
                hint="How each row picks chosen and rejected. Rows with no clear winner are left out."
              />
              <Select
                value={config.pairRule}
                onValueChange={(value) =>
                  onUpdate({ pairRule: value as TrainCardPairRule })
                }
              >
                <SelectTrigger className="nodrag w-full" id={`${config.id}-pair-rule`}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {PAIR_RULES.map((rule) => (
                    <SelectItem key={rule.value} value={rule.value}>
                      {rule.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </>
        )}
      </div>
      {preference && config.pairRule === "winner" && (
        <div className="grid gap-3 sm:grid-cols-2">
          {columnField(
            "winnerColumn",
            "Winner column",
            "Says A or B per row, for example an LLM judge column. Ties and anything else are left out.",
          )}
        </div>
      )}
      {preference && config.pairRule === "scores" && (
        <div className="grid gap-3 sm:grid-cols-2">
          {columnField("scoreAColumn", "Score of A", "A number or a judge column's score.")}
          {columnField("scoreBColumn", "Score of B", "A number or a judge column's score.")}
        </div>
      )}
      {grpo && (
        <TrainRewardsField
          id={config.id}
          rewards={config.rewards}
          library={library}
          libraryError={libraryError}
          sampleRow={sampleRow}
          answerColumn={config.responseColumn}
          onChange={(rewards) => onUpdate({ rewards })}
        />
      )}
      <div className="grid gap-3 sm:grid-cols-2">
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
          <div className="grid gap-3 sm:grid-cols-2">
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
            {grpo && GRPO_FIELDS.map(numberInput)}
            {preference && numberInput(BETA_FIELD)}
          </div>
          {grpo && (
            <div className="grid gap-1.5">
              <FieldLabel
                label="System prompt"
                htmlFor={`${config.id}-system-prompt`}
                hint="Asks for the format the rewards read. The default is the Unsloth GRPO notebooks' reasoning and answer tags."
              />
              <Textarea
                id={`${config.id}-system-prompt`}
                className="nodrag min-h-20 font-mono text-xs"
                value={config.systemPrompt}
                onChange={(event) =>
                  onUpdate({ systemPrompt: event.target.value })
                }
              />
            </div>
          )}
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
