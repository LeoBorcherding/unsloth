// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { fetchBenchmarkTasks } from "@/features/benchmark/api/benchmark-api";
import { type ReactElement, useEffect, useMemo, useState } from "react";
import { fetchLoadedModelStatus } from "../../api";
import { useRecipeExecutionsStore } from "../../stores/recipe-executions";
import { useRecipeStudioStore } from "../../stores/recipe-studio";
import type {
  BenchmarkBaselineMode,
  BenchmarkConfig,
  BenchmarkScorer,
} from "../../types";
import {
  BENCHMARK_MAX_ROWS,
  type BenchmarkTaskOption,
  RECIPE_HELDOUT_TASK,
  getBenchmarkColumns,
  getCardTasks,
  isRecipeTask,
  lmEvalBlockedReason,
  pickBenchmarkExecution,
} from "../../utils/benchmark";
import { FieldLabel } from "../shared/field-label";
import { NameField } from "../shared/name-field";

const NONE_VALUE = "__none__";

const SCORER_LABELS: Record<BenchmarkScorer, string> = {
  reference: "Matches the reference answer",
  judge: "LLM judge",
  json: "Valid JSON",
  markdown: "Valid markdown",
  python: "Valid Python",
};

const BASELINE_LABELS: Record<BenchmarkBaselineMode, string> = {
  // biome-ignore lint/style/useNamingConvention: ui schema
  adapter_off: "Same model, LoRA adapter off",
  alias: "Another model",
  none: "No baseline",
};

type BenchmarkDialogProps = {
  config: BenchmarkConfig;
  modelConfigAliases: string[];
  onUpdate: (patch: Partial<BenchmarkConfig>) => void;
};

function OptionSelect({
  id,
  value,
  options,
  placeholder,
  allowNone = false,
  onChange,
}: {
  id: string;
  value: string;
  options: Array<{ value: string; label: string; disabled?: boolean }>;
  placeholder: string;
  allowNone?: boolean;
  onChange: (value: string) => void;
}): ReactElement {
  return (
    <Select
      value={value || NONE_VALUE}
      onValueChange={(next) => onChange(next === NONE_VALUE ? "" : next)}
    >
      <SelectTrigger className="nodrag w-full" id={id}>
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent>
        {(allowNone || !value) && (
          <SelectItem value={NONE_VALUE}>
            {allowNone ? "None" : placeholder}
          </SelectItem>
        )}
        {options.map((option) => (
          <SelectItem
            key={option.value}
            value={option.value}
            disabled={option.disabled}
          >
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

export function BenchmarkDialog({
  config,
  modelConfigAliases,
  onUpdate,
}: BenchmarkDialogProps): ReactElement {
  const executions = useRecipeExecutionsStore((state) => state.executions);
  const recipeConfigs = useRecipeStudioStore((state) => state.configs);
  const execution = useMemo(
    () => pickBenchmarkExecution(executions),
    [executions],
  );
  const columns = useMemo(() => {
    const fromRun = getBenchmarkColumns(execution);
    if (fromRun.length > 0) {
      return fromRun;
    }
    return Object.values(recipeConfigs)
      .filter((item) => ["sampler", "llm", "expression"].includes(item.kind))
      .map((item) => item.name);
  }, [execution, recipeConfigs]);
  const columnOptions = columns.map((name) => ({ value: name, label: name }));
  const modelOptions = modelConfigAliases.map((name) => ({
    value: name,
    label: name,
  }));
  const id = (field: string) => `${config.id}-${field}`;
  const [tasks, setTasks] = useState<BenchmarkTaskOption[]>([
    { id: RECIPE_HELDOUT_TASK, name: "Recipe held-out", runner: "recipe" },
  ]);
  const [lmEvalBlocked, setLmEvalBlocked] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    fetchBenchmarkTasks()
      .then((response) => {
        const cardTasks = getCardTasks(response.tasks);
        if (!cancelled && cardTasks.length > 0) {
          setTasks(cardTasks);
        }
      })
      .catch(() => undefined);
    fetchLoadedModelStatus().then((status) => {
      if (!cancelled) {
        setLmEvalBlocked(lmEvalBlockedReason(status));
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);
  const recipeTask = isRecipeTask(config.task);

  return (
    <div className="space-y-4">
      <NameField
        label="Card name"
        hint="Name shown on the canvas."
        value={config.name}
        onChange={(value) => onUpdate({ name: value })}
      />
      <div className="grid gap-1.5">
        <FieldLabel
          label="Benchmark"
          htmlFor={id("task")}
          hint="Recipe held-out scores rows from this recipe on any chat model. The others are lm-eval tasks from the Benchmarks page and score the GGUF loaded in Chat."
        />
        <OptionSelect
          id={id("task")}
          value={config.task || RECIPE_HELDOUT_TASK}
          options={tasks.map((task) => ({
            value: task.id,
            label: task.name,
            disabled: task.runner !== "recipe" && Boolean(lmEvalBlocked),
          }))}
          placeholder="Select a benchmark"
          onChange={(value) => onUpdate({ task: value || RECIPE_HELDOUT_TASK })}
        />
        {lmEvalBlocked && (
          <p className="text-xs text-muted-foreground">{lmEvalBlocked}</p>
        )}
      </div>
      {!recipeTask && (
        <div className="grid gap-1.5">
          <FieldLabel
            label="Samples"
            htmlFor={id("limit")}
            hint="Score only the first N samples. Leave empty for the full task (can take hours)."
          />
          <Input
            id={id("limit")}
            className="nodrag"
            inputMode="numeric"
            value={config.limit ?? ""}
            onChange={(event) => onUpdate({ limit: event.target.value })}
          />
        </div>
      )}
      {recipeTask && (
        <>
          <div className="grid gap-1.5">
            <FieldLabel
              label="Model to test"
              htmlFor={id("model")}
              hint="A model block on the canvas. Load a fine-tuned adapter in Chat and point a local model block at it."
            />
            <OptionSelect
              id={id("model")}
              value={config.model_alias}
              options={modelOptions}
              placeholder="Select a model"
              // biome-ignore lint/style/useNamingConvention: ui schema
              onChange={(value) => onUpdate({ model_alias: value })}
            />
          </div>
          <div className="grid gap-1.5">
            <FieldLabel
              label="Compare against"
              htmlFor={id("baseline")}
              hint="The delta is the tested model's score minus this one. Adapter off needs a LoRA loaded in Chat."
            />
            <OptionSelect
              id={id("baseline")}
              value={config.baseline_mode}
              options={Object.entries(BASELINE_LABELS).map(
                ([value, label]) => ({ value, label }),
              )}
              placeholder="Select a baseline"
              onChange={(value) =>
                // biome-ignore lint/style/useNamingConvention: ui schema
                onUpdate({
                  baseline_mode: (value || "none") as BenchmarkBaselineMode,
                })
              }
            />
            {config.baseline_mode === "alias" && (
              <OptionSelect
                id={id("baseline-alias")}
                value={config.baseline_alias ?? ""}
                options={modelOptions}
                placeholder="Select the baseline model"
                // biome-ignore lint/style/useNamingConvention: ui schema
                onChange={(value) => onUpdate({ baseline_alias: value })}
              />
            )}
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="grid gap-1.5">
              <FieldLabel
                label="Prompt column"
                htmlFor={id("prompt")}
                hint="Sent to the model as the user message."
              />
              <OptionSelect
                id={id("prompt")}
                value={config.prompt_column}
                options={columnOptions}
                placeholder="Select a column"
                // biome-ignore lint/style/useNamingConvention: ui schema
                onChange={(value) => onUpdate({ prompt_column: value })}
              />
            </div>
            <div className="grid gap-1.5">
              <FieldLabel
                label="Reference column"
                htmlFor={id("reference")}
                hint="The expected answer. Needed for reference match, optional for the judge."
              />
              <OptionSelect
                id={id("reference")}
                value={config.reference_column ?? ""}
                options={columnOptions}
                placeholder="None"
                allowNone={true}
                // biome-ignore lint/style/useNamingConvention: ui schema
                onChange={(value) => onUpdate({ reference_column: value })}
              />
            </div>
          </div>
          <div className="grid gap-1.5">
            <FieldLabel
              label="Score with"
              htmlFor={id("scorer")}
              hint="Each answer passes or fails; the score is the pass rate."
            />
            <OptionSelect
              id={id("scorer")}
              value={config.scorer}
              options={Object.entries(SCORER_LABELS).map(([value, label]) => ({
                value,
                label,
              }))}
              placeholder="Select a scorer"
              onChange={(value) =>
                onUpdate({ scorer: (value || "reference") as BenchmarkScorer })
              }
            />
          </div>
          {config.scorer === "judge" && (
            <div className="grid gap-3">
              <div className="grid gap-1.5">
                <FieldLabel
                  label="Judge model"
                  htmlFor={id("judge")}
                  hint="On one GPU, point this at the same local model: with adapter off as baseline it judges with the adapter off."
                />
                <OptionSelect
                  id={id("judge")}
                  value={config.judge_alias ?? ""}
                  options={modelOptions}
                  placeholder="Select the judge"
                  // biome-ignore lint/style/useNamingConvention: ui schema
                  onChange={(value) => onUpdate({ judge_alias: value })}
                />
              </div>
              <div className="grid gap-1.5">
                <FieldLabel label="Judge criteria" htmlFor={id("rubric")} />
                <Textarea
                  id={id("rubric")}
                  className="corner-squircle nodrag min-h-20"
                  placeholder="The answer is correct, complete and follows the instruction."
                  value={config.rubric ?? ""}
                  onChange={(event) => onUpdate({ rubric: event.target.value })}
                />
              </div>
            </div>
          )}
          <div className="grid gap-1.5">
            <FieldLabel
              label="System prompt"
              htmlFor={id("system")}
              hint="Optional. Sent before every prompt."
            />
            <Textarea
              id={id("system")}
              className="corner-squircle nodrag min-h-16"
              value={config.system_prompt ?? ""}
              // biome-ignore lint/style/useNamingConvention: ui schema
              onChange={(event) =>
                onUpdate({ system_prompt: event.target.value })
              }
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="grid gap-1.5">
              <FieldLabel
                label="Held-out %"
                htmlFor={id("holdout")}
                hint="For a model you picked by hand: the last N% of the run's rows. A wired-in Train card's own held-out rows replace this."
              />
              <Input
                id={id("holdout")}
                className="nodrag"
                inputMode="numeric"
                value={config.holdout_percent}
                // biome-ignore lint/style/useNamingConvention: ui schema
                onChange={(event) =>
                  onUpdate({ holdout_percent: event.target.value })
                }
              />
            </div>
            <div className="grid gap-1.5">
              <FieldLabel
                label="Max rows"
                htmlFor={id("max-rows")}
                hint={`Caps how many held-out rows are scored (up to ${BENCHMARK_MAX_ROWS}).`}
              />
              <Input
                id={id("max-rows")}
                className="nodrag"
                inputMode="numeric"
                value={config.max_rows}
                // biome-ignore lint/style/useNamingConvention: ui schema
                onChange={(event) => onUpdate({ max_rows: event.target.value })}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
}
