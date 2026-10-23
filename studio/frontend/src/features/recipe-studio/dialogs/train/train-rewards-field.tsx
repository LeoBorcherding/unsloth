// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Checkbox } from "@/components/ui/checkbox";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  type RewardRecord,
  formatScore,
  previewCell,
  previewRewards,
} from "@/features/training";
import { cn } from "@/lib/utils";
import { type ReactElement, useEffect, useState } from "react";
import type { TrainCardReward } from "../../types";
import { rewardChoices } from "../../utils/train-card";
import { CollapsibleSectionTriggerButton } from "../shared/collapsible-section-trigger";
import { FieldLabel } from "../shared/field-label";

type RewardScore = { name: string; weighted: number };

function sampleReply(reference: string): string {
  return `<reasoning>\n...\n</reasoning>\n<answer>\n${reference || "..."}\n</answer>`;
}

export function TrainRewardsField({
  id,
  rewards,
  library,
  libraryError,
  sampleRow,
  answerColumn,
  onChange,
}: {
  id: string;
  rewards: TrainCardReward[];
  library: RewardRecord[];
  libraryError: string | null;
  sampleRow: Record<string, unknown> | null;
  answerColumn: string;
  onChange: (rewards: TrainCardReward[]) => void;
}): ReactElement {
  const rowReference =
    sampleRow && answerColumn ? previewCell(sampleRow[answerColumn]) : "";
  const [tryOpen, setTryOpen] = useState(false);
  const [reference, setReference] = useState(rowReference);
  const [reply, setReply] = useState(sampleReply(rowReference));
  const [scores, setScores] = useState<RewardScore[] | null>(null);
  const [total, setTotal] = useState<number | null>(null);
  const [scoreError, setScoreError] = useState<string | null>(null);
  const choices = rewardChoices(library, rewards);
  const picked = new Map(rewards.map((reward) => [reward.name, reward]));

  useEffect(() => {
    setReference(rowReference);
    setReply(sampleReply(rowReference));
  }, [rowReference]);

  useEffect(() => {
    if (!tryOpen || rewards.length === 0) {
      setScores(null);
      setTotal(null);
      return;
    }
    let cancelled = false;
    const timer = setTimeout(() => {
      previewRewards(rewards, reply, reference || null)
        .then((result) => {
          if (!cancelled) {
            setScores(result.scores);
            setTotal(result.total);
            setScoreError(null);
          }
        })
        .catch((error: unknown) => {
          if (!cancelled) {
            setScoreError(error instanceof Error ? error.message : String(error));
          }
        });
    }, 300);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [tryOpen, rewards, reply, reference]);

  const toggle = (name: string, on: boolean) => {
    onChange(
      on
        ? [...rewards, { name, weight: 1 }]
        : rewards.filter((reward) => reward.name !== name),
    );
  };
  const setWeight = (name: string, value: string) => {
    const weight = value.trim() === "" ? Number.NaN : Number(value);
    onChange(
      rewards.map((reward) =>
        reward.name === name ? { ...reward, weight } : reward,
      ),
    );
  };
  const scoreOf = (name: string) =>
    scores?.find((score) => score.name === name)?.weighted;

  return (
    <div className="grid gap-2">
      <FieldLabel
        label="Rewards"
        hint="GRPO writes several answers per prompt and learns from the ones these score higher. Rewards come from the reward library on the Train page."
      />
      {libraryError && (
        <p className="text-xs text-amber-600">
          Could not load the reward library: {libraryError}
        </p>
      )}
      <div className="corner-squircle divide-y divide-border/60 rounded-lg border border-border/70">
        {choices.length === 0 && !libraryError && (
          <p className="px-3 py-2 text-xs text-muted-foreground">
            Loading the reward library...
          </p>
        )}
        {choices.map((choice) => {
          const selected = picked.get(choice.name);
          const checkboxId = `${id}-reward-${choice.name}`;
          const score = scoreOf(choice.name);
          return (
            <div
              key={choice.name}
              className="flex items-center gap-3 px-3 py-2"
              data-reward-row={choice.name}
            >
              <Checkbox
                id={checkboxId}
                className="nodrag"
                checked={Boolean(selected)}
                onCheckedChange={(checked) => toggle(choice.name, checked === true)}
              />
              <label htmlFor={checkboxId} className="min-w-0 flex-1 cursor-pointer">
                <span className="block truncate text-sm font-medium text-foreground">
                  {choice.name}
                </span>
                <span
                  className={cn(
                    "block text-xs",
                    choice.missing ? "text-amber-600" : "text-muted-foreground",
                  )}
                >
                  {choice.description}
                </span>
              </label>
              {tryOpen && selected && score !== undefined && (
                <span className="shrink-0 font-mono text-xs tabular-nums text-foreground">
                  {formatScore(score)}
                </span>
              )}
              {selected && (
                <Input
                  aria-label={`${choice.name} weight`}
                  className="nodrag h-7 w-16 shrink-0 text-right text-xs"
                  inputMode="decimal"
                  defaultValue={String(selected.weight)}
                  onBlur={(event) => setWeight(choice.name, event.target.value)}
                />
              )}
            </div>
          );
        })}
      </div>
      <Collapsible open={tryOpen} onOpenChange={setTryOpen}>
        <CollapsibleTrigger asChild={true}>
          <CollapsibleSectionTriggerButton
            label="Try the rewards"
            open={tryOpen}
            summary={
              total !== null && tryOpen
                ? `Total ${formatScore(total)}`
                : rowReference
                  ? "Scores a reply against row 1"
                  : "Scores a sample reply"
            }
          />
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-2 grid gap-2">
          <div className="grid gap-1.5">
            <FieldLabel
              label="Answer"
              htmlFor={`${id}-try-reference`}
              hint="What the answer column holds for this row. Filled from row 1 of the latest full run."
            />
            <Input
              id={`${id}-try-reference`}
              className="nodrag"
              value={reference}
              onChange={(event) => setReference(event.target.value)}
            />
          </div>
          <div className="grid gap-1.5">
            <FieldLabel
              label="Model reply"
              htmlFor={`${id}-try-reply`}
              hint="Edit it to see how each reward scores a reply."
            />
            <Textarea
              id={`${id}-try-reply`}
              className="nodrag min-h-24 font-mono text-xs"
              value={reply}
              onChange={(event) => setReply(event.target.value)}
            />
          </div>
          {scoreError && <p className="text-xs text-rose-600">{scoreError}</p>}
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
