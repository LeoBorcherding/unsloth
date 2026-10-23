// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { buildFeedbackSeedRows } from "./feedback-seed";
import type {
  LoopRecord,
  LoopRound,
  LoopSettings,
  LoopStatus,
  LoopSteps,
} from "./types";

// A plateau is a run of rounds that never beats the best earlier score by a full point.
const PLATEAU_MIN_GAIN = 0.01;
const KEPT_FAILURES = 30;

export type LoopStopReason = Exclude<LoopStatus, "running" | "stopped" | "error">;

function scores(rounds: LoopRound[]): number[] {
  return rounds
    .filter((round) => round.status === "done" && round.result)
    .map((round) => (round.result as NonNullable<LoopRound["result"]>).score);
}

export function bestRound(rounds: LoopRound[]): LoopRound | null {
  let best: LoopRound | null = null;
  for (const round of rounds) {
    if (round.status !== "done" || !round.result) {
      continue;
    }
    if (!best?.result || round.result.score > best.result.score) {
      best = round;
    }
  }
  return best;
}

/** Why the loop stops before another round, or null to keep going. */
export function loopStopReason(
  rounds: LoopRound[],
  settings: LoopSettings,
): LoopStopReason | null {
  const done = scores(rounds);
  if (done.length > 0 && Math.max(...done) >= settings.targetScore) {
    return "target_reached";
  }
  const patience = settings.plateauRounds;
  if (patience > 0 && done.length > patience) {
    const before = Math.max(...done.slice(0, -patience));
    const recent = Math.max(...done.slice(-patience));
    if (recent < before + PLATEAU_MIN_GAIN) {
      return "plateau";
    }
  }
  if (done.length >= settings.maxRounds) {
    return "max_rounds";
  }
  return null;
}

export function createLoopRecord(input: {
  id: string;
  recipeId: string;
  settings: LoopSettings;
  now: number;
}): LoopRecord {
  return {
    id: input.id,
    recipeId: input.recipeId,
    createdAt: input.now,
    updatedAt: input.now,
    status: "running",
    settings: input.settings,
    holdout: null,
    rounds: [],
    error: null,
  };
}

/** Continue a saved loop: finished rounds stay, an unfinished one is redone from the last adapter. */
export function resumeLoopRecord(
  record: LoopRecord,
  settings: LoopSettings,
  now: number,
): LoopRecord {
  const rounds = record.rounds.filter((round) => round.status === "done");
  return {
    ...record,
    settings,
    status: "running",
    error: null,
    updatedAt: now,
    rounds,
    // The held-out split is fixed by round 1; without a finished round it is carved again.
    holdout: rounds.length > 0 ? record.holdout : null,
  };
}

function isAbort(error: unknown, signal: AbortSignal): boolean {
  return signal.aborted || (error instanceof Error && error.name === "AbortError");
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Drives rounds until a stop rule, saving after every step so the history survives a reload.
 * Each round writes rows from the last round's failures plus fresh rows, continues training from
 * the last round's adapter, and scores it on the held-out rows round 1 set aside.
 */
export async function runLoop(input: {
  record: LoopRecord;
  steps: LoopSteps;
  recipeColumns: string[];
  signal: AbortSignal;
  save: (record: LoopRecord) => void;
  now?: () => number;
}): Promise<LoopRecord> {
  const { steps, signal, save } = input;
  const now = input.now ?? Date.now;
  let record = input.record;
  const commit = (patch: Partial<LoopRecord>): void => {
    record = { ...record, ...patch, updatedAt: now() };
    save(record);
  };
  const setRound = (patch: Partial<LoopRound>): void => {
    const rounds = record.rounds.slice();
    rounds[rounds.length - 1] = {
      ...(rounds[rounds.length - 1] as LoopRound),
      ...patch,
    };
    commit({ rounds });
  };

  try {
    while (true) {
      if (signal.aborted) {
        commit({ status: "stopped" });
        return record;
      }
      const reason = loopStopReason(record.rounds, record.settings);
      if (reason) {
        commit({ status: reason });
        return record;
      }
      const previous = record.rounds[record.rounds.length - 1] ?? null;
      const index = record.rounds.length + 1;
      commit({
        rounds: [
          ...record.rounds,
          {
            index,
            status: "generating",
            dataset: null,
            adapter: null,
            result: null,
            error: null,
            startedAt: now(),
            finishedAt: null,
          },
        ],
      });
      const seedRows = buildFeedbackSeedRows({
        failures: previous?.result?.failedExamples ?? [],
        rows: record.settings.rowsPerRound,
        failureShare: record.settings.failureShare,
        recipeColumns: input.recipeColumns,
      });
      const dataset = await steps.generate({ round: index, seedRows, signal });
      setRound({ dataset, status: "training" });
      const { adapter, holdout } = await steps.train({
        round: index,
        dataset,
        holdout: record.holdout,
        continueFrom: previous?.adapter?.outputDir ?? null,
        signal,
      });
      if (!record.holdout) {
        commit({ holdout });
      }
      setRound({ adapter, status: "scoring" });
      const result = await steps.score({ round: index, adapter, holdout, signal });
      setRound({
        result: {
          ...result,
          failedExamples: result.failedExamples.slice(0, KEPT_FAILURES),
        },
        status: "done",
        finishedAt: now(),
      });
    }
  } catch (error) {
    const aborted = isAbort(error, signal);
    const message = aborted ? "Stopped." : errorText(error);
    const last = record.rounds[record.rounds.length - 1];
    if (last && last.status !== "done") {
      setRound({ status: "error", error: message, finishedAt: now() });
    }
    commit(aborted ? { status: "stopped" } : { status: "error", error: message });
    return record;
  }
}
