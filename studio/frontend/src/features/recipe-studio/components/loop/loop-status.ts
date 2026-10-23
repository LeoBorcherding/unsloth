// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { bestRound } from "../../loop/controller";
import type { LoopRecord, LoopRoundStatus } from "../../loop/types";

export function roundPhaseLabel(status: LoopRoundStatus): string {
  switch (status) {
    case "generating":
      return "Writing rows";
    case "training":
      return "Training";
    case "scoring":
      return "Benchmarking";
    case "done":
      return "Done";
    default:
      return "Failed";
  }
}

function percent(score: number): string {
  return `${Math.round(score * 100)}%`;
}

/** One sentence on where a loop stands. */
export function describeLoopStatus(record: LoopRecord, interrupted: boolean): string {
  const last = record.rounds[record.rounds.length - 1];
  const best = bestRound(record.rounds);
  const bestText = best?.result ? ` Best ${percent(best.result.score)} in round ${best.index}.` : "";
  if (interrupted) {
    return `Interrupted in round ${last?.index ?? 1}. Resume picks up from the last finished round.${bestText}`;
  }
  switch (record.status) {
    case "running":
      return last ? `Round ${last.index}: ${roundPhaseLabel(last.status)}.` : "Starting.";
    case "target_reached":
      return `Reached the ${percent(record.settings.targetScore)} target in round ${best?.index ?? last?.index}.`;
    case "plateau":
      return `Stopped after round ${last?.index}: no gain for ${record.settings.plateauRounds} rounds.${bestText}`;
    case "max_rounds":
      return `Ran ${record.rounds.length} rounds without reaching ${percent(record.settings.targetScore)}.${bestText}`;
    case "stopped":
      return `Stopped in round ${last?.index ?? 1}.${bestText}`;
    default:
      return record.error ?? "The loop failed.";
  }
}
