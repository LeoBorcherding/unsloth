// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// React-free so the node --test suite can import it.

import type { ApiMonitorResponse } from "@/features/chat/types/api";

type ChatResidentWindow = {
  residentCheckpoint: string | null | undefined;
  loadedContextLength: number | null;
  modelLoading: boolean;
};

// A refresh that failed or read the old value gets another try after this long.
const RETRY_MS = 15_000;

/**
 * True once per drift: the monitor reads the live window every poll, the chat store only on its own
 * loads and focus, so an API reload of the same model at a new context would leave the header stale.
 */
export function createResidentDriftWatch(now: () => number = Date.now): (
  monitor: Pick<ApiMonitorResponse, "active_model" | "context_length">,
  chat: ChatResidentWindow,
) => boolean {
  let flagged: { key: string; at: number } | null = null;
  return (monitor, chat) => {
    const live = monitor.context_length ?? null;
    if (
      !monitor.active_model ||
      live == null ||
      !chat.residentCheckpoint ||
      chat.modelLoading ||
      chat.loadedContextLength == null
    ) {
      return false;
    }
    if (live === chat.loadedContextLength) {
      flagged = null;
      return false;
    }
    // Once per value per RETRY_MS, so a backend whose two reads disagree cannot drive a /status loop.
    const key = `${monitor.active_model}\n${live}`;
    const at = now();
    if (flagged?.key === key && at - flagged.at < RETRY_MS) return false;
    flagged = { key, at };
    return true;
  };
}
