// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// React-free so the node --test suite can import it.

import type { ApiMonitorResponse } from "@/features/chat/types/api";

type ChatResidentWindow = {
  residentCheckpoint: string | null | undefined;
  loadedContextLength: number | null;
  modelLoading: boolean;
};

/**
 * True once per drift: the monitor reads the live window every poll, the chat store only on its own
 * loads and focus, so an API reload of the same model at a new context would leave the header stale.
 */
export function createResidentDriftWatch(): (
  monitor: Pick<ApiMonitorResponse, "active_model" | "context_length">,
  chat: ChatResidentWindow,
) => boolean {
  let flagged: string | null = null;
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
    // Once per value, so a backend whose two reads disagree cannot drive a /status loop.
    const key = `${monitor.active_model}\n${live}`;
    if (key === flagged) return false;
    flagged = key;
    return true;
  };
}
