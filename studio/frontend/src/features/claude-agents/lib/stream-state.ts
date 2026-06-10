// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ClaudeStreamEvent } from "../api/claude-agents-api";

/** Display blocks distilled from a Claude Code stream-json event stream. */
export type StreamBlock =
  | { kind: "text"; text: string }
  | { kind: "tool"; name: string; summary: string }
  | { kind: "error"; text: string };

export interface StreamState {
  blocks: StreamBlock[];
  /** Text accumulated from partial deltas since the last full snapshot. */
  liveText: string;
  sessionId: string | null;
  model: string | null;
  running: boolean;
  exitCode: number | null;
  costUsd: number | null;
  /** Final result text from the `result` event. */
  resultText: string | null;
}

export function emptyStreamState(): StreamState {
  return {
    blocks: [],
    liveText: "",
    sessionId: null,
    model: null,
    running: false,
    exitCode: null,
    costUsd: null,
    resultText: null,
  };
}

function toolSummary(input: Record<string, unknown> | undefined): string {
  if (!input) {
    return "";
  }
  const value =
    input.file_path ?? input.path ?? input.command ?? input.pattern ?? input.url;
  if (typeof value === "string") {
    return value.length > 120 ? `${value.slice(0, 117)}...` : value;
  }
  return "";
}

/** Fold one stream event into the display state. Returns a new state object. */
export function reduceStreamEvent(
  state: StreamState,
  event: ClaudeStreamEvent,
): StreamState {
  if (event.type === "system" && event.subtype === "init") {
    return {
      ...state,
      sessionId: event.session_id ?? state.sessionId,
      model: event.model ?? state.model,
    };
  }
  if (event.type === "stream_event") {
    const delta = event.event?.delta;
    if (delta?.type === "text_delta" && typeof delta.text === "string") {
      return { ...state, liveText: state.liveText + delta.text };
    }
    return state;
  }
  if (event.type === "assistant" && event.message?.content) {
    // Full message snapshot supersedes the live partial buffer.
    const blocks = [...state.blocks];
    for (const part of event.message.content) {
      if (part.type === "text" && part.text) {
        blocks.push({ kind: "text", text: part.text });
      } else if (part.type === "tool_use" && part.name) {
        blocks.push({
          kind: "tool",
          name: part.name,
          summary: toolSummary(part.input),
        });
      }
    }
    return { ...state, blocks, liveText: "" };
  }
  if (event.type === "result") {
    return {
      ...state,
      liveText: "",
      resultText: typeof event.result === "string" ? event.result : null,
      costUsd: event.total_cost_usd ?? state.costUsd,
    };
  }
  if (event.type === "studio") {
    if (event.subtype === "run_started") {
      return { ...emptyStreamState(), running: true };
    }
    if (event.subtype === "run_finished") {
      const blocks = [...state.blocks];
      const failed = event.exitCode != null && event.exitCode !== 0;
      if (failed) {
        blocks.push({
          kind: "error",
          text:
            event.error ??
            `claude exited with code ${event.exitCode}${event.stderr ? `\n${event.stderr}` : ""}`,
        });
      }
      return {
        ...state,
        blocks,
        running: false,
        exitCode: event.exitCode ?? null,
      };
    }
    if (event.subtype === "run_error") {
      return {
        ...state,
        blocks: [
          ...state.blocks,
          { kind: "error", text: event.error ?? "Run failed to start." },
        ],
        running: false,
      };
    }
  }
  return state;
}
