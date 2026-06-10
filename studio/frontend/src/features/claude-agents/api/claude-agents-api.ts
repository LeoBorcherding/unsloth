// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export interface AgentScript {
  id: string;
  name: string;
  description: string;
  promptTemplate: string;
  allowedTools: string;
  permissionMode: string;
  cwd: string;
  workflow: string | null;
  prebuilt: boolean;
  createdAt: number;
  updatedAt: number;
}

export interface RunAgentRequest {
  agentId?: string;
  promptTemplate?: string;
  variables?: Record<string, string>;
  allowedTools?: string;
  permissionMode?: string;
  cwd?: string;
  sessionId?: string;
  includeAgentCatalog?: boolean;
}

/** One line of Claude Code `stream-json` output, or a studio wrapper event. */
export interface ClaudeStreamEvent {
  type: string;
  subtype?: string;
  // studio wrapper events
  runId?: string;
  exitCode?: number;
  error?: string;
  stderr?: string;
  // system/init
  session_id?: string;
  model?: string;
  // assistant / user message snapshots
  message?: {
    role?: string;
    content?: Array<{
      type: string;
      text?: string;
      name?: string;
      input?: Record<string, unknown>;
    }>;
  };
  // stream_event partial deltas
  event?: {
    type?: string;
    delta?: { type?: string; text?: string };
  };
  // result
  result?: string;
  total_cost_usd?: number;
  num_turns?: number;
}

export function newAgentDraft(): AgentScript {
  return {
    id: "",
    name: "",
    description: "",
    promptTemplate: "",
    allowedTools: "Read,Glob,Grep",
    permissionMode: "default",
    cwd: "",
    workflow: null,
    prebuilt: false,
    createdAt: 0,
    updatedAt: 0,
  };
}

export const PLACEHOLDER_RE = /\{([A-Za-z_][A-Za-z0-9_]*)\}/g;

export function extractPlaceholders(template: string): string[] {
  const names: string[] = [];
  for (const match of template.matchAll(PLACEHOLDER_RE)) {
    if (!names.includes(match[1])) names.push(match[1]);
  }
  return names;
}

async function parseJsonOrThrow<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = (body as { detail?: string } | null)?.detail;
    throw new Error(detail ?? `Request failed (${res.status})`);
  }
  return body as T;
}

export async function getClaudeStatus(): Promise<{
  available: boolean;
  version: string | null;
}> {
  const res = await authFetch("/api/claude-agents/status");
  return parseJsonOrThrow(res);
}

export async function listAgentScripts(): Promise<AgentScript[]> {
  const res = await authFetch("/api/claude-agents/agents");
  const data = await parseJsonOrThrow<{ agents: AgentScript[] }>(res);
  return data.agents;
}

export async function saveAgentScript(agent: AgentScript): Promise<AgentScript> {
  const res = await authFetch(`/api/claude-agents/agents/${agent.id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(agent),
  });
  return parseJsonOrThrow<AgentScript>(res);
}

export async function deleteAgentScript(id: string): Promise<void> {
  const res = await authFetch(`/api/claude-agents/agents/${id}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`Delete failed (${res.status})`);
}

export async function cancelClaudeRun(runId: string): Promise<void> {
  await authFetch(`/api/claude-agents/runs/${runId}/cancel`, { method: "POST" });
}

/**
 * Start a run and stream events back. Resolves when the stream closes.
 * The X-Run-Id header (also sent as a run_started event) identifies the run
 * for cancellation.
 */
export async function streamClaudeRun(
  req: RunAgentRequest,
  onEvent: (event: ClaudeStreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await authFetch("/api/claude-agents/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => null);
    const detail = (body as { detail?: string } | null)?.detail;
    throw new Error(detail ?? `Run failed (${res.status})`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep = buffer.indexOf("\n\n");
    while (sep !== -1) {
      const chunk = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      sep = buffer.indexOf("\n\n");
      const line = chunk.startsWith("data: ") ? chunk.slice(6) : chunk;
      if (!line.trim() || line.trim() === "[DONE]") continue;
      try {
        onEvent(JSON.parse(line) as ClaudeStreamEvent);
      } catch {
        // Ignore non-JSON keepalive lines.
      }
    }
  }
}
