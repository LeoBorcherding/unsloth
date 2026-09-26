// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth/api";

export interface LinkedInstance {
  id: string;
  name: string;
  base_url: string;
  created_at: string;
  updated_at: string;
}

export interface LinkedInstanceStatus {
  id: string;
  online: boolean;
  error: string | null;
  models: string[];
}

async function detail(res: Response, fallback: string): Promise<Error> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return new Error(body.detail);
  } catch {
    // not JSON
  }
  return new Error(fallback);
}

export async function fetchLinkedInstances(): Promise<LinkedInstance[]> {
  const res = await authFetch("/api/linked-instances");
  if (!res.ok) throw await detail(res, "Failed to load linked instances");
  return res.json();
}

export async function createLinkedInstance(input: {
  name: string;
  base_url: string;
  api_key: string;
}): Promise<LinkedInstance> {
  const res = await authFetch("/api/linked-instances", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) throw await detail(res, "Failed to link instance");
  return res.json();
}

export async function deleteLinkedInstance(id: string): Promise<void> {
  const res = await authFetch(`/api/linked-instances/${id}`, {
    method: "DELETE",
  });
  if (!res.ok) throw await detail(res, "Failed to remove linked instance");
}

export async function testLinkedInstance(
  id: string,
): Promise<LinkedInstanceStatus> {
  const res = await authFetch(`/api/linked-instances/${id}/test`, {
    method: "POST",
  });
  if (!res.ok) throw await detail(res, "Failed to reach linked instance");
  return res.json();
}
