// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useEffect, useState } from "react";

export interface GpuBreakdownItem {
  kind: "chat" | "image" | "training" | "runtime";
  model: string | null;
  part: string | null;
  gb: number;
}

export interface GpuBreakdownDevice {
  index: number | null;
  /** Measured per process; null where this host has no per-process reading. */
  unsloth_gb: number | null;
  other_gb: number | null;
  items: GpuBreakdownItem[];
}

/** Per-GPU Unsloth vs other-app VRAM. Polled with the System page, never elsewhere: the probe costs a PowerShell call on Windows. */
export function useGpuBreakdown(pollMs?: number): GpuBreakdownDevice[] {
  const [devices, setDevices] = useState<GpuBreakdownDevice[]>([]);
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const res = await authFetch("/api/system/gpu-breakdown");
        if (!res.ok) return;
        const data = (await res.json()) as { devices?: GpuBreakdownDevice[] };
        if (!cancelled) setDevices(data.devices ?? []);
      } catch {
        // An older backend has no route; the rows keep their single bar.
      }
    };
    void load();
    if (!pollMs) return () => {
      cancelled = true;
    };
    const id = window.setInterval(load, pollMs);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [pollMs]);
  return devices;
}
