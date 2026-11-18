// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** /api/system's `gpu.other_vendor_devices`: a card of the vendor this PyTorch cannot drive
 * (an NVIDIA + AMD host). Display-only: `index` is the probe's row number, not a pin. */
export interface OtherVendorGpuDevice {
  vendor?: string;
  backend?: string;
  index?: number;
  qualified_id?: string | null;
  name?: string | null;
  memory_total_gb?: number | null;
  trainable?: boolean;
}

export interface OtherVendorGpuRow {
  key: string;
  name: string | null;
  vendorLabel: string;
  qualifiedId: string | null;
  memoryTotalGb: number | null;
}

const VENDOR_LABELS: Record<string, string> = { nvidia: "NVIDIA", amd: "AMD" };
const BACKEND_LABELS: Record<string, string> = { cuda: "CUDA", rocm: "ROCm" };

export function backendLabel(backend: string | null | undefined): string {
  return (backend && BACKEND_LABELS[backend]) ?? backend?.toUpperCase() ?? "GPU";
}

export function otherVendorGpuRows(
  devices: OtherVendorGpuDevice[] | null | undefined,
): OtherVendorGpuRow[] {
  return (devices ?? []).map((device, index) => ({
    key: device.qualified_id ?? `${device.vendor ?? "gpu"}-${device.index ?? index}`,
    name: device.name ?? null,
    vendorLabel:
      (device.vendor && VENDOR_LABELS[device.vendor]) ?? device.vendor?.toUpperCase() ?? "",
    qualifiedId: device.qualified_id ?? null,
    memoryTotalGb:
      typeof device.memory_total_gb === "number" && Number.isFinite(device.memory_total_gb)
        ? device.memory_total_gb
        : null,
  }));
}
