// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IpAdapterSpecInput } from "../api";

export const DEFAULT_IP_ADAPTER_SCALE = 0.6;

interface IpAdapterStatus {
  loaded?: boolean;
  supports_ip_adapter?: boolean;
}

interface IpAdapterRecord {
  ip_adapter?: string | null;
  ip_adapter_scale?: number | null;
}

/** The slot shows for a loaded model that takes an image prompt (SDXL), never for a Connection. */
export function ipAdapterCapable(status: IpAdapterStatus | null | undefined, external: boolean): boolean {
  return !external && Boolean(status?.loaded && status.supports_ip_adapter);
}

/** Sent only on Create with an image and a non-zero scale, so a plain run's request is unchanged. */
export function ipAdapterRequest(
  capable: boolean,
  workflow: string,
  image: string | null,
  scale: number,
): IpAdapterSpecInput | undefined {
  if (!capable || workflow !== "create" || !image || !(scale > 0)) return undefined;
  return { image, scale: Math.min(1, scale) };
}

/** The scale a recipe ran its image prompt at, or null when it had none. */
export function ipAdapterScaleFromRecipe(image: IpAdapterRecord): number | null {
  if (!image.ip_adapter) return null;
  const scale = image.ip_adapter_scale;
  return typeof scale === "number" && Number.isFinite(scale) && scale > 0 && scale <= 1 ? scale : null;
}

/** Recipe row text: the adapter and its scale. */
export function ipAdapterRecipeLabel(image: IpAdapterRecord): string | null {
  if (!image.ip_adapter) return null;
  const scale = ipAdapterScaleFromRecipe(image);
  return scale === null ? image.ip_adapter : `${image.ip_adapter} at ${scale}`;
}

/** Joins the restore toast's "add X again" list, adding the image prompt when the recipe used one. */
export function withIpAdapterInput(inputs: string | null | undefined, image: IpAdapterRecord): string | null {
  if (!image.ip_adapter) return inputs || null;
  return inputs ? `${inputs} and the image prompt` : "the image prompt";
}
