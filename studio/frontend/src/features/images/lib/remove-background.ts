// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Remove background: SAM 3 keeps the subject and the rest turns transparent, saved as a new gallery image. Pure logic
// only (no "@/" imports) so node:test can load it directly.

import type { GalleryImage } from "../api";

export const REMOVE_BACKGROUND_ROUTE = "/api/inference/images/remove-background";
export const REMOVE_BACKGROUND_LABEL = "Remove background";

export interface RemoveBackgroundRequest {
  image?: string;
  image_id?: string;
  /** What to keep; empty = the main subject. */
  prompt?: string;
  threshold?: number;
  grow?: number;
  feather?: number;
  trim?: boolean;
  save?: boolean;
}

/** The saved record. Throws when the answer has none, since that is what the page shows. */
export function parseRemoveBackgroundResult(raw: unknown): GalleryImage {
  const body = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  const image = body.image as Partial<GalleryImage> | null | undefined;
  if (!image || typeof image.id !== "string" || typeof image.url !== "string") {
    throw new Error("Removing the background returned no image.");
  }
  return image as GalleryImage;
}

/** A 409 that means the segmenter weights are missing, so the toast can offer the download. */
export function segmenterMissing(message: string): boolean {
  return /not downloaded/i.test(message);
}
