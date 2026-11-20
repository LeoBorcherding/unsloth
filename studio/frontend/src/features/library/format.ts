// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { InterpolationValues, Locale, TranslationKey } from "@/i18n";
import { formatNumber } from "../../lib/intl-cache.ts";

type Translate = (key: TranslationKey, values?: InterpolationValues) => string;

const SIZE_UNITS = [
  "library.size.kilobytes",
  "library.size.megabytes",
  "library.size.gigabytes",
  "library.size.terabytes",
] as const satisfies readonly TranslationKey[];

export function formatSize(bytes: number | null, locale: Locale, t: Translate): string | null {
  if (bytes === null) return null;
  if (bytes < 1024) {
    return t("library.size.bytes", { value: formatNumber(bytes, locale, { useGrouping: false }) });
  }
  let value = bytes / 1024;
  let unit: (typeof SIZE_UNITS)[number] = SIZE_UNITS[0];
  for (const next of SIZE_UNITS.slice(1)) {
    if (value < 1024) break;
    value /= 1024;
    unit = next;
  }
  const digits = value >= 10 ? 0 : 1;
  const formatted = formatNumber(value, locale, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
  return t(unit, { value: formatted });
}

export function formatItemCount(count: number, t: Translate): string {
  return t(count === 1 ? "library.itemCountOne" : "library.itemCount", { count });
}
