// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { formatDate } from "../lib/intl-cache.ts";
import { withTimeFormat } from "../lib/time-format.ts";
import type { Locale, TranslationKey } from "./messages";
import type { InterpolationValues } from "./types";

type Translate = (key: TranslationKey, values?: InterpolationValues) => string;

const relativeTimeFormatters = new Map<string, Intl.RelativeTimeFormat>();

function getFormatter(
  locale: Locale,
  style: "short" | "long",
): Intl.RelativeTimeFormat {
  const key = `${locale}:${style}`;
  let formatter = relativeTimeFormatters.get(key);
  if (!formatter) {
    formatter = new Intl.RelativeTimeFormat(locale, {
      numeric: "always",
      style,
    });
    relativeTimeFormatters.set(key, formatter);
  }
  return formatter;
}

export function formatRelativeTime(
  locale: Locale,
  value: number,
  unit: Intl.RelativeTimeFormatUnit,
): string {
  // format() throws on NaN, and a throw during commit would unmount the tree.
  if (!Number.isFinite(value)) {
    return "";
  }
  const short = getFormatter(locale, "short");
  const formatted = short.format(value, unit);
  // Some CLDR short patterns drop the tense marker (Arabic "few" months); fall back to long style.
  if (value !== 0 && formatted === short.format(-value, unit)) {
    return getFormatter(locale, "long").format(value, unit);
  }
  return formatted;
}

const yesterdayFormatters = new Map<Locale, Intl.RelativeTimeFormat>();

function yesterday(locale: Locale): string {
  let formatter = yesterdayFormatters.get(locale);
  if (!formatter) {
    formatter = new Intl.RelativeTimeFormat(locale, { numeric: "auto" });
    yesterdayFormatters.set(locale, formatter);
  }
  return formatter.format(-1, "day");
}

const DAY_MS = 86_400_000;

function startOfDay(date: Date): number {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
}

export function formatCardTime(ts: number, locale: Locale, now: number = Date.now()): string {
  if (!Number.isFinite(ts)) return "";
  const then = new Date(ts);
  const today = new Date(now);
  // Rounded: DST days are 23 or 25 hours long.
  const days = Math.round((startOfDay(today) - startOfDay(then)) / DAY_MS);
  if (days === 0) {
    return formatDate(then, locale, withTimeFormat({ hour: "numeric", minute: "2-digit" }, locale));
  }
  if (days === 1) {
    const word = yesterday(locale);
    return word.charAt(0).toLocaleUpperCase(locale) + word.slice(1);
  }
  if (days > 1 && days < 7) return formatDate(then, locale, { weekday: "long" });
  return formatDate(then, locale, {
    month: "short",
    day: "numeric",
    year: then.getFullYear() === today.getFullYear() ? undefined : "numeric",
  });
}

export function formatActivityTime(ts: number, locale: Locale, t: Translate): string {
  const elapsed = Date.now() - ts;
  if (elapsed < -60_000) return formatCardTime(ts, locale);
  const minutes = Math.floor(elapsed / 60_000);
  if (minutes < 1) return t("common.justNow");
  if (minutes < 60) return formatRelativeTime(locale, -minutes, "minute");
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return formatRelativeTime(locale, -hours, "hour");
  const days = Math.round((startOfDay(new Date()) - startOfDay(new Date(ts))) / DAY_MS);
  if (days <= 1) return yesterday(locale);
  if (days < 7) return formatRelativeTime(locale, -days, "day");
  return formatCardTime(ts, locale);
}
