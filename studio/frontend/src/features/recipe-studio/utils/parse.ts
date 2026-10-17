// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function parseNumber(value?: string): number | null {
  if (!value) {
    return null;
  }
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

export function parseIntNumber(value?: string): number | null {
  const num = parseNumber(value);
  if (num === null || !Number.isInteger(num)) {
    return null;
  }
  return num;
}

const UUID_SHORT = new Set(["short", "short_form"]);
const UUID_UPPER = new Set(["upper", "uppercase"]);

/**
 * Unique ID format: `short`, `upper` and `prefix:<text>` combine with commas; text that starts with
 * none of them is a bare prefix. Null when a list mixes the two.
 */
export function parseUuidFormat(raw: string): Record<string, unknown> | null {
  const value = raw.trim();
  const params: Record<string, unknown> = {};
  let rest = value;
  let first = true;
  while (rest) {
    const comma = rest.indexOf(",");
    const token = (comma === -1 ? rest : rest.slice(0, comma)).trim().toLowerCase();
    if (token.startsWith("prefix:")) {
      // Runs to the end, so a prefix can hold commas.
      const prefix = rest.trim().slice("prefix:".length).trim();
      if (prefix) {
        params.prefix = prefix;
      }
      return params;
    }
    if (UUID_SHORT.has(token)) {
      params.short_form = true;
    } else if (UUID_UPPER.has(token)) {
      params.uppercase = true;
    } else if (token !== "uuid4") {
      return first ? { prefix: value } : null;
    }
    first = false;
    rest = comma === -1 ? "" : rest.slice(comma + 1);
  }
  return params;
}

export function parseAgeRange(value?: string): [number, number] | null {
  if (!value) {
    return null;
  }
  const parts = value.split(/[^0-9.]+/).filter(Boolean);
  if (parts.length !== 2) {
    return null;
  }
  const min = Number(parts[0]);
  const max = Number(parts[1]);
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return null;
  }
  return [min, max];
}

export function parseJsonObject(
  value: string | undefined,
  label: string,
  errors: string[],
): Record<string, unknown> | undefined {
  if (!value || !value.trim()) {
    return undefined;
  }
  try {
    const parsed = JSON.parse(value);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    errors.push(`${label}: invalid JSON.`);
    return undefined;
  }
  errors.push(`${label}: must be a JSON object.`);
  return undefined;
}

export function isValidSex(value?: string): value is "Male" | "Female" {
  if (!value) {
    return false;
  }
  return value === "Male" || value === "Female";
}

