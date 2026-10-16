// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { buildSamplerColumn } = await import(
  "../src/features/recipe-studio/utils/payload/builders-sampler.ts"
);

type SamplerConfig = Parameters<typeof buildSamplerColumn>[0];
type Lookup = Parameters<typeof buildSamplerColumn>[2];

// With no format the engine prints only the year when every sampled date falls in one month,
// so a March 1-20 range came out as "2024" in every row.
const date = (unit: string): SamplerConfig =>
  ({
    id: "d",
    kind: "sampler",
    sampler_type: "datetime",
    name: "d",
    datetime_start: "2024-03-01",
    datetime_end: "2024-03-20",
    datetime_unit: unit,
  }) as SamplerConfig;

test("a Date block sends a format that matches its unit", () => {
  const formats = Object.fromEntries(
    ["year", "month", "day", "hour", "minute", "second"].map((unit) => [
      unit,
      buildSamplerColumn(date(unit), []).convert_to,
    ]),
  );
  assert.deepEqual(formats, {
    year: "%Y",
    month: "%Y-%m",
    day: "%Y-%m-%d",
    hour: "%Y-%m-%dT%H:%M",
    minute: "%Y-%m-%dT%H:%M",
    second: "%Y-%m-%dT%H:%M:%S",
  });
});

test("a Time offset prints at the finer of its own unit and its reference date's", () => {
  const offset = (unit: "D" | "h" | "m" | "s"): SamplerConfig =>
    ({
      id: "t",
      kind: "sampler",
      sampler_type: "timedelta",
      name: "t",
      dt_min: "0",
      dt_max: "3",
      reference_column_name: "d",
      timedelta_unit: unit,
    }) as SamplerConfig;
  const lookup = (unit: string) => new Map([["d", date(unit)]]) as Lookup;
  assert.equal(buildSamplerColumn(offset("D"), [], lookup("day")).convert_to, "%Y-%m-%d");
  assert.equal(buildSamplerColumn(offset("D"), [], lookup("minute")).convert_to, "%Y-%m-%dT%H:%M");
  assert.equal(buildSamplerColumn(offset("s"), [], lookup("day")).convert_to, "%Y-%m-%dT%H:%M:%S");
  assert.equal(buildSamplerColumn(offset("D"), []).convert_to, "%Y-%m-%d");
});

test("numeric samplers keep their own conversion", () => {
  const column = buildSamplerColumn(
    {
      id: "u",
      kind: "sampler",
      sampler_type: "uniform",
      name: "u",
      low: "1",
      high: "10",
      convert_to: "int",
    } as SamplerConfig,
    [],
  );
  assert.equal(column.convert_to, "int");
});
