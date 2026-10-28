// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { buildSamplerColumn } = await import(
  "../src/features/recipe-studio/utils/payload/builders-sampler.ts"
);
const { parseSampler } = await import(
  "../src/features/recipe-studio/utils/import/parsers/sampler-parser.ts"
);
const { getConfigErrors } = await import(
  "../src/features/recipe-studio/utils/validation.ts"
);

type SamplerConfig = Parameters<typeof buildSamplerColumn>[0];

function sampler(patch: Partial<SamplerConfig>): SamplerConfig {
  return { id: "s-1", kind: "sampler", name: "s", sampler_type: "uniform", ...patch } as SamplerConfig;
}

// Each case here passed the editor and Check, then failed or misbehaved inside the engine.

test("a Date block with no range is caught before the run", () => {
  const errors = getConfigErrors(
    sampler({ sampler_type: "datetime", datetime_start: "", datetime_end: "", datetime_unit: "day" }),
  );
  assert.ok(errors.includes("Pick a start and end date."), JSON.stringify(errors));
  assert.deepEqual(
    getConfigErrors(
      sampler({
        sampler_type: "datetime",
        datetime_start: "2024-03-01",
        datetime_end: "2024-03-20",
        datetime_unit: "day",
      }),
    ),
    [],
  );
});

test("the Date unit list only offers units the builder accepts", () => {
  const dialog = readFileSync(
    new URL("../src/features/recipe-studio/dialogs/samplers/datetime-dialog.tsx", import.meta.url),
    "utf8",
  );
  const units = dialog.match(/const DATETIME_UNITS = \[([\s\S]*?)\]/)?.[1] ?? "";
  for (const unit of units.match(/"(\w+)"/g)?.map((u) => u.slice(1, -1)) ?? []) {
    const errors: string[] = [];
    buildSamplerColumn(
      sampler({ sampler_type: "datetime", datetime_start: "2024-01-01", datetime_end: "2024-12-31", datetime_unit: unit }),
      errors,
    );
    assert.deepEqual(errors, [], `unit ${unit}`);
  }
});

test("Time offset needs whole numbers, 0 or more", () => {
  for (const [min, max] of [["1.5", "3"], ["-2", "3"], ["0", "2.5"]]) {
    const errors = getConfigErrors(
      sampler({
        sampler_type: "timedelta",
        dt_min: min,
        dt_max: max,
        reference_column_name: "d",
        timedelta_unit: "D",
      }),
    );
    assert.ok(
      errors.includes("Timedelta dt_min/dt_max must be whole numbers, 0 or more."),
      `${min}..${max}: ${JSON.stringify(errors)}`,
    );
  }
});

test("a person age range is checked against what the engine allows", () => {
  for (const range of ["18.5-70", "70-18", "18-200"]) {
    const errors = getConfigErrors(sampler({ sampler_type: "person", person_age_range: range }));
    assert.equal(errors.length, 1, `${range}: ${JSON.stringify(errors)}`);
  }
  assert.deepEqual(getConfigErrors(sampler({ sampler_type: "person", person_age_range: "18-70" })), []);
});

test("Unique ID short and upper combine instead of becoming a prefix", () => {
  const column = buildSamplerColumn(sampler({ sampler_type: "uuid", uuid_format: "short, upper" }), []);
  assert.deepEqual(column.params, { short_form: true, uppercase: true });

  const withPrefix = buildSamplerColumn(
    sampler({ sampler_type: "uuid", uuid_format: "short,prefix:ID-" }),
    [],
  );
  assert.deepEqual(withPrefix.params, { short_form: true, prefix: "ID-" });

  const reopened = parseSampler(withPrefix, "s", "s-1", []);
  assert.deepEqual(buildSamplerColumn(reopened as SamplerConfig, []).params, withPrefix.params);
});

test("a Unique ID list with an unknown word is an error, not a prefix", () => {
  const errors: string[] = [];
  buildSamplerColumn(sampler({ sampler_type: "uuid", uuid_format: "short, uppr" }), errors);
  assert.equal(errors.length, 1, JSON.stringify(errors));
  assert.equal(getConfigErrors(sampler({ sampler_type: "uuid", uuid_format: "short, uppr" })).length, 1);
});

test("int Random number gives each whole number the same share", () => {
  const column = buildSamplerColumn(
    sampler({ sampler_type: "uniform", low: "1", high: "10", convert_to: "int" }),
    [],
  );
  const { low, high } = column.params as { low: number; high: number };
  // The engine draws U(low, high) and rounds, so each integer k owns [k - 0.5, k + 0.5).
  const share = (k: number) => (Math.min(k + 0.5, high) - Math.max(k - 0.5, low)) / (high - low);
  for (let k = 1; k <= 10; k++) {
    assert.ok(Math.abs(share(k) - 0.1) < 1e-9, `P(${k}) = ${share(k)}`);
  }

  const reopened = parseSampler(column, "s", "s-1", []) as SamplerConfig;
  assert.equal(reopened.low, "1");
  assert.equal(reopened.high, "10");
  assert.deepEqual(buildSamplerColumn(reopened, []).params, column.params);

  const float = buildSamplerColumn(sampler({ sampler_type: "uniform", low: "1", high: "10" }), []);
  assert.deepEqual(float.params, { low: 1, high: 10 });
});
