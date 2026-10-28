# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import logging
from pathlib import Path

import pytest


def _recipe(seed: Path) -> dict:
    return {
        "seed_config": {
            "source": {"seed_type": "local", "path": str(seed)},
            "sampling_strategy": "ordered",
        },
        "columns": [
            {"column_type": "expression", "name": "copy", "expr": "{{ chunk_text }}", "dtype": "str"}
        ],
    }


@pytest.fixture(name = "records")
def fixture_records():
    # The worker listens on the named logger the same way: DataDesigner stops it propagating to root.
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger("data_designer")
    logger.addHandler(handler)
    yield records
    logger.removeHandler(handler)


def _repeat_warnings(records) -> list[str]:
    return [
        r.getMessage()
        for r in records
        if r.levelno == logging.WARNING and "more rows than the seed data has" in r.getMessage()
    ]


def test_rows_past_the_seed_size_are_reused_with_a_warning(tmp_path: Path, records) -> None:
    pytest.importorskip("data_designer")
    from core.data_recipe.service import build_config_builder, create_data_designer

    seed = tmp_path / "chunks.csv"
    seed.write_text("chunk_text\nfirst\nsecond\n", encoding = "utf-8")
    recipe = _recipe(seed)

    designer = create_data_designer(recipe, artifact_path = str(tmp_path / "artifacts"))
    result = designer.preview(build_config_builder(recipe), num_records = 5)

    assert list(result.dataset["copy"]) == ["first", "second", "first", "second", "first"]
    warnings = _repeat_warnings(records)
    assert len(warnings) == 1, warnings
    assert "(2)" in warnings[0]


def test_no_warning_when_the_seed_covers_the_run(tmp_path: Path, records) -> None:
    pytest.importorskip("data_designer")
    from core.data_recipe.service import build_config_builder, create_data_designer

    seed = tmp_path / "chunks.csv"
    seed.write_text("chunk_text\nfirst\nsecond\nthird\n", encoding = "utf-8")
    recipe = _recipe(seed)

    designer = create_data_designer(recipe, artifact_path = str(tmp_path / "artifacts"))
    designer.preview(build_config_builder(recipe), num_records = 3)

    assert _repeat_warnings(records) == []
