# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Code checks: broken Python fails, lint mode runs oxlint on every OS, a checker that can't run says so
instead of failing every row, and a JS/TS check refuses a language OXC can't parse."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.data_recipe import local_callable_validators as validators

_GOOD_PY = "def add(a, b):\n    return a + b\n"
_BAD_PY = "def add(a, b)\n    return a +\n"
_GOOD_JS = "function add(a, b) { return a + b; }"
_BAD_JS = "var unused = 1;\ndebugger;"

_HAS_OXLINT = (
    validators.resolve_node_executable() is not None
    and (validators._OXC_TOOL_DIR / "node_modules" / "oxlint" / "package.json").is_file()
)
real_oxlint = pytest.mark.skipif(
    not _HAS_OXLINT, reason = "needs Node and oxc-validator/node_modules/oxlint"
)


def _preview(tmp_path: Path, rows: list, columns: list):
    pytest.importorskip("data_designer")
    from core.data_recipe.service import build_config_builder, create_data_designer

    seed = tmp_path / "seed.jsonl"
    seed.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding = "utf-8")
    recipe = {
        "seed_config": {
            "source": {"seed_type": "local", "path": str(seed)},
            "sampling_strategy": "ordered",
        },
        "columns": columns,
    }
    designer = create_data_designer(recipe, artifact_path = str(tmp_path / "artifacts"))
    return designer.preview(build_config_builder(recipe), num_records = len(rows)).dataset


def test_python_check_fails_broken_python(tmp_path: Path) -> None:
    pytest.importorskip("ruff")
    dataset = _preview(
        tmp_path,
        [{"snippet": _BAD_PY}, {"snippet": _GOOD_PY}],
        [
            {"column_type": "expression", "name": "code", "expr": "{{ snippet }}"},
            {
                "column_type": "validation",
                "name": "chk",
                "target_columns": ["code"],
                "validator_type": "code",
                "validator_params": {"code_lang": "python"},
            },
        ],
    )

    by_code = dict(zip(dataset["code"], dataset["chk"]))
    assert by_code[_BAD_PY]["is_valid"] is False
    assert by_code[_BAD_PY]["python_linter_severity"] == "fatal"
    assert by_code[_GOOD_PY]["is_valid"] is True


@pytest.fixture(name = "oxc_tmp")
def fixture_oxc_tmp(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(validators, "oxc_validator_tmp_root", lambda: tmp_path)
    return tmp_path


@real_oxlint
@pytest.mark.parametrize("mode", ["lint", "syntax+lint"])
def test_lint_mode_runs_oxlint(oxc_tmp: Path, mode: str) -> None:
    good, bad = validators._run_oxc_batch(
        node_lang = "js",
        validation_mode = mode,
        code_shape = "auto",
        code_values = [_GOOD_JS, _BAD_JS],
    )

    assert good["is_valid"] is True, good["error_message"]
    assert good["error_message"] == ""
    # oxlint's default rules warn; the point is that it ran and found the debugger statement.
    assert bad["warning_count"] >= 1
    assert "no-debugger" in bad["error_message"]


@real_oxlint
def test_oxlint_that_cannot_run_leaves_rows_unchecked(tmp_path: Path) -> None:
    payload = {"lang": "js", "mode": "lint", "codes": [_GOOD_JS, _BAD_JS], "timeout_ms": 1}
    proc = subprocess.run(
        [validators.resolve_node_executable(), str(validators._OXC_RUNNER_PATH)],
        cwd = str(validators._OXC_TOOL_DIR),
        input = json.dumps(payload),
        capture_output = True,
        text = True,
        encoding = "utf-8",
        timeout = 60,
    )

    results = json.loads(proc.stdout)
    assert [r["is_valid"] for r in results] == [None, None]
    assert all(r["error_message"].startswith("Code check did not run: ") for r in results)


def test_missing_node_leaves_rows_unchecked(monkeypatch) -> None:
    monkeypatch.setattr(validators, "resolve_node_executable", lambda: None)

    results = validators._run_oxc_batch(
        node_lang = "js",
        validation_mode = "syntax",
        code_shape = "auto",
        code_values = [_GOOD_JS, _BAD_JS],
    )

    assert [r["is_valid"] for r in results] == [None, None]
    assert [r["error_count"] for r in results] == [0, 0]
    assert all(r["error_message"].startswith("Code check did not run: Node.js not found") for r in results)


def _oxc_column(lang: str) -> dict:
    return {
        "column_type": "validation",
        "name": "chk",
        "target_columns": ["code"],
        "validator_type": "local_callable",
        "validator_params": {"validation_function": f"unsloth_oxc_validator:{lang}:syntax:auto"},
    }


def test_unknown_oxc_language_is_refused() -> None:
    with pytest.raises(ValueError, match = "can't check python code"):
        validators.split_oxc_local_callable_validators({"columns": [_oxc_column("python")]})


def test_known_oxc_language_still_parses() -> None:
    _, [spec] = validators.split_oxc_local_callable_validators({"columns": [_oxc_column("tsx")]})
    assert spec.code_lang == "tsx"


def test_unknown_oxc_language_fails_check() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("data_designer")
    from routes.data_recipe import validate as validate_module

    recipe = {
        "columns": [
            {
                "column_type": "sampler",
                "name": "src",
                "sampler_type": "category",
                "params": {"values": ["print(1)"]},
            },
            {"column_type": "expression", "name": "code", "expr": "{{ src }}"},
            _oxc_column("python"),
        ]
    }
    response = validate_module.validate(SimpleNamespace(recipe = recipe), via_api_key = False)

    assert response.valid is False
    assert "can't check python code" in response.errors[0].message
