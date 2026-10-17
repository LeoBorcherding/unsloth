# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Check (POST /validate) must fail the recipes a run is certain to fail, and Formula output must match its input."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_PROVIDERS = {
    "model_providers": [
        {"name": "p", "endpoint": "http://127.0.0.1:9", "provider_type": "openai"}
    ],
    "model_configs": [{"alias": "m", "model": "x", "provider": "p", "inference_parameters": {}}],
}
_SAMPLER = {
    "column_type": "sampler",
    "name": "c",
    "sampler_type": "category",
    "params": {"values": [" alpha ", "beta"]},
}


def _validate(recipe: dict):
    pytest.importorskip("fastapi")
    pytest.importorskip("data_designer")
    from routes.data_recipe import validate as validate_module

    return validate_module.validate(SimpleNamespace(recipe = recipe), via_api_key = False)


def _codes(response) -> list:
    return [error.code for error in response.errors]


def _expression(name: str, expr: str, **extra) -> dict:
    return {"column_type": "expression", "name": name, "expr": expr, **extra}


def test_unknown_model_alias_fails_check() -> None:
    llm = {"column_type": "llm-text", "name": "t", "model_alias": "nope", "prompt": "say {{ c }}"}
    response = _validate({**_PROVIDERS, "columns": [_SAMPLER, llm]})

    assert response.valid is False
    assert "unknown_model_alias" in _codes(response)
    assert "'nope'" in response.errors[0].message


def test_known_model_alias_passes_check() -> None:
    llm = {"column_type": "llm-text", "name": "t", "model_alias": "m", "prompt": "say {{ c }}"}
    assert _validate({**_PROVIDERS, "columns": [_SAMPLER, llm]}).valid is True


@pytest.mark.parametrize("loop", [["a", "b"], ["a", "b", "d"]])
def test_reference_loop_fails_check(loop: list) -> None:
    columns = [
        _expression(name, "{{ " + loop[index - 1] + " }}") for index, name in enumerate(loop)
    ]
    response = _validate({"columns": [_SAMPLER, *columns]})

    assert response.valid is False
    [error] = [e for e in response.errors if e.code == "circular_reference"]
    assert "loop (" in error.message


def test_duplicate_block_names_fail_check() -> None:
    response = _validate(
        {"columns": [_SAMPLER, _expression("e", "{{ c }}"), _expression("e", "{{ c }}!")]}
    )

    assert response.valid is False
    assert "duplicate_column_name" in _codes(response)


@pytest.mark.parametrize(
    "expr, banned",
    [
        ("{{ c | upper }}", "upper"),
        ("{{ c | default('x') }}", "default"),
        ("{{ [c, c] | join('-') }}", "join"),
        ("{{ c | tojson }}", "tojson"),
        ("{% filter upper %}{{ c }}{% endfilter %}", "upper"),
    ],
)
def test_filters_the_engine_refuses_fail_check(expr: str, banned: str) -> None:
    response = _validate({"columns": [_SAMPLER, _expression("e", expr)]})

    assert response.valid is False
    [error] = [e for e in response.errors if e.code == "unsupported_filter"]
    assert f"`{banned}`" in error.message
    assert error.path == "e"


def test_filters_in_a_prompt_fail_check() -> None:
    llm = {"column_type": "llm-text", "name": "t", "model_alias": "m", "prompt": "{{ c | upper }}"}
    response = _validate({**_PROVIDERS, "columns": [_SAMPLER, llm]})

    assert "unsupported_filter" in _codes(response)


def test_allowed_filters_pass_check() -> None:
    assert _validate({"columns": [_SAMPLER, _expression("e", "{{ c | trim | title }}")]}).valid


def _github(expr: str) -> dict:
    return {
        "seed_config": {
            "source": {
                "seed_type": "github_repo",
                "repos": ["octo/repo"],
                "item_types": ["issues"],
                "limit": 5,
                "token": "",
            },
        },
        "columns": [_expression("e", expr)],
    }


def test_github_seed_bad_reference_fails_check() -> None:
    pytest.importorskip("data_designer_github_repo_seed")
    response = _validate(_github("{{ titel }}"))

    assert response.valid is False
    assert "expression_reference_missing" in _codes(response)


def test_github_seed_runs_the_other_checks_too() -> None:
    pytest.importorskip("data_designer_github_repo_seed")
    assert "unsupported_filter" in _codes(_validate(_github("{{ title | upper }}")))


def test_github_seed_block_named_like_a_seed_column_fails_check() -> None:
    pytest.importorskip("data_designer_github_repo_seed")
    recipe = _github("{{ body }}")
    recipe["columns"][0]["name"] = "title"
    response = _validate(recipe)

    assert response.valid is False
    assert "collide" in response.errors[0].message


def test_github_seed_good_reference_passes_check() -> None:
    pytest.importorskip("data_designer_github_repo_seed")
    response = _validate(_github("{{ title }}: {{ body }}"))

    assert response.valid is True
    assert "checked when the run starts" in response.raw_detail


def test_github_seed_columns_match_the_scraper(monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from routes.data_recipe.validate import _GITHUB_SEED_COLUMNS

    path = (
        Path(__file__).resolve().parents[1]
        / "plugins/data-designer-github-repo-seed/src/data_designer_github_repo_seed/scraper.py"
    )
    spec = importlib.util.spec_from_file_location("_github_seed_scraper", path)
    scraper = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, scraper)
    spec.loader.exec_module(scraper)

    for row in (
        scraper._flatten_issue_row({}, "o/r", False, 0),
        scraper._flatten_pr_row({}, "o/r", False, 0),
        scraper._flatten_commit_row({}, "o/r"),
    ):
        assert tuple(row) == _GITHUB_SEED_COLUMNS


def test_access_refusal_reads_without_status_prefix(monkeypatch) -> None:
    pytest.importorskip("data_designer")
    from fastapi import HTTPException

    import core.data_recipe.service as service

    def refuse(_recipe):
        raise HTTPException(status_code = 403, detail = "Supply your own seed-source token")

    monkeypatch.setattr(service, "validate_recipe_access", refuse)
    response = _validate({"columns": [_SAMPLER, _expression("e", "{{ c }}")]})

    assert response.valid is False
    assert [e.message for e in response.errors] == ["Supply your own seed-source token"]
    assert response.raw_detail == "Supply your own seed-source token"


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


def test_formula_copy_of_json_text_stays_json(tmp_path: Path) -> None:
    js = '{"role": "user", "ok": true, "text": "café"}'
    dataset = _preview(
        tmp_path,
        [{"js": js}],
        [_expression("copy", "{{ js }}"), _expression("wrapped", "payload={{ js }}")],
    )

    assert json.loads(dataset["copy"][0]) == json.loads(js)
    assert dataset["wrapped"][0] == "payload=" + json.dumps(json.loads(js), ensure_ascii = False)


@pytest.mark.parametrize("value, dtype", [("x", "int"), ("", "int"), ("x", "float"), ("", "bool")])
def test_uncastable_formula_names_the_value(tmp_path: Path, value: str, dtype: str) -> None:
    with pytest.raises(Exception, match = f"Formula 'out' produced {value!r}, which is not"):
        _preview(tmp_path, [{"v": value}], [_expression("out", "{{ v }}", dtype = dtype)])


def test_castable_formula_still_casts(tmp_path: Path) -> None:
    dataset = _preview(tmp_path, [{"v": "7"}], [_expression("out", "{{ v }}", dtype = "int")])
    assert int(dataset["out"][0]) == 7
