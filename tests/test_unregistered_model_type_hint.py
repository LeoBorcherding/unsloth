# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import ast
from pathlib import Path

import pytest

LOADER = Path(__file__).resolve().parents[1] / "unsloth/models/loader.py"


def _transformers_error(model_type):
    # What transformers 5.5.4's AutoConfig raises for a model_type it does not register.
    return (
        f"The checkpoint you are trying to load has model type `{model_type}` but Transformers "
        "does not recognize this architecture. This could be because of an issue with the "
        "checkpoint, or because your version of Transformers is out of date."
    )


def _architecture_branches():
    """Each `if "architecture" in autoconfig_error:` body in loader.py, as a callable."""
    tree = ast.parse(LOADER.read_text(encoding = "utf-8"))
    branches = []
    for node in ast.walk(tree):
        test = getattr(node, "test", None)
        if not (
            isinstance(node, ast.If)
            and isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Constant)
            and test.left.value == "architecture"
            and isinstance(test.comparators[0], ast.Name)
            and test.comparators[0].id == "autoconfig_error"
        ):
            continue
        signature = "def branch(autoconfig_error, model_name, transformers_version): pass"
        fn = ast.parse(signature).body[0]
        fn.body = node.body
        env = {}
        module = ast.fix_missing_locations(ast.Module(body = [fn], type_ignores = []))
        exec(compile(module, str(LOADER), "exec"), env)
        branches.append(env["branch"])
    return branches


BRANCHES = _architecture_branches()


def test_both_loaders_have_the_branch():
    assert len(BRANCHES) == 2  # FastLanguageModel and FastModel


@pytest.mark.parametrize("branch", BRANCHES)
@pytest.mark.parametrize("model_type", ["inkling2_loop", "inkling2"])
def test_inkling2_names_the_required_transformers(branch, model_type):
    with pytest.raises(ImportError) as info:
        branch(_transformers_error(model_type), "unsloth/Inkling-2-Mini", "5.5.4")
    message = str(info.value)
    assert "Inkling 2" in message
    assert 'pip install "transformers==5.24.0"' in message
    assert "restart" in message


@pytest.mark.parametrize("branch", BRANCHES)
def test_qwen3_5_hint_unchanged(branch):
    with pytest.raises(ImportError, match = "does not support Qwen3.5"):
        branch(_transformers_error("qwen3_5"), "Qwen/Qwen3.5-4B", "4.57.1")


@pytest.mark.parametrize("branch", BRANCHES)
def test_other_model_types_keep_the_generic_error(branch):
    with pytest.raises(ValueError, match = "is not supported yet in `transformers==5.5.4`"):
        branch(_transformers_error("brand_new_arch"), "org/new-model", "5.5.4")


def test_error_text_matches_installed_transformers(tmp_path):
    """The branches key on "architecture" and the model_type, so the real AutoConfig text must carry both."""
    transformers = pytest.importorskip("transformers")
    (tmp_path / "config.json").write_text('{"model_type": "inkling2_loop"}')
    try:
        transformers.AutoConfig.from_pretrained(str(tmp_path))
    except ValueError as error:
        assert "architecture" in str(error) and "inkling2_loop" in str(error)
    else:
        pytest.skip(f"transformers {transformers.__version__} registers inkling2_loop")
