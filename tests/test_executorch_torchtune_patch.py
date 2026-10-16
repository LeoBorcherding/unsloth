# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""fix_executorch's torchtune stub must only stand in for a missing torchtune, never replace a real one."""

import importlib
import subprocess
import sys

import pytest

from unsloth import import_fixes


_MODELS_INIT = "from enum import Enum\n\nfrom torchtune.models.convert_weights import get_mapped_key\n"

# executorch/examples/models/__init__.py as older Unsloth versions left it
_OLD_PATCHED = r'''from enum import Enum


import sys
import types
import re
from typing import Any, Optional
def get_mapped_key(key: str, mapping_dict: dict[str, str]) -> str:
    try:
        # Checks if there is a layer # in the key
        if any(k.isdigit() for k in key.split(".")):
            # Replace layer number with "{}" to create key for lookup
            abstract_key = re.sub(r"(\.\d+)", ".{}", key)
            layer_num = re.search(r"\d+", key).group(0)
            new_key = mapping_dict[abstract_key]
            new_key = new_key.format(layer_num)
        else:
            new_key = mapping_dict[key]
    except KeyError as e:
        raise Exception(
            f'Error converting the state dict. Found unexpected key: "{key}". '
            "Please make sure you're loading a checkpoint with the right format. "
        ) from e

    return new_key

torchtune = types.ModuleType("torchtune")
torchtune.__path__ = []
models = types.ModuleType("torchtune.models")
models.__path__ = []
convert_weights = types.ModuleType("torchtune.models.convert_weights")
convert_weights.get_mapped_key = get_mapped_key
torchtune.models = models
models.convert_weights = convert_weights
sys.modules["torchtune"] = torchtune
sys.modules["torchtune.models"] = models
sys.modules["torchtune.models.convert_weights"] = convert_weights
from torchtune.models.convert_weights import get_mapped_key
'''


def _make_site(root, with_torchtune, models_init = _MODELS_INIT):
    models = root / "executorch" / "examples" / "models"
    models.mkdir(parents = True)
    (root / "executorch" / "__init__.py").write_text("")
    (root / "executorch" / "examples" / "__init__.py").write_text("")
    (models / "__init__.py").write_text(models_init, encoding = "utf-8")
    if with_torchtune:
        tt = root / "torchtune" / "models"
        tt.mkdir(parents = True)
        (root / "torchtune" / "__init__.py").write_text("REAL = True\n")
        (tt / "__init__.py").write_text("")
        (tt / "convert_weights.py").write_text("def get_mapped_key(key, mapping):\n    return 'real'\n")
        (tt / "llama.py").write_text("")
    return models / "__init__.py"


def _patch(monkeypatch, root):
    monkeypatch.syspath_prepend(str(root))
    importlib.invalidate_caches()
    import_fixes.fix_executorch()


def _run(root, code):
    prelude = f"import sys; sys.path.insert(0, {str(root)!r})\n"
    out = subprocess.run(
        [sys.executable, "-c", prelude + code], capture_output = True, text = True, timeout = 120
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


_CHECK_REAL = (
    "import executorch.examples.models as m\n"
    "import torchtune, torchtune.models.llama\n"
    "print(getattr(torchtune, 'REAL', False), m.get_mapped_key('a', {}))\n"
)


@pytest.mark.parametrize("torchtune_first", [False, True])
def test_a_real_torchtune_survives_importing_executorch_models(tmp_path, monkeypatch, torchtune_first):
    _make_site(tmp_path, with_torchtune = True)
    _patch(monkeypatch, tmp_path)
    code = ("import torchtune\n" if torchtune_first else "") + _CHECK_REAL
    assert _run(tmp_path, code) == "True real"


def test_without_torchtune_the_stub_still_supplies_get_mapped_key(tmp_path, monkeypatch):
    _make_site(tmp_path, with_torchtune = False)
    _patch(monkeypatch, tmp_path)
    code = "import executorch.examples.models as m\nprint(m.get_mapped_key('x', {'x': 'y'}))\n"
    assert _run(tmp_path, code) == "y"


def test_an_old_unconditional_patch_is_rewritten_once(tmp_path, monkeypatch):
    init = _make_site(tmp_path, with_torchtune = True, models_init = _OLD_PATCHED)
    _patch(monkeypatch, tmp_path)
    patched = init.read_text(encoding = "utf-8")
    assert 'convert_weights = convert_weights\nsys.modules["torchtune"] = torchtune' not in patched
    assert _run(tmp_path, _CHECK_REAL) == "True real"
    import_fixes.fix_executorch()
    assert init.read_text(encoding = "utf-8") == patched
