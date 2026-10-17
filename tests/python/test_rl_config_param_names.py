# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The patched configs' checks key on whole parameter names, not substrings of them."""

from __future__ import annotations

import pytest


pytest.importorskip("trl")


@pytest.mark.parametrize("name", ["DDPOConfig", "AlignPropConfig"])
def test_a_config_with_only_train_learning_rate_builds(name):
    import unsloth  # noqa: F401
    import trl

    config_class = getattr(trl, name, None)
    if not isinstance(config_class, type):
        pytest.skip(f"this TRL has no {name}")
    assert getattr(config_class, "_unsloth_patched_rl_config", False), f"{name} was not patched"

    config = config_class()
    assert config.train_learning_rate > 0


def test_a_config_with_learning_rate_keeps_the_check(tmp_path, capsys):
    import unsloth  # noqa: F401
    import trl

    trl.SFTConfig(
        output_dir = str(tmp_path),
        learning_rate = 1e-9,
        bf16 = False,
        fp16 = False,
        use_cpu = True,
    )
    assert "is too small and less than 1e-7" in capsys.readouterr().out
