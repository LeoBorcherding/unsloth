# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import sys

import pytest

import unsloth_cli._inference as inf


@pytest.fixture
def clean_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("MLX_", "OMPI_", "PMI_", "PMIX_", "MPI_", "MV2_", "NCCL_")):
            monkeypatch.delenv(name, raising = False)
    return monkeypatch


@pytest.mark.parametrize("platform", ["linux", "win32"])
@pytest.mark.parametrize(
    "env",
    [
        {"PMI_RANK": "0", "PMI_SIZE": "2"},
        {"OMPI_COMM_WORLD_RANK": "1", "OMPI_COMM_WORLD_SIZE": "4"},
        {"MLX_RANK": "1", "MLX_WORLD_SIZE": "2", "NCCL_HOST_IP": "10.0.0.1", "NCCL_PORT": "5000"},
    ],
)
def test_mpi_env_off_macos_is_not_an_mlx_launch(clean_env, platform, env):
    # srun / mpirun on a CUDA cluster sets these; chat and inference must not refuse or go quiet.
    clean_env.setattr(sys, "platform", platform)
    for name, value in env.items():
        clean_env.setenv(name, value)
    assert inf.mlx_distributed_info() == (False, 0, None)
    assert inf.mlx_distributed_uses_mpi() is False


def test_mpi_env_on_macos_is_still_an_mlx_launch(clean_env):
    clean_env.setattr(sys, "platform", "darwin")
    clean_env.setenv("PMI_RANK", "1")
    clean_env.setenv("PMI_SIZE", "2")
    assert inf.mlx_distributed_info() == (True, 1, 2)
    assert inf.mlx_distributed_uses_mpi() is True
