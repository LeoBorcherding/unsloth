# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MergeKit merges: the YAML each method writes, request validation, the venv-located command
line, cancel, and the handoff to the export checkpoint list. No real merge runs here."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from core.export import mergekit
from models import MergeModelsRequest

FIXTURES = Path(__file__).parent / "fixtures" / "mergekit"


def _request(**fields) -> MergeModelsRequest:
    return MergeModelsRequest(**{"output_name": "merged", **fields})


def _config(**fields) -> dict:
    return mergekit.build_config(_request(**fields).model_dump())


def _fixture(name: str) -> dict:
    return yaml.safe_load((FIXTURES / name).read_text(encoding = "utf-8"))


# Requests mirroring Maxime Labonne's published configs make the same YAML, compared parsed.


def test_marcoro14_slerp_matches_the_published_config():
    a, b = "AIDC-ai-business/Marcoroni-7B-v3", "EmbeddedLLM/Mistral-7B-Merge-14-v0.1"
    config = _config(
        method = "slerp",
        models = [{"model": a, "layer_range": [0, 32]}, {"model": b, "layer_range": [0, 32]}],
        base_model = a,
        t = [
            {"filter": "self_attn", "value": [0, 0.5, 0.3, 0.7, 1]},
            {"filter": "mlp", "value": [1, 0.5, 0.7, 0.3, 0]},
            {"value": 0.5},
        ],
    )
    assert config == _fixture("marcoro14_7b_slerp.yml")


def test_neuralpipe_ties_matches_the_published_config():
    config = _config(
        method = "ties",
        base_model = "mistralai/Mistral-7B-v0.1",
        models = [
            {"model": "OpenPipe/mistral-ft-optimized-1218", "density": 0.5, "weight": 0.5},
            {"model": "mlabonne/NeuralHermes-2.5-Mistral-7B", "density": 0.5, "weight": 0.3},
        ],
        normalize = True,
        int8_mask = True,
        dtype = "float16",
    )
    assert config == _fixture("neuralpipe_7b_ties.yml")


def test_daredevil_dare_ties_matches_the_published_config():
    config = _config(
        method = "dare_ties",
        base_model = "mistralai/Mistral-7B-v0.1",
        models = [
            {"model": "samir-fama/SamirGPT-v1", "density": 0.53, "weight": 0.4},
            {"model": "abacusai/Slerp-CM-mist-dpo", "density": 0.53, "weight": 0.3},
            {"model": "EmbeddedLLM/Mistral-7B-Merge-14-v0.2", "density": 0.53, "weight": 0.3},
        ],
        int8_mask = True,
    )
    assert config == _fixture("daredevil_7b_dare_ties.yml")


# Every method's YAML.


def test_linear_has_weights_and_no_base():
    assert _config(
        method = "linear",
        models = [{"model": "org/a", "weight": 0.7}, {"model": "org/b", "weight": 0.3}],
        dtype = "float32",
    ) == {
        "models": [
            {"model": "org/a", "parameters": {"weight": 0.7}},
            {"model": "org/b", "parameters": {"weight": 0.3}},
        ],
        "merge_method": "linear",
        "dtype": "float32",
    }


def test_slerp_with_a_single_t_lists_both_models():
    assert _config(
        method = "slerp", models = [{"model": "org/a"}, {"model": "org/b"}], base_model = "org/a", t = 0.4
    ) == {
        "models": [{"model": "org/a"}, {"model": "org/b"}],
        "merge_method": "slerp",
        "base_model": "org/a",
        "parameters": {"t": 0.4},
        "dtype": "bfloat16",
    }


@pytest.mark.parametrize("method", ["ties", "dare_ties", "task_arithmetic"])
def test_task_methods_list_the_base_first_without_parameters(method):
    density = {"density": 0.6} if method != "task_arithmetic" else {}
    config = _config(
        method = method,
        base_model = "org/base",
        models = [
            {"model": "org/a", "weight": 1.0, **density},
            {"model": "org/b", "weight": 0.5, **density},
        ],
    )
    assert config["merge_method"] == method
    assert config["base_model"] == "org/base"
    assert config["models"][0] == {"model": "org/base"}
    assert config["models"][1] == {"model": "org/a", "parameters": {**density, "weight": 1.0}}
    assert "parameters" not in config


def test_config_yaml_round_trips():
    spec = _request(
        method = "ties",
        base_model = "org/base",
        models = [{"model": "org/a", "weight": 1, "density": 0.5}],
    ).model_dump()
    assert yaml.safe_load(mergekit.config_yaml(spec)) == mergekit.build_config(spec)


# Validation.


@pytest.mark.parametrize(
    "fields, message",
    [
        (dict(method = "linear", models = [{"model": "a", "weight": 1}]), "two or more"),
        (dict(method = "linear", models = [{"model": "a", "weight": 1}, {"model": "b"}]), "weight"),
        (
            dict(
                method = "linear",
                models = [{"model": "a", "weight": 1}, {"model": "b", "weight": 1}],
                base_model = "a",
            ),
            "no base_model",
        ),
        (
            dict(method = "ties", models = [{"model": "a", "weight": 1}, {"model": "b", "weight": 1}]),
            "needs a base_model",
        ),
        (
            dict(method = "dare_ties", models = [{"model": "a", "weight": 1}], base_model = "a"),
            "base model as base_model",
        ),
        (dict(method = "task_arithmetic", models = [{"model": "a"}], base_model = "base"), "weight"),
        (dict(method = "slerp", models = [{"model": "a"}, {"model": "b"}], t = 0.5), "base_model"),
        (
            dict(method = "slerp", models = [{"model": "a"}, {"model": "b"}], base_model = "c", t = 0.5),
            "base_model",
        ),
        (dict(method = "slerp", models = [{"model": "a"}, {"model": "b"}], base_model = "a"), "needs t"),
        (
            dict(
                method = "slerp",
                models = [{"model": "a"}, {"model": "b"}, {"model": "c"}],
                base_model = "a",
                t = 0.5,
            ),
            "exactly two",
        ),
        (
            dict(
                method = "slerp",
                models = [{"model": "a", "weight": 1}, {"model": "b"}],
                base_model = "a",
                t = 0.5,
            ),
            "not per-model",
        ),
        (
            dict(
                method = "linear",
                models = [{"model": "a", "weight": 1, "density": 0.5}, {"model": "b", "weight": 1}],
            ),
            "density",
        ),
        (
            dict(
                method = "linear", models = [{"model": "a", "weight": 1}, {"model": "a", "weight": 1}]
            ),
            "only once",
        ),
        (
            dict(
                method = "linear",
                models = [{"model": "a", "weight": 1}, {"model": "b", "weight": 1}],
                t = 0.5,
            ),
            "SLERP options",
        ),
    ],
)
def test_invalid_requests_are_refused(fields, message):
    with pytest.raises(ValidationError, match = message):
        _request(**fields)


@pytest.mark.parametrize("name", ["", "../x", "a/b", "a\\b", ".hidden", "c:x"])
def test_output_name_is_a_plain_folder_name(name):
    with pytest.raises(ValidationError, match = "plain folder name"):
        _request(
            method = "linear",
            models = [{"model": "a", "weight": 1}, {"model": "b", "weight": 1}],
            output_name = name,
        )


def test_density_out_of_range_is_refused():
    with pytest.raises(ValidationError):
        _request(
            method = "ties", base_model = "base", models = [{"model": "a", "weight": 1, "density": 1.5}]
        )


# The command line, from MergeKit's own venv.


def test_merge_command_runs_mergekit_yaml_from_the_venv(tmp_path):
    argv = mergekit.merge_command(tmp_path / "c.yml", tmp_path / "out")
    assert argv[0] == str(mergekit.venv_python())
    assert Path(argv[0]).is_relative_to(mergekit.venv_dir())
    assert argv[1:6] == [
        "-I",
        "-m",
        "mergekit.scripts.run_yaml",
        str(tmp_path / "c.yml"),
        str(tmp_path / "out"),
    ]
    assert "--lazy-unpickle" not in argv and "--cuda" not in argv
    assert argv[argv.index("--out-shard-size") + 1] == "5B"
    seeded = mergekit.merge_command(tmp_path / "c.yml", tmp_path / "out", cuda = True, random_seed = 7)
    assert "--cuda" in seeded and seeded[seeded.index("--random-seed") + 1] == "7"


def test_venv_lives_in_the_studio_home_not_the_studio_env():
    from utils.paths.storage_roots import studio_root
    assert mergekit.venv_dir() == studio_root() / ".venv_mergekit"
    assert not mergekit.venv_dir().is_relative_to(Path(sys.prefix))


def _mark_installed():
    root = mergekit.venv_dir()
    mergekit.venv_python(root).parent.mkdir(parents = True)
    mergekit.venv_python(root).write_text("", encoding = "utf-8")
    (root / mergekit._MARKER).write_text(mergekit._FINGERPRINT, encoding = "utf-8")


@pytest.fixture
def streams(monkeypatch):
    """Replaces the subprocess: records argv, and a MergeKit run writes a model folder."""
    calls = []

    def fake_stream(argv, env, log, on_spawn):
        calls.append((argv, env))
        log(f"ran {argv[0]}", "stdout")
        if "mergekit.scripts.run_yaml" in argv or "mergekit.scripts.moe" in argv:
            out = Path(argv[5])
            out.mkdir(parents = True)
            (out / "config.json").write_text("{}", encoding = "utf-8")
            (out / "model.safetensors").write_bytes(b"")
        return 0

    monkeypatch.setattr(mergekit, "_stream", fake_stream)
    return calls


SPEC = {
    "method": "linear",
    "models": [{"model": "org/a", "weight": 1.0}, {"model": "org/b", "weight": 1.0}],
    "dtype": "bfloat16",
    "output_name": "merged",
    "cuda": False,
}


def _noop(*args):
    return None


def test_merge_writes_the_config_and_publishes_only_a_finished_folder(tmp_path, streams):
    _mark_installed()
    out = tmp_path / "outputs" / "merged"
    assert mergekit.merge(SPEC, out, log = _noop, on_spawn = _noop, hf_token = "hf_x") == out
    ((argv, env),) = streams
    assert argv[0] == str(mergekit.venv_python())
    assert env["HF_TOKEN"] == "hf_x"
    assert (out / "config.json").is_file()
    assert [p.name for p in tmp_path.iterdir()] == ["outputs"], "staging was not cleaned up"


def test_a_failed_merge_leaves_no_output(tmp_path, monkeypatch):
    _mark_installed()
    monkeypatch.setattr(mergekit, "_stream", lambda argv, env, log, on_spawn: 1)
    out = tmp_path / "outputs" / "merged"
    with pytest.raises(RuntimeError, match = "exited with code 1"):
        mergekit.merge(SPEC, out, log = _noop, on_spawn = _noop)
    assert not out.exists()
    assert [p.name for p in tmp_path.iterdir()] == ["outputs"]


def test_an_existing_output_is_never_overwritten(tmp_path, streams):
    _mark_installed()
    out = tmp_path / "merged"
    out.mkdir()
    with pytest.raises(RuntimeError, match = "already exists"):
        mergekit.merge(SPEC, out, log = _noop, on_spawn = _noop)
    assert streams == []


def test_merge_without_mergekit_needs_consent(tmp_path, streams):
    with pytest.raises(RuntimeError, match = "not installed"):
        mergekit.merge(SPEC, tmp_path / "merged", log = _noop, on_spawn = _noop)
    assert streams == []


def test_install_creates_a_venv_on_studio_python_and_shares_its_torch(monkeypatch, streams):
    from core.inference import engine_install

    monkeypatch.setattr(mergekit.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(engine_install, "_studio_packages", lambda: {"torch": "2.11.0"})

    def fake_stream(argv, env, log, on_spawn):
        streams.append((argv, env))
        if argv[1] == "venv":
            mergekit._venv_site(Path(argv[-1])).mkdir(parents = True)
        return 0

    monkeypatch.setattr(mergekit, "_stream", fake_stream)
    mergekit.install(_noop, _noop)
    root = mergekit.venv_dir()
    (venv, _), (pip, _), (check, _) = streams
    assert venv == ["/usr/bin/uv", "venv", "--python", sys.executable, str(root)]
    assert pip[:5] == ["/usr/bin/uv", "pip", "install", "--python", str(mergekit.venv_python(root))]
    assert pip[-1] == f"mergekit=={mergekit.MERGEKIT_VERSION}"
    overrides = Path(pip[pip.index("--override") + 1])
    assert overrides.read_text(encoding = "utf-8").startswith("torch ;")
    assert check[0] == str(mergekit.venv_python(root))
    site = mergekit._venv_site(root)
    assert (site / engine_install._BASE_PTH).is_file()
    assert (root / mergekit._MARKER).read_text(encoding = "utf-8") == mergekit._FINGERPRINT


def test_install_without_studio_torch_lets_mergekit_bring_its_own(monkeypatch, streams):
    from core.inference import engine_install

    monkeypatch.setattr(mergekit.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(engine_install, "_studio_packages", lambda: {})

    def fake_stream(argv, env, log, on_spawn):
        streams.append(argv)
        mergekit.venv_dir().mkdir(parents = True, exist_ok = True)
        return 0

    monkeypatch.setattr(mergekit, "_stream", fake_stream)
    mergekit.install(_noop, _noop)
    assert "--override" not in streams[1]
    assert not (mergekit._venv_site(mergekit.venv_dir()) / engine_install._BASE_PTH).exists()


def test_a_failed_install_removes_the_venv(monkeypatch):
    monkeypatch.setattr(mergekit.shutil, "which", lambda name: "uv")

    def fake_stream(argv, env, log, on_spawn):
        mergekit.venv_dir().mkdir(parents = True, exist_ok = True)
        return 0 if argv[1] == "venv" else 2

    monkeypatch.setattr(mergekit, "_stream", fake_stream)
    with pytest.raises(RuntimeError, match = "installation failed"):
        mergekit.install(_noop, _noop)
    assert not mergekit.venv_dir().exists()
    assert not mergekit.is_installed()


def test_probe_asks_for_consent_until_installed(monkeypatch):
    monkeypatch.setattr(mergekit.shutil, "which", lambda name: "uv")
    probe = mergekit.probe()
    assert probe["ready"] is False and probe["needs_consent"] is True
    assert "own environment" in probe["install_summary"]
    _mark_installed()
    probe = mergekit.probe()
    assert probe["ready"] is True and probe["needs_consent"] is False


# The export orchestrator: logs, cancel, and the handoff to the checkpoint list.


class _FakeProc:
    def __init__(self):
        self.pid = 4321
        self.terminated = False

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self):
        self.terminated = True


class _FakeWorker:
    pid = 1234
    terminated = False

    def is_alive(self):
        return not self.terminated

    def terminate(self):
        self.terminated = True

    def join(self, timeout = None):
        pass


@pytest.fixture
def orchestrator(monkeypatch):
    from core.export.orchestrator import ExportOrchestrator

    backend = ExportOrchestrator()
    yield backend
    backend._cleanup()


def test_a_merged_model_is_listed_with_the_export_checkpoints(orchestrator, streams):
    from utils.models.checkpoints import scan_checkpoints
    from utils.paths.storage_roots import outputs_root

    _mark_installed()
    ok, message, path = orchestrator.merge_models(dict(SPEC, output_name = "my-merge"))
    assert ok, message
    assert Path(path) == outputs_root() / "my-merge"
    assert orchestrator.get_last_op()["kind"] == "merge"
    assert orchestrator.get_last_op()["status"] == "success"
    lines = [entry["line"] for entry in orchestrator.get_logs_since(0)[0]]
    assert any("Merged model saved" in line for line in lines)
    assert "my-merge" in [name for name, _checkpoints, _meta in scan_checkpoints()]


def test_cancel_stops_the_merge_and_not_the_export_worker(orchestrator, monkeypatch):
    _mark_installed()
    proc = _FakeProc()

    def fake_stream(argv, env, log, on_spawn):
        on_spawn(proc)
        assert orchestrator.is_export_active()
        assert orchestrator.cancel_export() is True
        return -15

    monkeypatch.setattr(mergekit, "_stream", fake_stream)
    monkeypatch.setattr(orchestrator, "_proc", None)
    ok, message, path = orchestrator.merge_models(dict(SPEC))
    assert proc.terminated
    assert (ok, path) == (False, None)
    assert message == "Export cancelled"
    assert orchestrator.get_last_op()["status"] == "cancelled"
    assert not orchestrator.is_export_active()
    assert orchestrator._merge_proc is None


def test_a_cancel_aimed_at_another_op_leaves_the_merge_running(orchestrator, monkeypatch):
    _mark_installed()
    proc = _FakeProc()

    def fake_stream(argv, env, log, on_spawn):
        on_spawn(proc)
        assert orchestrator.cancel_export(expected_op_id = "someone-else") is False
        out = Path(argv[5])
        out.mkdir(parents = True)
        (out / "config.json").write_text("{}", encoding = "utf-8")
        return 0

    monkeypatch.setattr(mergekit, "_stream", fake_stream)
    ok, _message, _path = orchestrator.merge_models(dict(SPEC), op_id = "mine")
    assert ok and not proc.terminated
    # Status recovery tells this merge's record from another tab's by this id.
    assert orchestrator.get_last_op()["op_id"] == "mine"


def test_a_cancel_before_mergekit_spawns_spares_the_worker_and_skips_the_merge(
    orchestrator, monkeypatch
):
    _mark_installed()
    worker = _FakeWorker()
    spawned = []
    monkeypatch.setattr(orchestrator, "_proc", worker)
    monkeypatch.setattr(
        mergekit, "_stream", lambda argv, env, log, on_spawn: spawned.append(argv) or 0
    )
    real_config_yaml = mergekit.config_yaml

    def cancel_while_staging(spec):
        assert orchestrator.cancel_export() is True
        return real_config_yaml(spec)

    monkeypatch.setattr(mergekit, "config_yaml", cancel_while_staging)
    ok, message, _path = orchestrator.merge_models(dict(SPEC))
    assert (ok, message) == (False, "Export cancelled")
    assert not worker.terminated and spawned == []
    worker.terminated = True  # so the fixture's cleanup does not talk to it


def test_a_cancel_right_after_the_merge_claims_its_id_spares_the_worker(orchestrator, monkeypatch):
    _mark_installed()
    worker = _FakeWorker()
    monkeypatch.setattr(orchestrator, "_proc", worker)
    real_begin = orchestrator._begin_op

    def begin_then_cancel(op_id, kind = None):
        started = real_begin(op_id, kind)
        assert orchestrator.cancel_export(expected_op_id = "mine") is True
        return started

    monkeypatch.setattr(orchestrator, "_begin_op", begin_then_cancel)
    monkeypatch.setattr(mergekit, "_stream", lambda *args: 0)
    ok, message, _path = orchestrator.merge_models(dict(SPEC), op_id = "mine")
    assert (ok, message) == (False, "Export cancelled")
    assert not worker.terminated
    worker.terminated = True  # so the fixture's cleanup does not talk to it


def test_a_cancel_after_the_merge_is_published_still_records_success(orchestrator, monkeypatch):
    _mark_installed()
    real_merge = mergekit.merge

    def merge_then_cancel(*args, **kwargs):
        out = real_merge(*args, **kwargs)
        orchestrator.cancel_export()
        return out

    def fake_stream(argv, env, log, on_spawn):
        out = Path(argv[5])
        out.mkdir(parents = True)
        (out / "config.json").write_text("{}", encoding = "utf-8")
        return 0

    monkeypatch.setattr(mergekit, "_stream", fake_stream)
    monkeypatch.setattr(mergekit, "merge", merge_then_cancel)
    monkeypatch.setattr(orchestrator, "_proc", None)
    ok, _message, path = orchestrator.merge_models(dict(SPEC))
    assert ok and Path(path).is_dir()
    assert orchestrator.get_last_op()["status"] == "success"


def test_route_authorizes_hub_models_and_reports_the_run(monkeypatch):
    from routes import export as export_routes

    checked, calls = [], []
    monkeypatch.setattr(
        export_routes.account_access,
        "authorize_download",
        lambda repo, repo_type, token: checked.append(repo),
    )

    class Backend:
        def merge_models(self, spec, **kwargs):
            calls.append((spec, kwargs))
            return True, "Models merged", "/x/outputs/m"

    monkeypatch.setattr(export_routes, "get_export_backend", lambda: Backend())
    request = MergeModelsRequest(
        method = "ties",
        base_model = "org/base",
        models = [{"model": "org/a", "weight": 1}],
        output_name = "m",
        install_missing_dependencies = True,
        hf_token = "hf_x",
    )
    response = asyncio.run(export_routes.merge_models(request, "subject", allow_ambient = True))
    assert checked == ["org/a", "org/base"]
    spec, kwargs = calls[0]
    assert "hf_token" not in spec and spec["output_name"] == "m"
    assert kwargs["install_missing_dependencies"] is True
    assert response.details == {"output_path": "/x/outputs/m", "run": "m"}


def test_route_refuses_a_share_path_before_touching_it(monkeypatch):
    from fastapi import HTTPException
    from core.training import account_jobs
    from routes import export as export_routes

    probed = []
    monkeypatch.setattr(account_jobs, "managed_account", lambda: True)
    monkeypatch.setattr(Path, "is_dir", lambda self: probed.append(str(self)) or False)
    monkeypatch.setattr(export_routes.account_access, "require_model_allowed", lambda name: None)
    monkeypatch.setattr(export_routes.account_access, "authorize_download", lambda *args: None)
    monkeypatch.setattr(
        export_routes, "_resolve_export_hf_token", lambda token, allow_ambient: None
    )
    request = MergeModelsRequest(
        method = "ties",
        base_model = "org/base",
        models = [{"model": "//host/share/model", "weight": 1}],
        output_name = "m",
        hf_token = "hf_x",
    )
    with pytest.raises(HTTPException):
        asyncio.run(export_routes.merge_models(request, "subject", allow_ambient = True))
    assert not any("share" in path for path in probed)


def test_tokenizer_source_is_written_when_vocabularies_differ():
    spec = MergeModelsRequest(
        method = "ties",
        base_model = "mistralai/Mistral-7B-v0.1",
        models = [
            {"model": "OpenPipe/mistral-ft-optimized-1218", "density": 0.5, "weight": 0.5},
            {"model": "mlabonne/NeuralHermes-2.5-Mistral-7B", "density": 0.5, "weight": 0.3},
        ],
        tokenizer_source = "base",
        output_name = "m",
    ).model_dump()
    assert mergekit.build_config(spec)["tokenizer_source"] == "base"


@pytest.mark.parametrize(
    "fields",
    [
        {
            "method": "linear",
            "models": [{"model": "a", "weight": 1}, {"model": "b", "weight": 1}],
            "tokenizer_source": "base",
        },
        {
            "method": "linear",
            "models": [{"model": "a", "weight": 1}, {"model": "b", "weight": 1}],
            "tokenizer_source": "c",
        },
    ],
)
def test_tokenizer_source_must_name_a_base_union_or_merged_model(fields):
    with pytest.raises(ValueError, match = "tokenizer_source"):
        MergeModelsRequest(**fields, output_name = "m")


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_weights_and_t_are_refused(bad):
    with pytest.raises(ValueError):
        MergeModelsRequest(
            method = "linear",
            models = [{"model": "a", "weight": bad}, {"model": "b", "weight": 1}],
            output_name = "m",
        )
    with pytest.raises(ValueError):
        MergeModelsRequest(
            method = "slerp",
            base_model = "a",
            models = [{"model": "a"}, {"model": "b"}],
            t = bad,
            output_name = "m",
        )


def test_an_ops_record_outlives_a_later_merge_for_status_by_op_id(orchestrator, monkeypatch):
    _mark_installed()

    def fake_stream(argv, env, log, on_spawn):
        out = Path(argv[5])
        out.mkdir(parents = True)
        (out / "config.json").write_text("{}", encoding = "utf-8")
        return 0

    monkeypatch.setattr(mergekit, "_stream", fake_stream)
    assert orchestrator._begin_op("export-1", kind = "export_gguf")
    orchestrator._record_op_finished(True, "", "/out/gguf")
    orchestrator._end_op()
    orchestrator._active_op_kind = None
    ok, _message, _path = orchestrator.merge_models(dict(SPEC), op_id = "merge-1")
    assert ok
    assert orchestrator.get_last_op()["op_id"] == "merge-1"
    assert orchestrator.get_last_op("export-1")["output_path"] == "/out/gguf"
    assert orchestrator.get_last_op("gone")["op_id"] == "merge-1"


# Mixture of experts (mergekit-moe).

BEYONDER_EXPERTS = [
    ("openchat/openchat-3.5-1210", ["chat", "assistant", "tell me", "explain"]),
    (
        "beowolx/CodeNinja-1.0-OpenChat-7B",
        ["code", "python", "javascript", "programming", "algorithm"],
    ),
    ("maywell/PiVoT-0.1-Starling-LM-RP", ["storywriting", "write", "scene", "story", "character"]),
    ("WizardLM/WizardMath-7B-V1.1", ["reason", "math", "mathematics", "solve", "count"]),
]
QWEN3_INSTRUCT, QWEN3_THINKING = "Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-4B-Thinking-2507"


def _moe(**fields) -> MergeModelsRequest:
    return _request(
        **{
            "method": "moe",
            "base_model": "org/base",
            "models": [
                {"model": "org/a", "positive_prompts": ["chat"]},
                {"model": "org/b", "positive_prompts": ["math"]},
            ],
            **fields,
        }
    )


def test_beyonder_moe_matches_the_published_config():
    config = _config(
        method = "moe",
        base_model = "mlabonne/Marcoro14-7B-slerp",
        models = [{"model": m, "positive_prompts": p} for m, p in BEYONDER_EXPERTS],
    )
    # The published config leaves these at MergeKit's defaults (and its base is bfloat16).
    assert config == {
        **_fixture("beyonder_4x7b_v2_moe.yml"),
        "gate_mode": "hidden",
        "dtype": "bfloat16",
        "experts_per_token": 2,
    }


def test_qwen3_moe_takes_the_base_as_an_expert_and_names_the_architecture():
    config = _config(
        method = "moe",
        base_model = QWEN3_INSTRUCT,
        models = [
            {"model": QWEN3_INSTRUCT, "positive_prompts": ["chat", "explain", "tell me", "write"]},
            {
                "model": QWEN3_THINKING,
                "positive_prompts": ["reason", "math", "solve", "prove", "step by step"],
                "negative_prompts": ["chat"],
            },
        ],
        experts_per_token = 1,
        architecture = "qwen3",
        gate_mode = "cheap_embed",
    )
    assert config == {
        "base_model": QWEN3_INSTRUCT,
        "gate_mode": "cheap_embed",
        "dtype": "bfloat16",
        "experts_per_token": 1,
        "architecture": "qwen3",
        "experts": [
            {
                "source_model": QWEN3_INSTRUCT,
                "positive_prompts": ["chat", "explain", "tell me", "write"],
            },
            {
                "source_model": QWEN3_THINKING,
                "positive_prompts": ["reason", "math", "solve", "prove", "step by step"],
                "negative_prompts": ["chat"],
            },
        ],
    }


def test_random_gates_need_no_prompts_and_blank_prompts_are_dropped():
    config = mergekit.build_config(
        _moe(
            gate_mode = "random",
            models = [{"model": "org/a"}, {"model": "org/b", "positive_prompts": [" ", ""]}],
        ).model_dump()
    )
    assert config["experts"] == [{"source_model": "org/a"}, {"source_model": "org/b"}]


@pytest.mark.parametrize(
    "fields, message",
    [
        (dict(models = [{"model": "org/a", "positive_prompts": ["x"]}]), "at least two experts"),
        (dict(base_model = None), "needs a base_model"),
        (dict(experts_per_token = 0), "between 1 and the number of experts"),
        (dict(experts_per_token = 3), "between 1 and the number of experts"),
        (
            dict(
                models = [
                    {"model": "org/a", "positive_prompts": ["x"]},
                    {"model": "org/a", "positive_prompts": ["y"]},
                ]
            ),
            "only once",
        ),
        (
            dict(models = [{"model": "org/a", "positive_prompts": ["x"]}, {"model": "org/b"}]),
            "positive_prompts for hidden",
        ),
        (
            dict(gate_mode = "uniform_random", models = [{"model": "org/a"}, {"model": "org/b"}]),
            "positive_prompts for uniform_random",
        ),
        (
            dict(
                models = [
                    {"model": "org/a", "positive_prompts": ["same"]},
                    {"model": "org/b", "positive_prompts": ["same"]},
                ]
            ),
            "different prompts",
        ),
        (
            dict(
                models = [
                    {"model": "org/a", "positive_prompts": ["x"], "weight": 1},
                    {"model": "org/b", "positive_prompts": ["y"]},
                ]
            ),
            "not weight",
        ),
        (dict(tokenizer_source = "union"), "base model's tokenizer"),
        (dict(t = 0.5), "do not apply to MoE"),
        (dict(architecture = "qwen"), "architecture"),
        (dict(gate_mode = "router"), "gate_mode"),
    ],
)
def test_invalid_moe_requests_are_refused(fields, message):
    with pytest.raises(ValidationError, match = message):
        _moe(**fields)


@pytest.mark.parametrize(
    "fields",
    [
        dict(gate_mode = "hidden"),
        dict(experts_per_token = 2),
        dict(architecture = "mixtral"),
        dict(
            models = [
                {"model": "a", "weight": 1, "positive_prompts": ["x"]},
                {"model": "b", "weight": 1},
            ]
        ),
    ],
)
def test_moe_options_are_refused_on_dense_methods(fields):
    with pytest.raises(ValidationError, match = "MoE"):
        _request(
            **{
                "method": "linear",
                "models": [{"model": "a", "weight": 1}, {"model": "b", "weight": 1}],
                **fields,
            }
        )


def test_moe_command_runs_mergekit_moe_from_the_venv(tmp_path):
    argv = mergekit.merge_command(tmp_path / "c.yml", tmp_path / "out", moe = True, cuda = True)
    assert argv[:6] == [
        str(mergekit.venv_python()),
        "-I",
        "-m",
        "mergekit.scripts.moe",
        str(tmp_path / "c.yml"),
        str(tmp_path / "out"),
    ]
    assert "--lazy-unpickle" not in argv and "--copy-tokenizer" in argv and "--cuda" in argv
    assert not any(arg.startswith("--i-understand") for arg in argv)


def test_moe_merge_writes_its_config_and_publishes_the_folder(tmp_path, monkeypatch):
    _mark_installed()
    seen = {}

    def fake_stream(argv, env, log, on_spawn):
        seen["argv"] = argv
        seen["config"] = yaml.safe_load(Path(argv[4]).read_text(encoding = "utf-8"))
        out = Path(argv[5])
        out.mkdir(parents = True)
        (out / "config.json").write_text("{}", encoding = "utf-8")
        return 0

    monkeypatch.setattr(mergekit, "_stream", fake_stream)
    spec = _moe(gate_mode = "random", random_seed = 3, cuda = True).model_dump()
    out = tmp_path / "outputs" / "moe"
    lines = []
    merged = mergekit.merge(spec, out, log = lambda line, stream: lines.append(line), on_spawn = _noop)
    assert merged == out and (out / "config.json").is_file()
    argv = seen["argv"]
    assert argv[3] == "mergekit.scripts.moe"
    assert "--cuda" in argv and argv[argv.index("--random-seed") + 1] == "3"
    assert seen["config"]["gate_mode"] == "random" and len(seen["config"]["experts"]) == 2
    assert lines[0] == "Building a mixture of 2 experts"
    assert [p.name for p in out.parent.iterdir()] == ["moe"], "staging was not cleaned up"


def test_route_authorizes_every_expert_and_the_base_once(monkeypatch):
    from routes import export as export_routes

    checked, calls = [], []
    monkeypatch.setattr(
        export_routes.account_access,
        "authorize_download",
        lambda repo, repo_type, token: checked.append(repo),
    )

    class Backend:
        def merge_models(self, spec, **kwargs):
            calls.append(spec)
            return True, "Models merged", "/x/outputs/m"

    monkeypatch.setattr(export_routes, "get_export_backend", lambda: Backend())
    request = _moe(
        base_model = QWEN3_INSTRUCT,
        models = [
            {"model": QWEN3_INSTRUCT, "positive_prompts": ["chat"]},
            {"model": QWEN3_THINKING, "positive_prompts": ["math"]},
        ],
    )
    asyncio.run(export_routes.merge_models(request, "subject", allow_ambient = True))
    assert sorted(checked) == sorted([QWEN3_INSTRUCT, QWEN3_THINKING])
    assert calls[0]["method"] == "moe" and calls[0]["models"][1]["positive_prompts"] == ["math"]


def test_route_refuses_an_expert_on_a_share_path_before_touching_it(monkeypatch):
    from fastapi import HTTPException
    from core.training import account_jobs
    from routes import export as export_routes

    probed = []
    monkeypatch.setattr(account_jobs, "managed_account", lambda: True)
    monkeypatch.setattr(Path, "is_dir", lambda self: probed.append(str(self)) or False)
    monkeypatch.setattr(export_routes.account_access, "require_model_allowed", lambda name: None)
    monkeypatch.setattr(export_routes.account_access, "authorize_download", lambda *args: None)
    monkeypatch.setattr(
        export_routes, "_resolve_export_hf_token", lambda token, allow_ambient: None
    )
    request = _moe(
        models = [
            {"model": "org/a", "positive_prompts": ["chat"]},
            {"model": "//host/share/expert", "positive_prompts": ["math"]},
        ],
    )
    with pytest.raises(HTTPException):
        asyncio.run(export_routes.merge_models(request, "subject", allow_ambient = True))
    assert not any("share" in path for path in probed)


@pytest.mark.parametrize(
    "field",
    [
        {
            "models": [
                {"model": "ws/m+/Users/bob/adapter", "weight": 1.0},
                {"model": "b", "weight": 1.0},
            ]
        },
        {"models": [{"model": "org/a@refs/pr/1", "weight": 1.0}, {"model": "b", "weight": 1.0}]},
        {
            "models": [{"model": "a", "weight": 1.0}, {"model": "b", "weight": 1.0}],
            "base_model": "base+lora",
        },
    ],
)
def test_lora_and_revision_syntax_is_refused(field):
    # MergeKit would load the part after + or @ as a LoRA or revision the path check never saw.
    with pytest.raises(ValidationError, match = r"'\+' or '@'"):
        _request(method = "ties", base_model = field.pop("base_model", "base"), **field)


def test_request_size_is_bounded():
    many = [{"model": f"m{i}", "weight": 1.0} for i in range(17)]
    with pytest.raises(ValidationError):
        _request(method = "linear", models = many)
    experts = [
        {"model": "a", "positive_prompts": ["x"] * 65},
        {"model": "b", "positive_prompts": ["y"]},
    ]
    with pytest.raises(ValidationError):
        _request(method = "moe", base_model = "a", models = experts)
    experts[0]["positive_prompts"] = ["x" * 2001]
    with pytest.raises(ValidationError):
        _request(method = "moe", base_model = "a", models = experts)
    two = [{"model": "a", "weight": 1.0}, {"model": "b", "weight": 1.0}]
    with pytest.raises(ValidationError):
        _request(method = "linear", models = two, random_seed = 2**32)


def test_stale_staging_is_swept_and_a_fresh_one_kept(tmp_path):
    import os, time

    stale, fresh = tmp_path / ".mergekit-old", tmp_path / ".mergekit-new"
    for d in (stale, fresh):
        (d / "model").mkdir(parents = True)
    old = time.time() - mergekit.STALE_STAGING_SECONDS - 60
    for p in (stale, stale / "model"):
        os.utime(p, (old, old))
    keep = tmp_path / "merged-model"
    keep.mkdir()
    os.utime(keep, (old, old))
    mergekit._sweep_stale_staging(tmp_path)
    assert not stale.exists() and fresh.exists() and keep.exists()
