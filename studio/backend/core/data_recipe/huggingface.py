# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from core.training.account_jobs import account_hf_token
import json
from pathlib import Path

from utils.paths import recipe_datasets_root, resolve_dataset_path

_DATA_DESIGNER_FOOTER = (
    '<sub style="white-space: nowrap;">Made with ❤️ using 🎨 '
    '<a href="https://github.com/NVIDIA-NeMo/DataDesigner">NeMo Data Designer</a></sub>'
)
_UNSLOTH_STUDIO_FOOTER = (
    '<sub style="white-space: nowrap;">Made with ❤️ using 🦥 ' "Unsloth Studio</sub>"
)


# Top-level folders the main dataset owns on the Hub; a processor uploaded there would replace them.
_RESERVED_REPO_FOLDERS = frozenset({"data", "images"})
_HUB_ERRORS = {
    401: "Hugging Face rejected the token (401). Check the token saved in Settings or the one entered here.",
    403: "This token can't write to {repo_id} (403). Use a token with write access to it.",
    429: "Hugging Face is rate limiting uploads (429). Wait a few minutes, then publish again.",
}


class RecipeDatasetPublishError(ValueError):
    """Raised when a recipe dataset cannot be published to Hugging Face."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _resolve_recipe_artifact_path(artifact_path: str) -> Path:
    root = recipe_datasets_root().expanduser().resolve()
    try:
        candidate = resolve_dataset_path(artifact_path).expanduser()
    except ValueError as exc:
        # Outside every dataset root, so it never reaches the check below: a 500, not a refusal.
        raise RecipeDatasetPublishError(
            "This execution artifact is outside the Recipe Studio dataset storage."
        ) from exc
    resolved = candidate.resolve(strict = False)

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RecipeDatasetPublishError(
            "This execution artifact is outside the Recipe Studio dataset storage."
        ) from exc

    if not resolved.exists():
        raise RecipeDatasetPublishError("Execution artifacts are no longer available.")
    if not resolved.is_dir():
        raise RecipeDatasetPublishError("Execution artifact path is not a dataset folder.")

    return resolved


def _drop_seed_token(builder_config: dict) -> None:
    # The hf and github_repo seed sources save their token in plain text.
    seed_config = builder_config.get("data_designer", {}).get("seed_config") or {}
    seed_config.get("source", {}).pop("token", None)


def _ensure_hub_repo_private(hf_api, repo_id: str) -> None:
    # create_repo ignores `private` when the repo already exists.
    try:
        hf_api.update_repo_settings(repo_id = repo_id, private = True, repo_type = "dataset")
        return
    except Exception as exc:
        try:
            info = hf_api.repo_info(repo_id = repo_id, repo_type = "dataset")
            if bool(getattr(info, "private", False)):
                return
        except Exception:
            pass
        raise RecipeDatasetPublishError(
            f"Could not make {repo_id} private, so nothing was uploaded. The token likely "
            "lacks permission to change this repo's settings."
        ) from exc


def _repo_files(folder: Path, path_in_repo: str) -> dict[str, Path]:
    if not folder.is_dir():
        return {}
    return {
        f"{path_in_repo}/{path.relative_to(folder).as_posix()}": path
        for path in sorted(folder.rglob("*"))
        if path.is_file() and ".git" not in path.relative_to(folder).parts
    }


def _hub_file_paths(file_paths: dict, final_folder: str, processors_folder: str) -> dict:
    # The engine's own rewrite only matches "/", so Windows paths kept their local prefix.
    def posix(paths):
        return [str(path).replace("\\", "/") for path in paths]

    hub_paths = {}
    if final_folder in file_paths:
        hub_paths["data"] = [
            path.replace(f"{final_folder}/", "data/", 1)
            for path in posix(file_paths[final_folder])
        ]
    if "processor-files" in file_paths:
        hub_paths["processor-files"] = {
            name: [
                path.replace(f"{processors_folder}/{name}/", f"{name}/", 1)
                for path in posix(paths)
            ]
            for name, paths in file_paths["processor-files"].items()
        }
    return hub_paths


def _hub_error(exc: Exception, repo_id: str, created: bool) -> RecipeDatasetPublishError:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    message = _HUB_ERRORS.get(status, "Upload to {repo_id} failed: {exc}").format(
        repo_id = repo_id, exc = exc
    )
    if created:
        message += f" The repo {repo_id} was created on the Hub, but nothing from this run is in it."
    else:
        message += f" Nothing in {repo_id} was changed."
    return RecipeDatasetPublishError(message, 429 if status == 429 else 400)


def publish_recipe_dataset(
    *,
    artifact_path: str,
    repo_id: str,
    description: str,
    hf_token: str | None = None,
    private: bool = False,
    link_endpoint: str | None = None,
) -> str:
    hf_token = account_hf_token(hf_token)
    dataset_path = _resolve_recipe_artifact_path(artifact_path)

    try:
        from data_designer.engine.storage.artifact_storage import (
            FINAL_DATASET_FOLDER_NAME,
            METADATA_FILENAME,
            PROCESSORS_OUTPUTS_FOLDER_NAME,
            SDG_CONFIG_FILENAME,
        )
        from data_designer.integrations.huggingface.client import (
            HuggingFaceHubClient,
            HuggingFaceHubClientUploadError,
        )
        from data_designer.integrations.huggingface.dataset_card import (
            DataDesignerDatasetCard,
        )
    except ImportError as exc:
        raise RecipeDatasetPublishError(
            "NeMo Data Designer Hugging Face integration is not installed."
        ) from exc

    try:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    except ImportError as exc:
        raise RecipeDatasetPublishError("huggingface_hub is not installed.") from exc

    try:
        client = HuggingFaceHubClient(token = hf_token)
        client._validate_repo_id(repo_id = repo_id)
        client._validate_dataset_path(base_dataset_path = dataset_path)
        processors_folder = dataset_path / PROCESSORS_OUTPUTS_FOLDER_NAME
        processor_dirs = (
            sorted(path for path in processors_folder.iterdir() if path.is_dir())
            if processors_folder.is_dir()
            else []
        )
        for processor_dir in processor_dirs:
            if processor_dir.name in _RESERVED_REPO_FOLDERS:
                raise RecipeDatasetPublishError(
                    f"A processor named '{processor_dir.name}' would replace the dataset's own "
                    f"'{processor_dir.name}' folder on the Hub. Rename the processor and run the recipe again."
                )

        api = client._api
        try:
            existed = bool(api.repo_exists(repo_id = repo_id, repo_type = "dataset"))
        except Exception:
            existed = True
        client._create_or_get_repo(repo_id = repo_id, private = private)
        if private:
            _ensure_hub_repo_private(api, repo_id)

        metadata_path = dataset_path / METADATA_FILENAME
        builder_config_path = dataset_path / SDG_CONFIG_FILENAME

        with metadata_path.open(encoding = "utf-8") as fh:
            metadata = json.load(fh)

        builder_config = None
        if builder_config_path.exists():
            with builder_config_path.open(encoding = "utf-8") as fh:
                builder_config = json.load(fh)
            _drop_seed_token(builder_config)

        card = DataDesignerDatasetCard.from_metadata(
            metadata = metadata,
            builder_config = builder_config,
            repo_id = repo_id,
            description = description,
            tags = None,
        )
        card.text = card.text.replace(_DATA_DESIGNER_FOOTER, _UNSLOTH_STUDIO_FOOTER)

        files = _repo_files(dataset_path / FINAL_DATASET_FOLDER_NAME, "data")
        files.update(_repo_files(dataset_path / "images", "images"))
        for processor_dir in processor_dirs:
            files.update(_repo_files(processor_dir, processor_dir.name))
        hub_metadata = dict(metadata)
        if isinstance(metadata.get("file_paths"), dict):
            hub_metadata["file_paths"] = _hub_file_paths(
                metadata["file_paths"],
                FINAL_DATASET_FOLDER_NAME,
                PROCESSORS_OUTPUTS_FOLDER_NAME,
            )
        blobs = {
            "README.md": str(card).encode("utf-8"),
            METADATA_FILENAME: json.dumps(hub_metadata, indent = 2).encode("utf-8"),
        }
        if builder_config is not None:
            blobs[SDG_CONFIG_FILENAME] = json.dumps(
                builder_config, indent = 2, ensure_ascii = False
            ).encode("utf-8")

        # One commit with the client's token, so a failure leaves the repo as it was and a
        # re-publish drops the last run's files from every folder. Other top-level files stay.
        try:
            stale = [
                path
                for path in api.list_repo_files(repo_id = repo_id, repo_type = "dataset")
                if "/" in path and path not in files
            ]
            api.create_commit(
                repo_id = repo_id,
                repo_type = "dataset",
                operations = [
                    *(CommitOperationDelete(path_in_repo = path) for path in stale),
                    *(
                        CommitOperationAdd(path_in_repo = name, path_or_fileobj = str(path))
                        for name, path in files.items()
                    ),
                    *(
                        CommitOperationAdd(path_in_repo = name, path_or_fileobj = data)
                        for name, data in blobs.items()
                    ),
                ],
                commit_message = "Upload dataset from Unsloth Studio",
            )
        except Exception as exc:
            raise _hub_error(exc, repo_id, created = not existed) from exc

        from utils.hf_endpoint import get_hf_endpoint

        # Browser-facing URL: a loopback mirror is not the same host.
        return f"{link_endpoint or get_hf_endpoint()}/datasets/{repo_id}"
    except HuggingFaceHubClientUploadError as exc:
        raise RecipeDatasetPublishError(str(exc)) from exc
