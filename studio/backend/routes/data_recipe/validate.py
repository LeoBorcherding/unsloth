# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Validation endpoints for data recipe."""

from __future__ import annotations

import json
from collections import Counter
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from auth.authentication import (
    authenticated_via_api_key,
    require_ui_session_for_local_commands,
)
from core.data_recipe.service import (
    build_config_builder,
    create_data_designer,
    recipe_has_stdio_mcp,
    require_ui_session_for_host_env_credentials,
    validate_recipe,
)
from loggers import get_logger
from models.data_recipe import RecipePayload, ValidateError, ValidateResponse
from utils.utils import safe_error_detail, safe_curated_detail, log_and_http_error

from .jobs import _resolve_seed_endpoint

logger = get_logger(__name__)
router = APIRouter()

# Only UI sessions may supply stdio commands, since this host would run them; Annotated so direct calls get False.
ViaApiKey = Annotated[bool, Depends(authenticated_via_api_key)]

_GITHUB_VALIDATE_NOTE = (
    "Recipe shape is valid. GitHub access and rate limits are checked when the run starts."
)
_GITHUB_ITEM_TYPES = {"issues", "pulls", "commits"}
# Every GitHub seed row has these (the plugin's scraper.py flattens issues, pulls and commits to them), so Check can
# resolve references without scraping.
_GITHUB_SEED_COLUMNS = (
    "item_type",
    "repo",
    "number",
    "title",
    "body",
    "state",
    "author",
    "created_at",
    "closed_at",
    "url",
    "labels",
    "comments",
)
_TEMPLATE_FIELDS = ("prompt", "system_prompt", "expr")


def _github_seed_source(recipe: dict[str, Any]) -> dict[str, Any] | None:
    seed_config = recipe.get("seed_config")
    if not isinstance(seed_config, dict):
        return None
    source = seed_config.get("source")
    if not isinstance(source, dict) or source.get("seed_type") != "github_repo":
        return None
    return source


def _validate_github_seed_static(source: dict[str, Any]) -> list[ValidateError]:
    errors: list[ValidateError] = []

    repos = source.get("repos")
    if not isinstance(repos, list) or not repos:
        errors.append(ValidateError(message = "GitHub seed requires at least one repo."))
    else:
        for repo in repos:
            if not isinstance(repo, str) or not repo.strip() or "/" not in repo:
                errors.append(ValidateError(message = "GitHub repos must be owner/name strings."))
                break

    item_types = source.get("item_types")
    if not isinstance(item_types, list) or not item_types:
        errors.append(ValidateError(message = "GitHub seed requires at least one item type."))
    else:
        invalid_items = [item for item in item_types if item not in _GITHUB_ITEM_TYPES]
        if invalid_items:
            errors.append(
                ValidateError(message = "GitHub item types must be issues, pulls, or commits.")
            )

    try:
        limit = int(source.get("limit"))
    except (TypeError, ValueError):
        limit = 0
    if limit < 1 or limit > 5000:
        errors.append(ValidateError(message = "GitHub limit must be from 1 to 5000."))

    return errors


def _all_blocks_dropped_message(columns: list[Any]) -> str:
    # Reword only: Data Designer's profiler also fails when no generated column is kept (#10836).
    names = [
        column.name
        for column in columns
        if getattr(column, "drop", False) and column.name != "_internal_row_id"
    ]
    listed = f" ({', '.join(names)})" if names else ""
    return (
        f'Every block is set to "Keep out of final dataset"{listed}. Turn that off on at least '
        "one block: source data fields cannot be the only columns in the output."
    )


class _FixedSeedColumns:
    def __init__(self, names: tuple[str, ...]) -> None:
        self._names = list(names)

    def get_column_names(self) -> list[str]:
        return list(self._names)


def _collect_validation_errors(
    recipe: dict[str, Any],
    seed_columns: tuple[str, ...] | None = None,
) -> list[ValidateError]:
    try:
        from data_designer.engine.compiler import (
            _add_internal_row_id_column_if_needed,
            _get_allowed_references,
            _resolve_and_add_seed_columns,
        )
        from data_designer.engine.validation import (
            ViolationLevel,
            validate_data_designer_config,
        )
    except ImportError:
        return []

    try:
        builder = build_config_builder(recipe)
        if seed_columns is None:
            designer = create_data_designer(recipe)
            seed_reader = designer._create_resource_provider(  # type: ignore[attr-defined]
                "validate-configuration",
                builder,
            ).seed_reader
        else:
            seed_reader = _FixedSeedColumns(seed_columns)
        config = builder.build()
        _resolve_and_add_seed_columns(config, seed_reader)
        _add_internal_row_id_column_if_needed(config)
        violations = validate_data_designer_config(
            columns = config.columns,
            processor_configs = config.processors or [],
            allowed_references = _get_allowed_references(config),
        )
    except (TypeError, ValueError, AttributeError):
        return []

    errors: list[ValidateError] = []
    for violation in violations:
        if violation.level != ViolationLevel.ERROR:
            continue
        code = getattr(violation.type, "value", None)
        path = violation.column if violation.column else None
        if code == "all_columns_dropped":
            message = _all_blocks_dropped_message(config.columns)
        else:
            message = str(violation.message).strip() or "Validation failed."
        errors.append(
            ValidateError(
                message = message,
                path = path,
                code = code,
            )
        )
    return errors


def _cycle_names(columns: list[Any]) -> list[str]:
    import data_designer.lazy_heavy_imports as lazy

    names = {column.name for column in columns}
    graph = lazy.nx.DiGraph()
    for column in columns:
        for required in getattr(column, "required_columns", []):
            if required in names:
                graph.add_edge(required, column.name)
    try:
        return [edge[0] for edge in lazy.nx.find_cycle(graph)]
    except lazy.nx.NetworkXNoCycle:
        return []


def _template_filter_errors(name: str, template: Any, env: Any) -> list[ValidateError]:
    from jinja2 import TemplateSyntaxError, nodes

    if not isinstance(template, str):
        return []
    try:
        ast = env.parse(template)
    except TemplateSyntaxError:
        return []
    banned = sorted({node.name for node in ast.find_all(nodes.Filter)} - set(env.filters))
    allowed = ", ".join(sorted(env.filters))
    return [
        ValidateError(
            message = f"'{name}' uses the `{filter_name}` filter, which data recipes don't allow. "
            f"Allowed filters: {allowed}.",
            path = name,
            code = "unsupported_filter",
        )
        for filter_name in banned
    ]


def _recipe_check_errors(recipe: dict[str, Any]) -> list[ValidateError]:
    """What the engine's validate skips and a run then fails on: duplicate block names, unknown model
    aliases, reference loops and template filters the engine refuses."""
    try:
        from data_designer.engine.dataset_builders.utils.dag import (
            topologically_sort_column_configs,
        )
        from data_designer.engine.dataset_builders.utils.errors import (
            DAGCircularDependencyError,
        )
        from data_designer.engine.processing.ginja.environment import (
            UserTemplateSandboxEnvironment,
        )
    except ImportError:
        return []

    errors: list[ValidateError] = []
    names = Counter(
        column.get("name") for column in recipe.get("columns") or [] if isinstance(column, dict)
    )
    for name, count in names.items():
        if isinstance(name, str) and name and count > 1:
            errors.append(
                ValidateError(
                    message = f"{count} blocks are named '{name}'. Give each block its own name.",
                    path = name,
                    code = "duplicate_column_name",
                )
            )

    builder = build_config_builder(recipe)
    config = builder.build()
    aliases = {model_config.alias for model_config in builder.model_configs}
    for column in config.columns:
        alias = getattr(column, "model_alias", None)
        if isinstance(alias, str) and alias not in aliases:
            errors.append(
                ValidateError(
                    message = f"'{column.name}' uses model '{alias}', which isn't one of this recipe's models.",
                    path = column.name,
                    code = "unknown_model_alias",
                )
            )

    try:
        topologically_sort_column_configs(list(config.columns))
    except DAGCircularDependencyError:
        loop = _cycle_names(list(config.columns))
        shown = f" ({' -> '.join([*loop, loop[0]])})" if loop else ""
        errors.append(
            ValidateError(
                message = f"Blocks reference each other in a loop{shown}. Remove one of the references.",
                path = loop[0] if loop else None,
                code = "circular_reference",
            )
        )

    env = UserTemplateSandboxEnvironment()
    for column in config.columns:
        for field in _TEMPLATE_FIELDS:
            errors.extend(_template_filter_errors(column.name, getattr(column, field, None), env))
    for processor in config.processors or []:
        template = getattr(processor, "template", None)
        if isinstance(template, dict):
            errors.extend(_template_filter_errors(processor.name, json.dumps(template), env))
    return errors


def _patch_local_providers(recipe: dict[str, Any]) -> None:
    """Strip is_local and fill a dummy endpoint so validation doesn't choke. Strict `is True` matches
    _inject_local_providers: truthy non-boolean values aren't treated as local."""
    for provider in recipe.get("model_providers", []):
        if not isinstance(provider, dict):
            continue
        if provider.pop("is_local", None) is True:
            provider["endpoint"] = "http://127.0.0.1"
            # Same as _inject_local_providers: external-only auth must not ride along to loopback.
            provider["api_key"] = None
            provider.pop("api_key_env", None)
            provider.pop("extra_headers", None)
            provider.pop("extra_body", None)


def _refused(exc: HTTPException) -> ValidateResponse:
    # An access refusal is already worded for the user; str(exc) would prefix it with "403: ".
    detail = str(exc.detail)
    return ValidateResponse(valid = False, errors = [ValidateError(message = detail)], raw_detail = detail)


@router.post("/validate", response_model = ValidateResponse)
def validate(payload: RecipePayload, via_api_key: ViaApiKey = False) -> ValidateResponse:
    recipe = payload.recipe
    if not recipe.get("columns"):
        return ValidateResponse(
            valid = False,
            errors = [ValidateError(message = "Recipe must include columns.")],
        )
    if recipe_has_stdio_mcp(recipe):
        require_ui_session_for_local_commands(via_api_key)
    require_ui_session_for_host_env_credentials(recipe, via_api_key)

    _patch_local_providers(recipe)
    _resolve_seed_endpoint(recipe)

    github_source = _github_seed_source(recipe)
    if github_source is not None:
        static_errors = _validate_github_seed_static(github_source)
        if static_errors:
            return ValidateResponse(valid = False, errors = static_errors)
        try:
            build_config_builder(recipe)
        except ModuleNotFoundError as exc:
            # Only data_designer's own ImportError is bypassed (optional dep).
            if not (exc.name or "").startswith("data_designer"):
                raise
            logger.debug(
                "data_designer not installed; deferring full config validation to run start",
                missing_module = exc.name,
            )
        except HTTPException as exc:
            return _refused(exc)
        except Exception as exc:
            logger.error(
                "data_recipe.validate.github_config_failed",
                error = str(exc),
                exc_info = True,
            )
            detail = safe_error_detail(exc, fallback = "Validation failed.")
            return ValidateResponse(
                valid = False,
                errors = [ValidateError(message = detail)],
                raw_detail = detail,
            )
        else:
            # The same checks as any other seed, with the GitHub columns standing in for the scrape.
            try:
                errors = _collect_validation_errors(recipe, _GITHUB_SEED_COLUMNS)
                errors += _recipe_check_errors(recipe)
            except Exception as exc:
                # A block named like a GitHub column collides with it, as it would at run start.
                errors = [ValidateError(message = safe_curated_detail(exc, fallback = "Validation failed."))]
            if errors:
                return ValidateResponse(valid = False, errors = errors)
        return ValidateResponse(valid = True, raw_detail = _GITHUB_VALIDATE_NOTE)

    try:
        validate_recipe(recipe)
    except RuntimeError as exc:
        raise log_and_http_error(
            exc,
            503,
            safe_error_detail(exc),
            event = "data_recipe.validate.service_unavailable",
            log = logger,
        ) from exc
    except HTTPException as exc:
        return _refused(exc)
    except Exception as exc:
        logger.error(
            "data_recipe.validate.recipe_failed",
            error = str(exc),
            exc_info = True,
        )
        detail = safe_curated_detail(exc, fallback = "Validation failed.")
        try:
            parsed_errors = _collect_validation_errors(recipe)
        except Exception:
            # The seed is re-read here; keep errors answerable instead of a 500.
            parsed_errors = []
        return ValidateResponse(
            valid = False,
            errors = parsed_errors or [ValidateError(message = detail)],
            raw_detail = detail,
        )

    errors = _recipe_check_errors(recipe)
    if errors:
        return ValidateResponse(valid = False, errors = errors)
    return ValidateResponse(valid = True)
