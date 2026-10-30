# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Flows, their stored runs, player saves, triggers and assets in the account's studio.db and flows/assets."""

from .factory_base import Factory, seeder

MARKER = "flows-matrix-sentinel"
EDITED = "flows-matrix-edited"
FLOW_ID = "flows-matrix-flow"
BROKEN_FLOW_ID = "flows-matrix-broken"
RUN_ID = "flows-matrix-run"
SLOT = "matrix-slot"
TEXT_NODE = "text"
HOOK_NODE = "hook"


def _node(nid: str, type_: str, **params) -> dict:
    return {"id": nid, "type": type_, "v": 1, "params": params, "pos": [0, 0]}


def _flow_edge(src: str, dst: str) -> dict:
    return {
        "id": f"{src}-next-{dst}",
        "source": src,
        "source_port": "next",
        "target": dst,
        "target_port": "in",
        "kind": "flow",
    }


def _doc(flow_id: str, name: str, nodes: list[dict], edges: list[dict] = ()) -> dict:
    return {
        "format": "unsloth-flow",
        "version": 1,
        "id": flow_id,
        "name": name,
        "nodes": nodes,
        "edges": list(edges),
    }


FLOW_DOC = _doc(
    FLOW_ID,
    MARKER,
    [
        _node(TEXT_NODE, "input.text", value = MARKER),
        _node("world", "world.spec", world = {"title": MARKER, "intro_text": MARKER}),
        _node("start", "story.start", title = MARKER, mode = "fixed"),
        _node("scene", "story.scene", title = MARKER, lines = f"Narrator: {MARKER}"),
        _node("end", "story.ending", title = MARKER, kind = "good"),
        _node(HOOK_NODE, "trigger.webhook", enabled = False),
    ],
    [_flow_edge("start", "scene"), _flow_edge("scene", "end")],
)
PUT_BODY = {"doc": _doc(FLOW_ID, EDITED, [_node(TEXT_NODE, "input.text", value = EDITED)])}
ASSET_REF: dict = {}


@seeder("flow")
def seed_flow(account) -> dict[str, str]:
    from core.flows import triggers
    from storage import flows_db
    from utils.account_context import run_as

    def seed() -> None:
        flows_db.save_flow(FLOW_DOC)
        flows_db.create_run(RUN_ID, FLOW_ID, {})
        flows_db.save_node_result(RUN_ID, TEXT_NODE, "done", {"text": MARKER}, None)
        flows_db.append_event(
            RUN_ID,
            1,
            {"seq": 1, "run_id": RUN_ID, "type": "run.completed", "node_id": None, "data": {}},
        )
        flows_db.update_run(
            RUN_ID,
            status = "completed",
            outputs = {TEXT_NODE: MARKER},
            started_at = 1000,
            finished_at = 2000,
        )
        flows_db.put_session(FLOW_ID, SLOT, {"note": MARKER}, MARKER)
        flows_db.replace_triggers(
            FLOW_ID,
            [{"node_id": HOOK_NODE, "kind": "webhook", "config": {"enabled": False}}],
            triggers.new_token,
        )

    run_as(account, seed)
    return {"flow_id": FLOW_ID, "run_id": RUN_ID, "slot": SLOT, "node_id": TEXT_NODE}


@seeder("flow-uncompilable")
def seed_uncompilable_flow(account) -> dict[str, str]:
    from storage import flows_db
    from utils.account_context import run_as

    run_as(
        account,
        flows_db.save_flow,
        _doc(BROKEN_FLOW_ID, MARKER, [_node(MARKER, "matrix.no_such_block")]),
    )
    return {"flow_id": BROKEN_FLOW_ID}


@seeder("flow-asset")
def seed_flow_asset(account) -> dict[str, str]:
    from core.flows import assets
    from utils.account_context import run_as

    digest, _mime = run_as(account, assets.store, MARKER.encode(), "text/plain")
    return {"ref": f"sha256:{digest}"}


_OWNER_ONLY = (
    "owner-only route: a managed account gets 403 whether or not the flow is its own, and the owner "
    "gets 404 for another account's flow"
)
_IDEMPOTENT = (
    "idempotent delete in the caller's own studio.db: another account's id is a no-op there, and the "
    "unchanged snapshot proves the seeded account's rows survive"
)

FACTORIES = {
    "routes.flows:GET:/{flow_id}": Factory("flow", fragment = MARKER, absent = MARKER),
    "routes.flows:PUT:/{flow_id}": Factory(
        "flow",
        PUT_BODY,
        fragment = EDITED,
        absent = MARKER,
        owner = (200,),
        wrong = (200,),
        reason = (
            "upsert by id: an account without the flow saves its own copy in its own studio.db; the "
            "unchanged snapshot proves the seeded account's flow is untouched"
        ),
    ),
    "routes.flows:DELETE:/{flow_id}": Factory(
        "flow", success = 204, owner = (204,), wrong = (204,), reason = _IDEMPOTENT
    ),
    "routes.flows:GET:/runs/{run_id}": Factory("flow", fragment = MARKER, absent = MARKER),
    "routes.flows:POST:/runs/{run_id}/cancel": Factory("flow", fragment = '"status":"completed"'),
    "routes.flows:GET:/runs/{run_id}/events": Factory("flow", fragment = MARKER, absent = MARKER),
    "routes.flows:POST:/runs/{run_id}/events": Factory("flow", fragment = MARKER, absent = MARKER),
    "routes.flows:GET:/{flow_id}/runs": Factory(
        "flow",
        fragment = RUN_ID,
        absent = RUN_ID,
        owner = (200,),
        wrong = (200,),
        reason = (
            "no flow lookup: another account's id lists that account's own runs for it, which is "
            "empty, and the absent run id proves none of the seeded account's leak"
        ),
    ),
    "routes.flows:POST:/{flow_id}/runs": Factory(
        "flow-uncompilable",
        {},
        success = 400,
        fragment = "matrix.no_such_block",
        reason = (
            "covered to the compile guard on the stored document: a compilable flow would start an "
            "engine task that outlives the request"
        ),
    ),
    "routes.flows:POST:/{flow_id}/plan": Factory("flow", {}, fragment = '"gpu_ops":0'),
    "routes.flows:POST:/{flow_id}/invoke": Factory(
        "flow",
        {"entry": TEXT_NODE},
        success = 422,
        fragment = "is not an entry point",
        reason = "covered to the entry point check on the stored document, so no node ever runs",
    ),
    "routes.flows:GET:/{flow_id}/latest-outputs": Factory("flow", fragment = MARKER, absent = MARKER),
    "routes.flows:GET:/{flow_id}/nodes/{node_id}/history": Factory(
        "flow", fragment = MARKER, absent = MARKER
    ),
    "routes.flows:GET:/{flow_id}/story": Factory("flow", fragment = MARKER, absent = MARKER),
    "routes.flows:GET:/{flow_id}/world": Factory("flow", fragment = MARKER, absent = MARKER),
    "routes.flows:GET:/{flow_id}/export": Factory(
        "flow", fragment = MARKER, absent = MARKER, query = {"mode": "json"}
    ),
    "routes.flows:GET:/{flow_id}/sessions": Factory("flow", fragment = MARKER, absent = MARKER),
    "routes.flows:GET:/{flow_id}/sessions/{slot}": Factory("flow", fragment = MARKER, absent = MARKER),
    "routes.flows:PUT:/{flow_id}/sessions/{slot}": Factory(
        "flow", {"state": {"note": EDITED}, "label": EDITED}, fragment = EDITED
    ),
    "routes.flows:DELETE:/{flow_id}/sessions/{slot}": Factory(
        "flow", success = 204, owner = (204,), wrong = (204,), reason = _IDEMPOTENT
    ),
    "routes.flows:GET:/{flow_id}/capabilities": Factory("flow", fragment = '"can_approve":false'),
    "routes.flows:POST:/{flow_id}/approve": Factory(
        "flow",
        {"triggers": True},
        right = (403,),
        wrong = (403,),
        self_expected = (200,),
        reason = _OWNER_ONLY,
    ),
    "routes.flows:GET:/{flow_id}/triggers": Factory("flow", fragment = '"owner_only":true'),
    "routes.flows:POST:/{flow_id}/triggers/{node_id}/rotate": Factory(
        "flow",
        right = (403,),
        wrong = (403,),
        self_expected = (200,),
        reason = _OWNER_ONLY,
        extra_params = {"node_id": HOOK_NODE},
    ),
    "routes.flows:GET:/assets/{ref}": Factory("flow-asset", fragment = MARKER, absent = MARKER),
}

SKIPPED = {
    "routes.flows:POST:/examples/{example_id}/copy": (
        "not account-owned: the id names a bundled example, the same for every account"
    ),
}
