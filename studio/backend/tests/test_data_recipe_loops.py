# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key
from routes.data_recipe.library import router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    return TestClient(app)


def _recipe(recipe_id = "r1") -> dict:
    return {
        "id": recipe_id,
        "name": "Recipe",
        "payload": {"recipe": {}},
        "createdAt": 1000,
        "updatedAt": 2000,
    }


def _loop(loop_id = "l1", recipe_id = "r1", **extra) -> dict:
    return {"id": loop_id, "recipeId": recipe_id, "createdAt": 1000, **extra}


def test_loop_history_round_trips_and_later_saves_replace_it():
    client = _client()
    client.put("/recipes/r1", json = _recipe())
    first = _loop(status = "running", rounds = [{"index": 1, "result": {"score": 0.55}}])
    assert client.put("/recipes/r1/loops/l1", json = first).status_code == 204
    second = _loop(
        status = "target_reached",
        rounds = [
            {"index": 1, "result": {"score": 0.55}},
            {"index": 2, "dataset": {"fromFailures": 40}, "result": {"score": 0.8}},
        ],
    )
    assert client.put("/recipes/r1/loops/l1", json = second).status_code == 204

    loops = client.get("/recipes/r1/loops").json()["loops"]
    assert loops == [second]


def test_loop_needs_its_recipe_and_cannot_move_to_another_one():
    client = _client()
    assert client.put("/recipes/missing/loops/l1", json = _loop(recipe_id = "missing")).status_code == 404
    client.put("/recipes/r1", json = _recipe("r1"))
    client.put("/recipes/r2", json = _recipe("r2"))
    assert client.put("/recipes/r1/loops/l1", json = _loop()).status_code == 204
    moved = client.put("/recipes/r2/loops/l1", json = _loop(recipe_id = "r2"))
    assert moved.status_code == 404
    assert client.put("/recipes/r1/loops/l2", json = _loop("l3")).status_code == 400
    assert [l["id"] for l in client.get("/recipes/r1/loops").json()["loops"]] == ["l1"]
    assert client.get("/recipes/r2/loops").json() == {"loops": []}


def test_deleting_the_recipe_drops_its_loops():
    client = _client()
    client.put("/recipes/r1", json = _recipe())
    client.put("/recipes/r1/loops/l1", json = _loop())
    client.delete("/recipes/r1")
    assert client.get("/recipes/r1/loops").json() == {"loops": []}
