"""Harness only: the Settings > Sandbox API end to end on a real Windows host, through the real routes.

Auth dependencies are overridden to the installation owner exactly as the unit tests do; everything
else (settings store, MXC runtime, host prep job, probes, tools) is real. Payloads are harmless.
Usage: python sb_e2e.py
"""

import os
import sys
import time

sys.path.insert(0, "studio/backend")
os.environ.pop("UNSLOTH_MXC_ALLOW_DACL_FALLBACK", None)
os.environ.pop("UNSLOTH_MXC_PERSISTENT_READ_GRANTS", None)

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import routes.settings as settings  # noqa: E402
from core.inference import tools  # noqa: E402
from utils.account_context import OWNER, bind_account, reset_account  # noqa: E402


def client(host):
    app = FastAPI()
    app.include_router(settings.router)

    async def subject():
        token = bind_account(OWNER)
        try:
            yield OWNER.username
        finally:
            reset_account(token)

    app.dependency_overrides[settings.get_current_subject] = subject
    app.dependency_overrides[settings.authenticated_via_api_key] = lambda: False
    return TestClient(app, client = (host, 50000), raise_server_exceptions = False)


def show(label, status):
    body = status.json() if hasattr(status, "json") else status
    if not isinstance(body, dict) or "python" not in body:
        print(f"STATUS {label} http={getattr(status, 'status_code', '?')} {str(body)[:300]}", flush = True)
        return body
    win = body.get("windows") or {}
    print(
        f"STATUS {label} python={body['python']['backend']}/{body['python']['available']} "
        f"terminal={body['terminal']['backend']}/{body['terminal']['available']} shell={body.get('terminal_shell')} "
        f"dacl={win.get('allow_dacl_fallback')} saved={win.get('allow_dacl_fallback_saved')} "
        f"locked={win.get('dacl_locked_by_environment')} runtime={win.get('runtime_installed')} "
        f"prep_missing={win.get('host_prep_missing')} restored={body.get('grants_restored')}",
        flush = True,
    )
    return body


def run_tool(label, tool, payload):
    tools._last_tool_execution_record = None
    key = "code" if tool == "python" else "command"
    started = time.monotonic()
    out = tools.execute_tool(tool, {key: payload}, session_id = "__LOCALID_sb_e2e", timeout = 150) or ""
    rec = tools._last_tool_execution_record
    print(
        f"TOOL {label} {tool} mode={getattr(rec, 'effective_mode', None)} backend={getattr(rec, 'backend', None)} "
        f"s={time.monotonic() - started:.1f} ok={'SB_OK' in out} | {out.strip()[-160:]!r}",
        flush = True,
    )


def main():
    local = client("127.0.0.1")
    with local:
        show("initial", local.get("/api/settings/sandbox", params = {"refresh": "true"}))
        run_tool("off", "python", "print('SB_OK')")
        show("enable", local.put("/api/settings/sandbox", json = {"allow_dacl_fallback": True}))
        with client("203.0.113.9") as remote:
            r = remote.post("/api/settings/sandbox/prepare")
            print(f"REMOTE_PREPARE http={r.status_code} {r.text[:200]}", flush = True)
        r = local.post("/api/settings/sandbox/prepare")
        print(f"PREPARE_START http={r.status_code} {r.text[:200]}", flush = True)
        deadline = time.monotonic() + 1200
        job = r.json() if r.status_code == 200 else {}
        while job.get("state") == "running" and time.monotonic() < deadline:
            time.sleep(5)
            job = local.get("/api/settings/sandbox/prepare").json()
        print(f"PREPARE_DONE state={job.get('state')} exit={job.get('exit_code')} steps={job.get('steps')} tail={job.get('output_tail', [])[-3:]}", flush = True)
        show("prepared", local.get("/api/settings/sandbox", params = {"refresh": "true"}))
        run_tool("on", "python", "print('SB_OK')")
        run_tool("on", "terminal", "git --version && echo SB_OK")
        run_tool("on_again", "python", "print('SB_OK')")
        show("disable", local.put("/api/settings/sandbox", json = {"allow_dacl_fallback": False}))
        run_tool("off_again", "python", "print('SB_OK')")
        os.environ["UNSLOTH_MXC_ALLOW_DACL_FALLBACK"] = "0"
        r = local.put("/api/settings/sandbox", json = {"allow_dacl_fallback": True})
        print(f"ENV_LOCK_PUT http={r.status_code} {r.text[:200]}", flush = True)
        show("env_locked", local.get("/api/settings/sandbox", params = {"refresh": "true"}))


if __name__ == "__main__":
    main()
