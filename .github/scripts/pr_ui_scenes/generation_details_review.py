"""Replay identical progress frames through each build's chat UI."""

import asyncio
import json
from pathlib import Path

from pr_ui_scenes._common import api_delete, api_get, api_post
from pr_ui_scenes.chat_stream_content_parity import (
    SHIM,
    _ScriptedStreamServer,
    _sse,
    _HEAD,
    _DONE,
    _delta,
    _SETTINGS,
)
from studio_test_kit.auth import seed_init_script
from studio_test_kit.ui import open_chat, send_prompt


async def drive(
    session,
    out_dir: Path,
    label: str,
    mode = "eta",
    **kwargs,
):
    total = 30382
    samples = [
        (2049, 1230),
        (4097, 2643),
        (6145, 4180),
        (8193, 5911),
        (10241, 7781),
        (12289, 9807),
        (14337, 11955),
        (16385, 14277),
    ]
    script = [(0, _HEAD)]
    for processed, elapsed in samples:
        script.append(
            (
                0.25,
                _sse(
                    {
                        "type": "generation_progress",
                        "phase": "prefill",
                        "processed": processed,
                        "total": total,
                        "cached": 0,
                        "time_ms": elapsed,
                        "tok_per_s": processed * 1000 / elapsed,
                    }
                ),
            )
        )
    script.extend(
        [
            (45, _delta("Identical recorded progress frames.")),
            (
                0.2,
                _sse(
                    {
                        "type": "generation_progress",
                        "phase": "decode",
                        "tokens": 3060,
                        "tok_per_s": 58.25,
                        "draft_n": 5030,
                        "draft_accepted": 1791,
                    }
                ),
            ),
            (15, _delta("", finish = "stop")),
            (0, _DONE),
        ]
    )
    server = _ScriptedStreamServer([script])
    try:
        for item in api_get(session, "/api/providers/") or []:
            if item.get("display_name") == "Generation Evidence":
                api_delete(session, f"/api/providers/{item['id']}")
        provider = api_post(
            session,
            "/api/providers/",
            {
                "provider_type": "custom",
                "display_name": "Generation Evidence",
                "base_url": f"http://127.0.0.1:{server.port}/v1",
                "models": ["progress-replay"],
                "available_models": ["progress-replay"],
            },
        )
        pid = provider["id"]
        extra = dict(_SETTINGS)
        extra.update(
            {
                "unsloth_chat_last_external_checkpoint": f"external::{pid}::progress-replay",
                "unsloth_locale": "de" if mode == "translation" else "en",
                "unsloth_chat_preferences": json.dumps(
                    {"version": 1, "state": {"showGenerationDetails": True, "showReplyTimes": True}}
                ),
            }
        )
        init = seed_init_script(session, [], connections_enabled = True, extra_local_storage = extra)
        shim = SHIM.replace("__SOURCE__", f"http://127.0.0.1:{server.port}/stream")
        shim = shim.replace(
            "if (!/chat\\/completions|generate\\/stream/.test(url))",
            'if (url.includes("/api/inference/chat-runs/active")) return new Response("{}", {status:404});\n'
            "    if (!/chat\\/completions|generate\\/stream/.test(url))",
        )
        async with open_chat(
            session.base_url,
            init_scripts = [init, shim],
            viewport = (1400, 900),
            headless = True,
            bypass_csp = True,
        ) as sp:
            page = sp.page
            try:
                await page.wait_for_function(
                    "id => (localStorage.getItem('unsloth_chat_external_providers') || '').includes(id)",
                    arg = pid,
                )
            except Exception:
                await sp.screenshot(out_dir / f"{label}_bootstrap_failure.png", full_page = False)
                print(
                    f"[{label}] bootstrap body={(await page.locator('body').inner_text())[-4000:]}",
                    flush = True,
                )
                raise
            print(f"[{label}] provider synced", flush = True)
            await page.evaluate("() => document.fonts.ready")
            await page.wait_for_function(
                '() => !document.body.innerText.includes("Unsloth isn\'t running")'
            )
            await send_prompt(sp, "Show the recorded prompt-processing progress.")
            print(f"[{label}] prompt submitted", flush = True)
            prefill = page.locator('[data-slot="prefill-progress"]')
            try:
                await prefill.wait_for(state = "visible", timeout = 30000)
            except Exception:
                await sp.screenshot(out_dir / f"{label}_failure.png", full_page = False)
                print(
                    f"[{label}] served={server.served}; body={(await page.locator('body').inner_text())[-4000:]}",
                    flush = True,
                )
                raise
            await page.wait_for_function(
                "() => document.querySelector('[data-slot=prefill-progress]')?.innerText.includes('53%')"
            )
            await page.wait_for_function(
                "() => !document.body.innerText.includes(\"Unsloth isn't running\") && !document.body.innerText.includes('Could not sync pinned chats')"
            )
            facts = {
                "input": "identical recorded SSE, no model execution",
                "prefill": await prefill.inner_text(),
            }
            first = out_dir / f"{label}_prefill.png"
            await sp.screenshot(first, full_page = False)
            shots = [first]
            if mode == "translation":
                live = page.locator('[data-slot="live-generation-line"]')
                await live.wait_for(state = "visible", timeout = 60000)
                facts["decode"] = await live.inner_text()
                second = out_dir / f"{label}_decode.png"
                await sp.screenshot(second, full_page = False)
                shots.append(second)
            return shots, facts
    finally:
        server.close()
