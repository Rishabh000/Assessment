"""Task 2: an HTTP JSON-RPC MCP proxy with tool-level authorization."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from aiohttp import ClientSession, ClientTimeout, web

log = logging.getLogger(__name__)
DEFAULT_ROLES = {"admin-token": "admin", "viewer-token": "viewer"}


def role_from_request(request: web.Request) -> str | None:
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    roles = json.loads(os.getenv("ROLE_TOKENS_JSON", json.dumps(DEFAULT_ROLES)))
    return roles.get(token)


def rpc_error(request_id: Any, code: int, message: str) -> web.Response:
    return web.json_response({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})


async def proxy(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return rpc_error(None, -32700, "Parse error")
    if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0" or not isinstance(payload.get("method"), str):
        return rpc_error(payload.get("id") if isinstance(payload, dict) else None, -32600, "Invalid Request")

    method, params, request_id = payload["method"], payload.get("params", {}), payload.get("id")
    if method == "tools/call":
        name = params.get("name") if isinstance(params, dict) else None
        if not isinstance(name, str):
            return rpc_error(request_id, -32602, "Invalid params: params.name must be a string")
        if name.startswith("admin_") and role_from_request(request) != "admin":
            # The request is intentionally never forwarded in this branch.
            return rpc_error(request_id, -32001, "Unauthorized Tool Call")

    # tools/list and all permitted JSON-RPC methods are transparent forwards.
    downstream = request.app["downstream_url"]
    try:
        async with request.app["http"].post(downstream, json=payload, headers={"Content-Type": "application/json"}) as upstream:
            body = await upstream.read()
            return web.Response(status=upstream.status, body=body, content_type=upstream.headers.get("Content-Type", "application/json").split(";", 1)[0])
    except Exception:
        log.exception("downstream MCP request failed")
        return rpc_error(request_id, -32000, "Downstream MCP unavailable")


async def on_startup(app: web.Application) -> None:
    app["http"] = ClientSession(timeout=ClientTimeout(total=15))


async def on_cleanup(app: web.Application) -> None:
    await app["http"].close()


def create_app(downstream_url: str | None = None) -> web.Application:
    app = web.Application()
    app["downstream_url"] = downstream_url or os.getenv("DOWNSTREAM_MCP_URL", "http://127.0.0.1:9000/mcp")
    app.router.add_post("/mcp", proxy)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    web.run_app(create_app(), host="127.0.0.1", port=int(os.getenv("PORT", "8080")))
