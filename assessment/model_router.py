"""Task 4: SQLite sliding-window token limiter and 429/timeout failover router."""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout, web

WINDOW_SECONDS = 60
TOKEN_LIMIT = 50_000


class TokenWindow:
    """Small durable sliding window; one lock makes select+insert atomic per process."""
    def __init__(self, database: str, limit: int = TOKEN_LIMIT) -> None:
        Path(database).parent.mkdir(parents=True, exist_ok=True) if Path(database).parent != Path(".") else None
        self.db = sqlite3.connect(database, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS token_events (tenant TEXT NOT NULL, ts REAL NOT NULL, tokens INTEGER NOT NULL)")
        self.db.execute("CREATE INDEX IF NOT EXISTS token_events_tenant_ts ON token_events(tenant, ts)")
        self.db.commit()
        self.limit = limit
        self.lock = asyncio.Lock()

    async def reserve(self, tenant: str, tokens: int) -> bool:
        if tokens < 0:
            return False
        now = time.time()
        cutoff = now - WINDOW_SECONDS
        async with self.lock:
            # SQLite parameterization prevents tenant-key SQL injection.
            self.db.execute("DELETE FROM token_events WHERE ts < ?", (cutoff,))
            used = self.db.execute("SELECT COALESCE(SUM(tokens), 0) FROM token_events WHERE tenant = ? AND ts >= ?", (tenant, cutoff)).fetchone()[0]
            if used + tokens > self.limit:
                self.db.commit()
                return False
            self.db.execute("INSERT INTO token_events (tenant, ts, tokens) VALUES (?, ?, ?)", (tenant, now, tokens))
            self.db.commit()
            return True

    def close(self) -> None:
        self.db.close()


def estimated_tokens(payload: dict[str, Any]) -> int:
    """Conservative estimate when an upstream tokenizer is unavailable."""
    messages = payload.get("messages", [])
    text = " ".join(str(item.get("content", "")) for item in messages if isinstance(item, dict))
    # Typical English approximation plus requested output budget.
    return max(1, (len(text) + 3) // 4) + int(payload.get("max_tokens", 0) or 0)


def gateway_error(status: int, code: str, message: str) -> web.Response:
    return web.json_response({"error": {"code": code, "message": message}}, status=status)


async def invoke(session: ClientSession, url: str, payload: dict[str, Any]) -> tuple[int, bytes]:
    # `total` includes read time, providing the requested three-second hard cutoff.
    async with session.post(url, json=payload, timeout=ClientTimeout(total=3)) as response:
        return response.status, await response.read()


async def completion(request: web.Request) -> web.Response:
    tenant = request.headers.get("X-API-Key")
    if not tenant:
        return gateway_error(401, "unauthorized", "API key is required")
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError
        tokens = estimated_tokens(payload)
    except (ValueError, TypeError, json.JSONDecodeError):
        return gateway_error(400, "invalid_request", "Request must be a valid completion object")
    if not await request.app["limiter"].reserve(tenant, tokens):
        return gateway_error(429, "rate_limit_exceeded", "Tenant token limit exceeded")

    session: ClientSession = request.app["http"]
    primary = request.app["primary_url"]
    backup = request.app["backup_url"]
    try:
        status, body = await invoke(session, primary, payload)
        if status != 429:
            if 200 <= status < 300:
                return web.Response(status=status, body=body, content_type="application/json")
            return gateway_error(502, "upstream_error", "Primary model request failed")
    except (asyncio.TimeoutError, OSError):
        pass  # timeout and network errors are failover conditions

    try:
        status, body = await invoke(session, backup, payload)
        if 200 <= status < 300:
            return web.Response(status=status, body=body, content_type="application/json")
        return gateway_error(502, "upstream_error", "Model provider request failed")
    except (asyncio.TimeoutError, OSError):
        return gateway_error(503, "model_unavailable", "No model provider is currently available")


async def on_startup(app: web.Application) -> None:
    app["http"] = ClientSession()


async def on_cleanup(app: web.Application) -> None:
    await app["http"].close()
    app["limiter"].close()


def create_app(primary_url: str | None = None, backup_url: str | None = None, database: str | None = None) -> web.Application:
    app = web.Application()
    app["primary_url"] = primary_url or os.getenv("PRIMARY_MODEL_URL", "http://127.0.0.1:9001/v1/chat/completions")
    app["backup_url"] = backup_url or os.getenv("BACKUP_MODEL_URL", "http://127.0.0.1:9002/v1/chat/completions")
    app["limiter"] = TokenWindow(database or os.getenv("RATE_LIMIT_DB", "rate_limit.sqlite3"))
    app.router.add_post("/v1/chat/completions", completion)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


if __name__ == "__main__":
    web.run_app(create_app(), host="127.0.0.1", port=int(os.getenv("PORT", "8082")))
