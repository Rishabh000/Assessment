import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from mcp.shared.exceptions import McpError
from mcp import types

from assessment.mcp_gateway import create_app as create_gateway
from assessment.mcp_server import CustomerInput, RefundInput, call_tool, validated
from assessment.model_router import TokenWindow, create_app as create_router, estimated_tokens
from assessment.streaming_guardrail import StreamingRedactor, create_app as create_guardrail, redact_sse_event, split_sse


@asynccontextmanager
async def app_client(app: web.Application):
    async with TestServer(app) as server:
        async with TestClient(server) as client:
            yield client, server


# --- Task 1 ---


def test_customer_id_schema():
    assert validated(CustomerInput, {"customer_id": "CUST-12345"}).customer_id == "CUST-12345"
    with pytest.raises(McpError) as exc:
        validated(CustomerInput, {"customer_id": "cust-12345"})
    assert exc.value.error.code == -32602


async def test_call_tool_success_and_invalid_params():
    ok = await call_tool(
        types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name="get_customer_record", arguments={"customer_id": "CUST-00001"}),
        )
    )
    assert isinstance(ok.root, types.CallToolResult)
    assert "CUST-00001" in ok.root.content[0].text

    refund = await call_tool(
        types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(
                name="trigger_refund",
                arguments={"customer_id": "CUST-00001", "amount": 12.5, "reason": "duplicate charge"},
            ),
        )
    )
    assert "accepted" in refund.root.content[0].text

    with pytest.raises(McpError) as exc:
        await call_tool(
            types.CallToolRequest(
                method="tools/call",
                params=types.CallToolRequestParams(
                    name="trigger_refund",
                    arguments={"customer_id": "CUST-00001", "amount": 0, "reason": "duplicate charge"},
                ),
            )
        )
    assert exc.value.error.code == -32602

    with pytest.raises(McpError) as exc:
        await call_tool(
            types.CallToolRequest(
                method="tools/call",
                params=types.CallToolRequestParams(name="not_a_tool", arguments={}),
            )
        )
    assert exc.value.error.code == -32601


def test_refund_rejects_short_reason_and_extra_fields():
    with pytest.raises(McpError):
        validated(RefundInput, {"customer_id": "CUST-00001", "amount": 1, "reason": "too short"})
    with pytest.raises(McpError):
        validated(CustomerInput, {"customer_id": "CUST-00001", "extra": True})


# --- Task 2 ---


async def test_gateway_filters_admin_tools():
    hits = {"count": 0}

    async def downstream(request):
        hits["count"] += 1
        payload = await request.json()
        return web.json_response({"jsonrpc": "2.0", "id": payload.get("id"), "result": {"ok": True}})

    down = web.Application()
    down.router.add_post("/mcp", downstream)
    async with app_client(down) as (_, down_server):
        app = create_gateway(str(down_server.make_url("/mcp")))
        async with app_client(app) as (client, _):
            payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "admin_reset_key"}}

            denied = await client.post("/mcp", json=payload, headers={"Authorization": "Bearer viewer-token"})
            body = await denied.json()
            assert body["error"]["code"] == -32001
            assert hits["count"] == 0

            listed = await client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                headers={"Authorization": "Bearer viewer-token"},
            )
            assert (await listed.json())["result"]["ok"] is True
            assert hits["count"] == 1

            allowed = await client.post("/mcp", json=payload, headers={"Authorization": "Bearer admin-token"})
            assert (await allowed.json())["result"]["ok"] is True
            assert hits["count"] == 2


async def test_gateway_parse_error():
    app = create_gateway("http://127.0.0.1:9/mcp")
    async with app_client(app) as (client, _):
        resp = await client.post("/mcp", data="{not-json", headers={"Content-Type": "application/json"})
        body = await resp.json()
        assert body["error"]["code"] == -32700


# --- Task 3 ---


def test_redactor_catches_pii_across_chunks():
    redactor = StreamingRedactor()
    assert redactor.feed("contact jane@exa") == "contact "
    assert redactor.feed("mple.com please") == "[REDACTED] "
    assert redactor.flush() == "please"


def test_redactor_ssn_and_card():
    redactor = StreamingRedactor()
    assert "[REDACTED]" in redactor.feed("ssn 123-45-6789 done")
    redactor = StreamingRedactor()
    emitted = redactor.feed("card 4111 1111 1111 1111 done")
    emitted += redactor.flush()
    assert "[REDACTED]" in emitted
    assert "4111" not in emitted


def test_redactor_releases_normal_text_without_fixed_holdback():
    redactor = StreamingRedactor()
    assert redactor.feed("Hello, ") == "Hello, "


async def test_sse_splitter_accepts_crlf():
    async def chunks():
        yield b"data: one\r\n\r\ndata: two\n\n"

    assert [event async for event in split_sse(chunks())] == [b"data: one", b"data: two"]


async def test_guardrail_redacts_stream():
    async def upstream(request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b'data: {"choices":[{"delta":{"content":"hi jane@ex"}}]}\n\n')
        await resp.write(b'data: {"choices":[{"delta":{"content":"ample.com bye"}}]}\n\n')
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    up = web.Application()
    up.router.add_post("/v1/chat/completions", upstream)
    async with app_client(up) as (_, up_server):
        app = create_guardrail(str(up_server.make_url("/v1/chat/completions")))
        async with app_client(app) as (client, _):
            resp = await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
            text = await resp.text()
            assert "[REDACTED]" in text
            assert "jane@example.com" not in text


def test_redact_sse_preserves_done():
    redactor = StreamingRedactor()
    redactor.feed("x")
    event = redact_sse_event(b"data: [DONE]", redactor)
    assert b"[DONE]" in event
    assert b"x" in event


# --- Task 4 ---


def test_rate_limiter_sliding_window(tmp_path):
    limiter = TokenWindow(str(tmp_path / "rate.sqlite"), limit=10)
    assert asyncio.run(limiter.reserve("tenant", 7))
    assert not asyncio.run(limiter.reserve("tenant", 4))
    limiter.db.execute("UPDATE token_events SET ts = ts - 61")
    limiter.db.commit()
    assert asyncio.run(limiter.reserve("tenant", 10))
    limiter.close()


def test_estimated_tokens_includes_requested_output():
    assert estimated_tokens({"messages": [{"content": "12345678"}], "max_tokens": 7}) == 9


async def test_router_failsover_on_timeout(tmp_path):
    async def primary_slow(request):
        await asyncio.sleep(4)
        return web.json_response({"choices": [{"message": {"content": "primary"}}]})

    async def backup_ok(request):
        return web.json_response({"choices": [{"message": {"content": "backup"}}]})

    primary = web.Application()
    primary.router.add_post("/v1/chat/completions", primary_slow)
    backup = web.Application()
    backup.router.add_post("/v1/chat/completions", backup_ok)
    async with app_client(primary) as (_, p_server):
        async with app_client(backup) as (_, b_server):
            app = create_router(
                str(p_server.make_url("/v1/chat/completions")),
                str(b_server.make_url("/v1/chat/completions")),
                str(tmp_path / "rl-timeout.sqlite"),
            )
            async with app_client(app) as (client, _):
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"content": "hello"}], "max_tokens": 8},
                    headers={"X-API-Key": "tenant-a"},
                )
                body = await resp.json()
                assert resp.status == 200
                assert body["choices"][0]["message"]["content"] == "backup"


async def test_router_failsover_on_429_and_timeout(tmp_path):
    async def primary_429(request):
        return web.json_response({"error": "slow down"}, status=429)

    async def backup_ok(request):
        return web.json_response({"choices": [{"message": {"content": "backup"}}]})

    primary = web.Application()
    primary.router.add_post("/v1/chat/completions", primary_429)
    backup = web.Application()
    backup.router.add_post("/v1/chat/completions", backup_ok)
    async with app_client(primary) as (_, p_server):
        async with app_client(backup) as (_, b_server):
            app = create_router(
                str(p_server.make_url("/v1/chat/completions")),
                str(b_server.make_url("/v1/chat/completions")),
                str(tmp_path / "rl.sqlite"),
            )
            async with app_client(app) as (client, _):
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"content": "hello"}], "max_tokens": 8},
                    headers={"X-API-Key": "tenant-a"},
                )
                body = await resp.json()
                assert resp.status == 200
                assert body["choices"][0]["message"]["content"] == "backup"


async def test_router_sanitizes_errors(tmp_path):
    async def boom(request):
        raise RuntimeError("secret stack")

    primary = web.Application()
    primary.router.add_post("/v1/chat/completions", boom)
    backup = web.Application()
    backup.router.add_post("/v1/chat/completions", boom)
    async with app_client(primary) as (_, p_server):
        async with app_client(backup) as (_, b_server):
            app = create_router(
                str(p_server.make_url("/v1/chat/completions")),
                str(b_server.make_url("/v1/chat/completions")),
                str(tmp_path / "rl2.sqlite"),
            )
            async with app_client(app) as (client, _):
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"content": "hello"}]},
                    headers={"X-API-Key": "tenant-a"},
                )
                body = await resp.json()
                assert resp.status in {502, 503}
                blob = json.dumps(body)
                assert "secret stack" not in blob
                assert "RuntimeError" not in blob
                assert "error" in body
