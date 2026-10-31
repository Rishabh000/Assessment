"""Task 3: low-memory SSE LLM proxy with cross-chunk PII redaction."""
from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator

from aiohttp import ClientSession, ClientTimeout, web

# Keep a bounded suffix while a possible PII token is unfinished.  Complete text
# is released immediately; this avoids the fixed 320-character TTFT delay.
HOLD_BACK = 320
PII = re.compile(
    r"(?:\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b|\b\d{3}-\d{2}-\d{4}\b|\b(?:\d[ -]?){12,18}\d\b)"
)


class StreamingRedactor:
    def __init__(self, hold_back: int = HOLD_BACK) -> None:
        self.tail = ""
        self.hold_back = hold_back

    def feed(self, text: str) -> str:
        combined = self.tail + text
        safe_end = self._safe_end(combined)
        safe, self.tail = combined[:safe_end], combined[safe_end:]
        return PII.sub("[REDACTED]", safe)

    def flush(self) -> str:
        value, self.tail = self.tail, ""
        return PII.sub("[REDACTED]", value)

    def _safe_end(self, value: str) -> int:
        """Return the earliest suffix that might grow into one of our patterns."""
        starts: list[int] = []
        # An unfinished email must retain its final local/domain token.  Once a
        # delimiter arrives, the full address is redacted before it is emitted.
        email = re.search(r"[A-Za-z0-9._%+@-]+$", value)
        if email:
            starts.append(email.start())
        # Keep any final SSN/card-like run starting with a digit. Spaces and
        # hyphens are included because providers commonly split formatted cards.
        number = re.search(r"\d[\d -]*$", value)
        if number:
            starts.append(number.start())
        keep_from = min(starts) if starts else len(value)
        # RFC email addresses are at most 254 chars; a small margin also covers
        # formatted cards.  Therefore older data cannot become a supported PII
        # match and can be safely released.
        return max(keep_from, len(value) - self.hold_back)


def redact_sse_event(raw: bytes, redactor: StreamingRedactor, final: bool = False) -> bytes | None:
    """Return a rewritten SSE event, or None when its text is still held back."""
    if raw.strip() == b"data: [DONE]":
        tail = redactor.flush()
        prefix = (b"data: " + json.dumps({"choices": [{"delta": {"content": tail}}]}).encode() + b"\n\n") if tail else b""
        return prefix + raw + b"\n\n"
    if not raw.startswith(b"data: "):
        return raw + b"\n\n"
    try:
        payload = json.loads(raw[6:])
        delta = payload["choices"][0]["delta"]
        content = delta.get("content")
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return raw + b"\n\n"  # preserve non-chat provider events unchanged
    if not isinstance(content, str):
        return raw + b"\n\n"
    emitted = redactor.flush() if final else redactor.feed(content)
    if not emitted:
        return None
    delta["content"] = emitted
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"


async def split_sse(chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    buffer = b""
    async for chunk in chunks:
        buffer += chunk
        while match := re.search(rb"\r?\n\r?\n", buffer):
            event, buffer = buffer[: match.start()], buffer[match.end() :]
            yield event
    if buffer:
        yield buffer


async def chat_completions(request: web.Request) -> web.StreamResponse:
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": {"code": "invalid_request", "message": "Request must be JSON"}}, status=400)
    session: ClientSession = request.app["http"]
    try:
        upstream = await session.post(request.app["upstream_url"], json=body, headers={"Accept": "text/event-stream"})
    except Exception:
        return web.json_response({"error": {"code": "upstream_unavailable", "message": "LLM provider unavailable"}}, status=502)
    if upstream.status >= 400:
        await upstream.release()
        return web.json_response({"error": {"code": "upstream_error", "message": "LLM provider rejected request"}}, status=502)

    response = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
    await response.prepare(request)
    redactor = StreamingRedactor()
    try:
        async for event in split_sse(upstream.content.iter_any()):
            rewritten = redact_sse_event(event, redactor)
            if rewritten:
                await response.write(rewritten)
        remaining = redactor.flush()
        if remaining:
            await response.write(b"data: " + json.dumps({"choices": [{"delta": {"content": remaining}}]}).encode() + b"\n\n")
    finally:
        upstream.release()
    await response.write_eof()
    return response


async def on_startup(app: web.Application) -> None:
    app["http"] = ClientSession(timeout=ClientTimeout(total=None, sock_connect=5))


async def on_cleanup(app: web.Application) -> None:
    await app["http"].close()


def create_app(upstream_url: str | None = None) -> web.Application:
    app = web.Application()
    app["upstream_url"] = upstream_url or os.getenv("LLM_UPSTREAM_URL", "http://127.0.0.1:9001/v1/chat/completions")
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


if __name__ == "__main__":
    web.run_app(create_app(), host="127.0.0.1", port=int(os.getenv("PORT", "8081")))
