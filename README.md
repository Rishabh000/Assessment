# MCP / LLM gateway assessment

Four small, runnable Python services demonstrating strict MCP tool validation,
an MCP authorization proxy, streaming PII redaction, and a SQLite-backed model
router.

## Install

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
```

## Run

```sh
# Task 1: MCP stdio server. stdout is protocol-only; diagnostics go to stderr.
python -m assessment.mcp_server

# Task 2: MCP HTTP JSON-RPC authorization proxy
DOWNSTREAM_MCP_URL=http://127.0.0.1:9000/mcp python -m assessment.mcp_gateway

# Task 3: streaming LLM proxy
LLM_UPSTREAM_URL=http://127.0.0.1:9001/v1/chat/completions python -m assessment.streaming_guardrail

# Task 4: SQLite rate limiter and fallback router
PRIMARY_MODEL_URL=http://127.0.0.1:9001/v1/chat/completions \
BACKUP_MODEL_URL=http://127.0.0.1:9002/v1/chat/completions \
python -m assessment.model_router
```

The HTTP services listen on `127.0.0.1:8080`, `8081`, and `8082` respectively
(override with `PORT`).  Tokens in the gateway proxy are configured through
`ROLE_TOKENS_JSON`, e.g. `{"admin-token":"admin","viewer-token":"viewer"}`.
The development default is deliberately explicit and should be replaced in
production with JWT verification or an identity-provider introspection call.

Run checks with `pytest`.
