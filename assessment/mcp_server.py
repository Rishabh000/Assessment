"""Task 1: official MCP SDK server over stdio with strict Pydantic validation."""
from __future__ import annotations

import asyncio
import logging
import sys
from decimal import Decimal

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp import types
from mcp.types import ErrorData, TextContent, Tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

INVALID_PARAMS = -32602  # JSON-RPC "Invalid params"

# Never configure a stdout handler: it corrupts stdio JSON-RPC framing.
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)
    customer_id: str = Field(pattern=r"^CUST-[0-9]{5}$")


class RefundInput(CustomerInput):
    amount: Decimal = Field(gt=0, max_digits=12, decimal_places=2)
    reason: str = Field(min_length=10, max_length=1000)


server = Server("strict-customer-service")


def validated(model: type[BaseModel], arguments: dict | None) -> BaseModel:
    """Map validation failures to the JSON-RPC invalid-params error."""
    try:
        # Pydantic v2 already rejects integer-to-string coercion; allowing a JSON
        # number for Decimal is intentional because `amount` is specified as float.
        return model.model_validate(arguments or {})
    except ValidationError as exc:
        # Do not expose a Python traceback; validation detail is useful to callers.
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Invalid tool arguments", data=exc.errors())) from exc


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_customer_record",
            description="Return a customer record by canonical customer ID.",
            inputSchema=CustomerInput.model_json_schema(),
        ),
        Tool(
            name="trigger_refund",
            description="Trigger a refund after strictly validating the request.",
            inputSchema=RefundInput.model_json_schema(),
        ),
    ]


async def call_tool(request: types.CallToolRequest) -> types.ServerResult:
    """Validate inside the request handler so invalid arguments stay RPC errors.

    `Server.call_tool()` helpfully converts all failures into tool `isError`
    results.  This assessment requires JSON-RPC's ``-32602`` instead, so this
    handler is registered directly with the official SDK's server dispatcher.
    """
    name, arguments = request.params.name, request.params.arguments
    if name == "get_customer_record":
        request = validated(CustomerInput, arguments)
        # Replace this stub with a database/API lookup.
        result = {"customer_id": request.customer_id, "status": "active", "tier": "standard"}
    elif name == "trigger_refund":
        request = validated(RefundInput, arguments)
        # Replace this stub with an idempotent payment-provider call.
        result = {"customer_id": request.customer_id, "amount": str(request.amount), "reason": request.reason, "status": "accepted"}
    else:
        raise McpError(ErrorData(code=-32601, message=f"Unknown tool: {name}"))
    return types.ServerResult(types.CallToolResult(content=[TextContent(type="text", text=str(result))]))


# Deliberately bypass Server.call_tool's error-to-isError-result wrapper.
server.request_handlers[types.CallToolRequest] = call_tool


async def main() -> None:
    log.info("starting strict customer MCP server")
    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
