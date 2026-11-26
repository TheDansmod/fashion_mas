"""Tool output formats, in-process tools, and the MCP server + client over HTTP."""

import asyncio
import json
import socket

import httpx
import pytest

from frag.mcp_server.server import build_server
from frag.tools.base import (
    blocks_to_item,
    item_to_blocks,
    normalise_blocks,
)
from frag.tools.local import LocalToolsClient
from frag.tools.mcp_client import MCPToolsClient
from tests.fakes import FIXTURE_IMAGE_B64, fake_item


def test_item_blocks_round_trip():
    item = fake_item(101, score=0.75)
    blocks = item_to_blocks(item)
    assert [b["type"] for b in blocks] == ["text", "image_url"]
    assert blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert blocks_to_item(blocks) == item


def test_mcp_image_blocks_are_normalised():
    # what langchain-mcp-adapters returns for MCP TextContent + ImageContent
    mcp_blocks = [
        item_to_blocks(fake_item(102))[0],
        {"type": "image", "base64": FIXTURE_IMAGE_B64, "mime_type": "image/png"},
    ]
    assert normalise_blocks(mcp_blocks) == item_to_blocks(fake_item(102))
    assert blocks_to_item(mcp_blocks) == fake_item(102)


def test_non_item_output():
    assert normalise_blocks("plain") == [{"type": "text", "text": "plain"}]
    assert blocks_to_item([{"type": "text", "text": "Error: no item"}]) is None
    with pytest.raises(ValueError, match="unexpected"):
        normalise_blocks([{"type": "audio"}])


async def _call(tools, name, args):
    (tool,) = [t for t in tools if t.name == name]
    return await tool.ainvoke(args)


SEARCH = {"description": "jeans", "categories": ["JEANS", "SHORTS"], "num_matches": 5}


async def test_local_tools(catalogue):
    client = LocalToolsClient(catalogue)
    tools = await client.get_llm_tools()
    assert [t.name for t in tools] == ["semantic_search", "get_product_categories"]
    blocks = await _call(tools, "semantic_search", SEARCH)
    assert [json.loads(b["text"])["id"] for b in blocks if b["type"] == "text"] == [
        101,
        102,
    ]
    error = await _call(tools, "semantic_search", {**SEARCH, "categories": ["HATS"]})
    assert error[0]["text"].startswith("Error:")
    (cats,) = await _call(tools, "get_product_categories", {})
    assert json.loads(cats["text"]) == catalogue.categories
    assert (await client.get_item(103)).id == 103
    assert await client.get_item(999) is None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
async def mcp_url(catalogue):
    """Serve the fake catalogue with the real MCP server on a local port."""
    port = _free_port()
    server = build_server(catalogue)
    task = asyncio.create_task(
        server.run_async(
            transport="http",
            host="127.0.0.1",
            port=port,
            show_banner=False,
            # uvicorn logs every request, and a traceback when the task is cancelled
            log_level="critical",
        )
    )
    base = f"http://127.0.0.1:{port}"
    async with httpx.AsyncClient() as http:
        for _ in range(100):
            try:
                if (await http.get(f"{base}/health")).status_code == 200:
                    break
            except httpx.TransportError:
                await asyncio.sleep(0.05)
        else:
            pytest.fail("MCP server did not start")
    yield base
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_mcp_health(mcp_url):
    async with httpx.AsyncClient() as http:
        response = await http.get(f"{mcp_url}/health")
    assert response.json() == {"status": "ok", "items": 3}


async def test_mcp_tools_match_local_tools(mcp_url, catalogue):
    remote = MCPToolsClient(f"{mcp_url}/mcp")
    local = LocalToolsClient(catalogue)
    remote_tools = await remote.get_llm_tools()
    local_tools = await local.get_llm_tools()
    assert [t.name for t in remote_tools] == [t.name for t in local_tools]
    for name, args in [
        ("semantic_search", SEARCH),
        ("semantic_search", {**SEARCH, "categories": ["HATS"]}),
        ("get_product_categories", {}),
    ]:
        # the LLM sees identical tool output in both modes
        assert await _call(remote_tools, name, args) == await _call(
            local_tools, name, args
        )
    assert await remote.get_item(102) == await local.get_item(102)
    assert await remote.get_item(999) is None
