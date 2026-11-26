"""MCP server exposing the product catalogue as LLM tools (streamable HTTP)."""

import json

from fastmcp import FastMCP
from loguru import logger as log
from mcp.types import ImageContent, TextContent
from starlette.requests import Request
from starlette.responses import JSONResponse

from frag.catalogue.models import CatalogueItem
from frag.catalogue.service import Catalogue, InvalidCategoryError
from frag.settings import ToolsSettings
from frag.tools.base import (
    GET_DATAPOINT_BY_INDEX,
    GET_DATAPOINT_BY_INDEX_DOC,
    GET_PRODUCT_CATEGORIES,
    GET_PRODUCT_CATEGORIES_DOC,
    SEMANTIC_SEARCH,
    SEMANTIC_SEARCH_DOC,
    item_metadata,
)

Content = TextContent | ImageContent


def _item_content(item: CatalogueItem) -> list[Content]:
    # the MCP twin of tools.base.item_to_blocks; the client normalises it back
    return [
        TextContent(type="text", text=json.dumps(item_metadata(item))),
        ImageContent(type="image", data=item.image_b64, mimeType="image/png"),
    ]


def _text(text: str) -> list[Content]:
    return [TextContent(type="text", text=text)]


def build_server(catalogue: Catalogue) -> FastMCP:
    """Create the MCP server with the three catalogue tools and a health route."""
    mcp = FastMCP("Product Catalogue MCP Server")

    @mcp.tool(name=SEMANTIC_SEARCH, description=SEMANTIC_SEARCH_DOC)
    async def semantic_search(
        description: str, categories: list[str], num_matches: int
    ) -> list[Content]:
        log.debug("semantic_search tool call: {!r} in {}", description, categories)
        try:
            items = await catalogue.search(description, categories, num_matches)
        except InvalidCategoryError as exc:
            return _text(f"Error: {exc}")
        return [content for item in items for content in _item_content(item)]

    @mcp.tool(name=GET_PRODUCT_CATEGORIES, description=GET_PRODUCT_CATEGORIES_DOC)
    async def get_product_categories() -> list[Content]:
        log.debug("get_product_categories tool call")
        return _text(json.dumps(catalogue.categories))

    @mcp.tool(name=GET_DATAPOINT_BY_INDEX, description=GET_DATAPOINT_BY_INDEX_DOC)
    async def get_datapoint_by_index(index: int) -> list[Content]:
        log.debug("get_datapoint_by_index tool call: {}", index)
        item = await catalogue.get(index)
        if item is None:
            return _text(f"Error: no catalogue item with index {index}.")
        return _item_content(item)

    @mcp.custom_route("/health", methods=["GET"])
    async def health(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "items": len(catalogue)})

    return mcp


async def serve(catalogue: Catalogue, settings: ToolsSettings) -> None:
    """Serve the tools over streamable HTTP until cancelled."""
    # run_async rather than run: the caller is already inside an event loop
    await build_server(catalogue).run_async(
        transport="http", host=settings.mcp_host, port=settings.mcp_port
    )
