"""Catalogue tools served by the MCP server (TOOLS__MODE=mcp)."""

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from frag.catalogue.models import CatalogueItem
from frag.tools.base import (
    GET_DATAPOINT_BY_INDEX,
    LLM_TOOL_NAMES,
    blocks_to_item,
    normalise_mcp_blocks,
)


class MCPToolsClient:
    """Fetches the tool list from the server once, then reuses it."""

    def __init__(self, url: str) -> None:
        """Create the client; nothing is contacted until the first call."""
        self._client = MultiServerMCPClient(
            {"product_catalogue": {"transport": "streamable_http", "url": url}}
        )
        self._tools: dict[str, BaseTool] | None = None

    async def _get_tools(self) -> dict[str, BaseTool]:
        if self._tools is None:
            self._tools = {t.name: t for t in await self._client.get_tools()}
        return self._tools

    async def get_llm_tools(self) -> list[BaseTool]:
        """The LLM tools, with output normalised to the in-process format."""
        tools = await self._get_tools()
        return [normalise_mcp_blocks(tools[name]) for name in LLM_TOOL_NAMES]

    async def get_item(self, item_id: int) -> CatalogueItem | None:
        """Look up one item through the server's `get_datapoint_by_index` tool."""
        tool = (await self._get_tools())[GET_DATAPOINT_BY_INDEX]
        return blocks_to_item(await tool.ainvoke({"index": item_id}))
