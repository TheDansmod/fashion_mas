"""MCP server exposing the product catalogue as LLM tools (streamable HTTP)."""

import asyncio
import json

from fastmcp import FastMCP
from fastmcp.tools import tool
from loguru import logger as log
from mcp.types import ImageContent, TextContent

from frag.data_manager.catalogue import Catalogue
from frag.settings import ToolsSettings


class ProductCatalogueMCPServer:
    def __init__(self, catalogue: Catalogue) -> None:
        """Wrap the catalogue; the methods below are registered as MCP tools."""
        self._catalogue = catalogue
        self._connector = catalogue.vector_db
        self._embedder = catalogue.embedder
        self._product_categories = catalogue.categories

    def _reformat_image_data(self, matches):
        matched_images = []
        for match in matches:
            metadata = {
                "price": match["price"],
                "category": match["category"],
                "description": match["description"],
                "id": match["id"],
                "score": match.get("score", 0),
            }
            text_content = TextContent(type="text", text=json.dumps(metadata))
            image_content = ImageContent(
                type="image", data=match["image"], mimeType="image/png"
            )
            matched_images.append(text_content)
            matched_images.append(image_content)
        return matched_images

    @tool
    async def semantic_search(
        self, description: str, categories: list[str], num_matches: int
    ):
        """Get num_matches images and their metadata that match the description and categories.

        Given some text description of an item of clothing or an accessory, this
        function returns `num_matches` images along with their metadata (which
        includes id, price, category, description, and score) from within the images
        that belong to the listed categories.

        Args:
            description (str): The description to which the returned images are matched.
            categories (list[str]): The matched images will belong to one of these
                listed valid categories.
            num_matches (int): The number of matched images that should be returned.

        Returns:
            Matched Images and their metadata. The metadata is a dictionary with the
            keys of `price`, `category`, `description`, `id`, and `score`. The score
            tells how good of a match (to the input text) the returned item is.
        """
        log.debug("semantic_search tool call made")
        for category in categories:
            if category not in self._product_categories:
                return {"error": f"{category} is not a valid category."}
        embedding = await asyncio.to_thread(self._embedder.get_text_embedding_batch, [description])
        embedding = embedding[0]
        matches = await self._connector.get_image_matches(
            embedding, categories=categories, num_matches=num_matches
        )
        log.info("returning some matches.")
        return self._reformat_image_data(matches)

    @tool
    async def get_datapoint_by_index(self, index: int):
        """Get a datapoint, including image and metadata, using index in db."""
        log.debug("get_datapoint_by_index tool call made")
        data = await self._catalogue.data.get(index)
        return self._reformat_image_data([data])

    @tool
    def get_product_categories(self) -> list[str]:
        """Returns a list of valid product categories."""
        log.info("get_product_categories tool call made")
        return self._product_categories

async def main(catalogue: Catalogue, settings: ToolsSettings) -> None:
    """Register the tools and serve them over streamable HTTP until cancelled."""
    mcp = FastMCP("Product Catalogue MCP Server")
    server = ProductCatalogueMCPServer(catalogue)
    mcp.add_tool(server.semantic_search)
    mcp.add_tool(server.get_product_categories)
    mcp.add_tool(server.get_datapoint_by_index)
    # run_async rather than run: we are already inside an event loop
    await mcp.run_async(
        transport="http", host=settings.mcp_host, port=settings.mcp_port
    )
