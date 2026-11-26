"""An MCP Server for the Vector DB.

1. For now, we are only adding the semantic search tool so that we can figure out how it
works.
2. We are also not doing the inject the function signature thing.
3. We assume the collection already exists
4. Assume we always prefer GRPC
5. self._category_key, self._image_vectors_name, self._index_key
6. all the categories are hard-coded
7. all the string hard-coded values in get_fashion_gen_data
8. Later we can increase the information returned by matched image
"""

import base64
import json
import asyncio

import numpy as np
from fastmcp import FastMCP
from fastmcp.tools import tool
from mcp.types import ImageContent, TextContent
from loguru import logger as log
from dependency_injector.wiring import inject, Provide as PV

from frag.data_manager.dataset_read_write import get_fashion_gen_data
from frag.config.container import Container

cfg = Container.config.provided
mcp = FastMCP("Product Catalogue MCP Server")

class ProductCatalogueMCPServer:
    @inject
    def __init__(
        self,
        connector,
        embedder,
        product_categories: list[str] = PV[cfg.data.fashion_gen.product_categories],
    ):
        self._connector = connector
        self._embedder = embedder
        self._product_categories = product_categories

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
        data = await get_fashion_gen_data(index)
        return self._reformat_image_data([data])

    @tool
    def get_product_categories(self) -> list[str]:
        """Returns a list of valid product categories."""
        log.info("get_product_categories tool call made")
        return self._product_categories

@inject
async def main(
    transport: str = PV[cfg.orchestration.mcp.host_transport_method],
    port: int = PV[cfg.orchestration.mcp.port],
    connector = PV[Container.qdrant_connector.provided],
    embedder = PV[Container.multimodal_embedder.provided],
):
    server = ProductCatalogueMCPServer(connector=connector, embedder=embedder)
    mcp.add_tool(server.semantic_search)
    mcp.add_tool(server.get_product_categories)
    mcp.add_tool(server.get_datapoint_by_index)
    # since this main function is being invoked from an async context already, we can't just do mcp.run which internally calls asyncio.run (assuming that we started from a sync context) - which then throws an error. so we can simply fix that by asynchronously invoking the mcp runner through run_async
    await mcp.run_async(transport=transport, port=port)
