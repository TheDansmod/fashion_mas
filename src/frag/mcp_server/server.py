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

import io
import base64
import json
import asyncio

import numpy as np
import pyarrow.parquet as pq
from fastmcp import FastMCP
from fastmcp.tools import tool
from mcp.types import ImageContent, TextContent
from qdrant_client import AsyncQdrantClient, models
from loguru import logger as log
from dependency_injector.wiring import inject, Provide as PV

from frag.utils.common_utils import encode_image
from frag.config.container import Container
from frag.mcp_server.embedding import FashionSigLIPEmbedding

cfg = Container.config.provided
mcp = FastMCP("Product Catalogue MCP Server")
metadata_lookup = None

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
                type="image", data=match["image"], mimeType="image/jpeg"
            )
        return [text_content, image_content]

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

class QdrantConnector:
    @inject
    def __init__(
        self,
        url: str = PV[cfg.data.vector_db.vector_store_network_path],
        prefer_grpc: bool = PV[cfg.data.vector_db.prefer_grpc],
        collection_name: str = PV[cfg.data.vector_db.collection_name],
        category_key: str = PV[cfg.data.fashion_gen.categories_key],
        image_vectors_name: str = PV[cfg.data.vector_db.image_vectors_name],
        index_key: str = PV[cfg.data.fashion_gen.index_key],
    ):
        self._client = AsyncQdrantClient(url=url, prefer_grpc=prefer_grpc)
        log.debug("connected to qdrant.")
        self._collection_name = collection_name
        self._category_key = category_key
        self._image_vectors_name = image_vectors_name
        self._index_key = index_key

    async def validate(self):
        # validate collection existence
        if not await self._client.collection_exists(self._collection_name):
            raise ValueError(f"Collection {self._collection_name} does not exist.")

    async def get_image_matches(self, embedding, categories, num_matches):
        log.debug("getting matching images from qdrant vector db")
        matches = []
        should_filter = []
        if categories:
            for cat in categories:
                condition = models.FieldCondition(
                    key=self._category_key, match=models.MatchValue(value=cat)
                )
                should_filter.append(condition)
            query_filter = models.Filter(should=should_filter)
        else:
            query_filter = None
        query_response = await self._client.query_points(
            collection_name=self._collection_name,
            query=embedding,
            using=self._image_vectors_name,
            query_filter=query_filter,
            limit=num_matches,
        )
        for scored_points in query_response.points:
            item_id = scored_points.payload[self._index_key]
            score = scored_points.score
            img_data = await get_fashion_gen_data(item_id)
            img_data["score"] = score
            matches.append(img_data)
        return matches

@inject
async def get_fashion_gen_data(
    fetch_index,
    num_datapoints: int = PV[cfg.data.fashion_gen.num_datapoints],
    prices_key: str = PV[cfg.data.fashion_gen.prices_key],
    categories_key: str = PV[cfg.data.fashion_gen.categories_key],
    descriptions_key: str = PV[cfg.data.fashion_gen.descriptions_key],
    s3_client = PV[Container.s3_client.provided],
    bucket_name: str = PV[cfg.data.aws_fashion_gen.s3_bucket_name],
):
    """Get data from the fashion-gen dataset in dictionary format.

    We will be extracting all the attributes in the dataset in a dictionary. What
    data to fetch is determined by the from_idx (included) and to_idx (excluded)
    values. If there is no data within the provided bounds then we return empty
    dictionary.

    Returns:
        data (dict): This dictionary contains the datapoints from the requested
            indices from the fashion-gen dataset. Each key in the dictionary
            corresponds to the name of a dataset in the fashion-gen hdf5 file.
            The value of each key in the dictionary varies depending on requirement.
            If we are sending back images, they are send back as numpy ndarrays. If
            we are sending back string values, they are lists of strings. If we are
            sending back floats, they are lists of floats.
    """
    global metadata_lookup
    data = dict()
    if not 0 <= fetch_index < num_datapoints:
        return data
    image_key = f"images/{fetch_index // 1000:03d}/{fetch_index}.png"
    response = s3_client.get_object(Bucket=bucket_name, Key=image_key)
    data["image"] = base64.b64encode(response["Body"].read()).decode("utf-8")
    data["price"] = metadata_lookup[fetch_index][prices_key]
    data["category"] = metadata_lookup[fetch_index][categories_key]
    data["description"] = metadata_lookup[fetch_index][descriptions_key]
    data["id"] = fetch_index
    return data

@inject
async def setup_metadata_lookup(
    s3_client = PV[Container.s3_client.provided],
    bucket_name: str = PV[cfg.data.aws_fashion_gen.s3_bucket_name],
    metadata_key: str = PV[cfg.data.aws_fashion_gen.fashion_gen_metadata_s3_key],
    index_key: str = PV[cfg.data.fashion_gen.index_key],
):
    global metadata_lookup
    buffer = io.BytesIO()
    await asyncio.to_thread(s3_client.download_fileobj, bucket_name, metadata_key, buffer)
    buffer.seek(0)
    # read with pyarrow
    table = pq.read_table(buffer)
    df = table.to_pandas()
    df.set_index(index_key, inplace=True)
    metadata_lookup = df.to_dict(orient='index')

@inject
async def main(
    transport: str = PV[cfg.orchestration.mcp.host_transport_method],
    port: int = PV[cfg.orchestration.mcp.port],
):
    connector = QdrantConnector()
    await connector.validate()
    embedder = FashionSigLIPEmbedding()
    await setup_metadata_lookup()
    server = ProductCatalogueMCPServer(connector=connector, embedder=embedder)
    mcp.add_tool(server.semantic_search)
    mcp.add_tool(server.get_product_categories)
    mcp.add_tool(server.get_datapoint_by_index)
    # since this main function is being invoked from an async context already, we can't just do mcp.run which internally calls asyncio.run (assuming that we started from a sync context) - which then throws an error. so we can simply fix that by asynchronously invoking the mcp runner through run_async
    await mcp.run_async(transport=transport, port=port)
