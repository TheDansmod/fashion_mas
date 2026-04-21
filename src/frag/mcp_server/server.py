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
import logging
from io import BytesIO
from typing import Literal

import h5py
import s3fs
import numpy as np
import open_clip
import torch
from fastmcp import FastMCP
from fastmcp.tools import tool
from mcp.types import ImageContent, TextContent
from PIL import Image
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from loguru import logger as log
from dependency_injector.wiring import inject, Provide as PV

from frag.utils.common_utils import encode_image
from frag.config.container import Container
from frag.mcp_server.embedding import FashionSigLIPEmbedding

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
            b64_image = encode_image(match["input_image"][0])
            metadata = {
                "price": match["input_msrpUSD"][0],
                "category": match["input_category"][0],
                "description": match["input_description"][0],
                "id": match["index_2"][0],
                "score": match.get("score", 0),
            }
            text_content = TextContent(type="text", text=json.dumps(metadata))
            image_content = ImageContent(
                type="image", data=b64_image, mimeType="image/jpeg"
            )
        return [text_content, image_content]

    @tool
    def semantic_search(
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
        log.info("in semantic search.")
        for category in categories:
            if category not in self._product_categories:
                return {"error": f"{category} is not a valid category."}
        embedding = self._embedder.get_text_embedding_batch([description])[0]
        matches = self._connector.get_image_matches(
            embedding, categories=categories, num_matches=num_matches
        )
        log.info("returning some matches.")
        return self._reformat_image_data(matches)

    @tool
    def get_datapoint_by_index(self, index: int):
        """Get a datapoint, including image and metadata, using index in db."""
        data = get_fashion_gen_data(from_idx=index, to_idx=index + 1)
        return self._reformat_image_data([data])

    @tool
    def get_product_categories(self) -> list[str]:
        """Returns a list of valid product categories."""
        log.info("in get product categories.")
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
        self._client = QdrantClient(url=url, prefer_grpc=prefer_grpc)
        log.info("connected to qdrant.")
        # validate collection existence
        if not self._client.collection_exists(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist.")
        self._collection_name = collection_name
        self._category_key = category_key
        self._image_vectors_name = image_vectors_name
        self._index_key = index_key

    def get_image_matches(self, embedding, categories, num_matches):
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
        query_response = self._client.query_points(
            collection_name=self._collection_name,
            query=embedding,
            using=self._image_vectors_name,
            query_filter=query_filter,
            limit=num_matches,
        )
        for scored_points in query_response.points:
            item_id = scored_points.payload[self._index_key]
            score = scored_points.score
            img_data = get_fashion_gen_data(from_idx=item_id, to_idx=item_id + 1)
            img_data["score"] = score
            matches.append(img_data)
        return matches

@inject
def get_fashion_gen_data(
    from_idx, to_idx,
    bucket: str = PV[cfg.data.aws_fashion_gen.s3_bucket_name],
    s3_key: str = PV[cfg.data.aws_fashion_gen.dataset_object_name],
    images_key: str = PV[cfg.data.fashion_gen.images_key],
    prices_key: str = PV[cfg.data.fashion_gen.prices_key],
    index_key: str = PV[cfg.data.fashion_gen.index_key],
    num_datapoints: int = PV[cfg.data.fashion_gen.num_datapoints],
    codec: str = PV[cfg.data.fashion_gen.string_codec],
    string_attributes: list[str] = PV[cfg.data.fashion_gen.string_attributes],
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
    fs = s3fs.S3FileSystem(anon=False)

    # to be returned
    data = dict()

    if from_idx >= num_datapoints or from_idx >= to_idx:
        return data
    else:
        from_idx = max(0, from_idx)
    if to_idx >= num_datapoints:
        to_idx = num_datapoints
    vec_decode = np.vectorize(pyfunc=lambda x: x.decode(codec))
    with fs.open(f"s3://{bucket}/{s3_key}", "rb") as s3_file:
        with h5py.File(s3_file, "r") as file:
            data[images_key] = file[images_key][from_idx:to_idx].astype("uint8")
            data[prices_key] = np.ravel(file[prices_key][from_idx:to_idx]).tolist()
            data[index_key] = file[index_key][from_idx:to_idx].tolist()  # don't need ravel
            for key in string_attributes:
                data[key] = vec_decode(np.ravel(file[key][from_idx:to_idx])).tolist()
    return data

@inject
async def main(
    transport: str = PV[cfg.orchestration.mcp.host_transport_method],
    port: int = PV[cfg.orchestration.mcp.port],
):
    connector = QdrantConnector()
    embedder = FashionSigLIPEmbedding()
    server = ProductCatalogueMCPServer(connector=connector, embedder=embedder)
    mcp.add_tool(server.semantic_search)
    mcp.add_tool(server.get_product_categories)
    mcp.add_tool(server.get_datapoint_by_index)
    # since this main function is being invoked from an async context already, we can't just do mcp.run which internally calls asyncio.run (assuming that we started from a sync context) - which then throws an error. so we can simply fix that by asynchronously invoking the mcp runner through run_async
    await mcp.run_async(transport=transport, port=port)
