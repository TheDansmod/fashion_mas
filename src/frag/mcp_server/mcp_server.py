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

import argparse
import base64
import json
import logging
from io import BytesIO
from typing import Literal

import h5py
import numpy as np
import open_clip
import torch
from fastmcp import FastMCP
from fastmcp.tools import tool
from mcp.types import ImageContent, TextContent
from PIL import Image
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s - %(levelname)s] - %(message)s"
)

log = logging.getLogger(__name__)

mcp = FastMCP("Product Catalogue MCP Server")

class ProductCatalogueMCPServer:
    def __init__(self, connector, embedder):
        self._connector = connector
        self._embedder = embedder

    def _reformat_image_data(self, matches):
        matched_images = []
        for match in matches:
            b64_image = encode_image(match["input_image"][0])
            metadata = {
                "price": match["input_msrpUSD"][0],
                "category": match["input_category"][0],
                "description": match["input_description"][0],
                "id": match["index_2"][0],
                "score": match.get('score', 0),
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
            if category not in self._connector.get_valid_categories():
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
        data = get_fashion_gen_data(from_idx=index, to_idx=index+1)
        return self._reformat_image_data([data])

    @tool
    def get_product_categories(self) -> list[str]:
        """Returns a list of valid product categories."""
        log.info("in get product categories.")
        return [
            "CLUTCHES & POUCHES",
            "POUCHES & DOCUMENT HOLDERS",
            "BOOTS",
            "BACKPACKS",
            "SWEATERS",
            "SWIMWEAR",
            "MONKSTRAPS",
            "JEWELRY",
            "DUFFLE & TOP HANDLE BAGS",
            "JEANS",
            "LACE UPS",
            "SKIRTS",
            "DUFFLE BAGS",
            "TOPS",
            "DRESSES",
            "MESSENGER BAGS & SATCHELS",
            "SOCKS",
            "LOAFERS",
            "ESPADRILLES",
            "UNDERWEAR & LOUNGEWEAR",
            "BAG ACCESSORIES",
            "HATS",
            "SANDALS",
            "JACKETS & COATS",
            "MESSENGER BAGS",
            "GLOVES",
            "TRAVEL BAGS",
            "LINGERIE",
            "SCARVES",
            "KEYCHAINS",
            "BLANKETS",
            "TIES",
            "FLATS",
            "SHORTS",
            "PANTS",
            "SUITS & BLAZERS",
            "TOTE BAGS",
            "HEELS",
            "EYEWEAR",
            "BRIEFCASES",
            "JUMPSUITS",
            "FINE JEWELRY",
            "BELTS & SUSPENDERS",
            "SHIRTS",
            "BOAT SHOES & MOCCASINS",
            "SNEAKERS",
            "POCKET SQUARES & TIE BARS",
            "SHOULDER BAGS",
        ]

class QdrantConnector:
    def __init__(self, url, collection_name):
        self._client = QdrantClient(url=url, prefer_grpc=True)
        log.info("connected to qdrant.")
        # validate collection existence
        if not self._client.collection_exists(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist.")
        self._collection_name = collection_name
        self._category_key = "input_category"
        self._image_vectors_name = "images"
        self._index_key = "index_2"

    def get_valid_categories(self):
        return [
            "CLUTCHES & POUCHES",
            "POUCHES & DOCUMENT HOLDERS",
            "BOOTS",
            "BACKPACKS",
            "SWEATERS",
            "SWIMWEAR",
            "MONKSTRAPS",
            "JEWELRY",
            "DUFFLE & TOP HANDLE BAGS",
            "JEANS",
            "LACE UPS",
            "SKIRTS",
            "DUFFLE BAGS",
            "TOPS",
            "DRESSES",
            "MESSENGER BAGS & SATCHELS",
            "SOCKS",
            "LOAFERS",
            "ESPADRILLES",
            "UNDERWEAR & LOUNGEWEAR",
            "BAG ACCESSORIES",
            "HATS",
            "SANDALS",
            "JACKETS & COATS",
            "MESSENGER BAGS",
            "GLOVES",
            "TRAVEL BAGS",
            "LINGERIE",
            "SCARVES",
            "KEYCHAINS",
            "BLANKETS",
            "TIES",
            "FLATS",
            "SHORTS",
            "PANTS",
            "SUITS & BLAZERS",
            "TOTE BAGS",
            "HEELS",
            "EYEWEAR",
            "BRIEFCASES",
            "JUMPSUITS",
            "FINE JEWELRY",
            "BELTS & SUSPENDERS",
            "SHIRTS",
            "BOAT SHOES & MOCCASINS",
            "SNEAKERS",
            "POCKET SQUARES & TIE BARS",
            "SHOULDER BAGS",
        ]

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


def get_fashion_gen_data(from_idx, to_idx):
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
    data = dict()
    images_key = "input_image"
    prices_key = "input_msrpUSD"
    index_key = "index_2"
    num_datapoints = 260490
    codec = "latin-1"
    hdf5_path = args.datapath
    string_attributes = [
        "input_brand",
        "input_category",
        "input_composition",
        "input_department",
        "input_gender",
        "input_name",
        "input_season",
        "input_subcategory",
        "input_description",
    ]
    if from_idx >= num_datapoints or from_idx >= to_idx:
        return data
    else:
        from_idx = max(0, from_idx)
    if to_idx >= num_datapoints:
        to_idx = num_datapoints
    vec_decode = np.vectorize(pyfunc=lambda x: x.decode(codec))
    with h5py.File(hdf5_path, "r") as file:
        data[images_key] = file[images_key][from_idx:to_idx].astype("uint8")
        data[prices_key] = np.ravel(file[prices_key][from_idx:to_idx]).tolist()
        data[index_key] = file[index_key][from_idx:to_idx].tolist()  # don't need ravel
        for key in string_attributes:
            data[key] = vec_decode(np.ravel(file[key][from_idx:to_idx])).tolist()
    return data


def encode_image(numpy_image):
    img = Image.fromarray(numpy_image)
    buffer = BytesIO()
    img.save(buffer, format="png")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class FashionSigLIPEmbedding:
    """Create and return multi-modal embeddings."""

    def __init__(self):
        """Initialise device, model, image and text processor, and batch size.

        None of the attributes are intended for external use. So all of them start
        with _.
        """
        self._device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(self._device_type)
        self._model, _, self._preprocess_val = open_clip.create_model_and_transforms(
            "hf-hub:Marqo/marqo-fashionSigLIP"
        )
        self._tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
        self._model.to(self._device)
        self._embed_batch_size = 512

    def get_image_embedding_batch(self, images):
        """Generates embeddings for an array of images.

        We iterate through the images and give them to the model to encode in batches.

        Args:
            images (numpy ndarray): Array of images of shape [B, H, W, C].

        Returns:
            results (list[list[float]]): It returns a list of embeddings. Each embedding
                is a python list of floats length
                cfg.data.data_processing.embedding_size.
        """
        results = []
        for i in range(0, images.shape[0], self._embed_batch_size):
            batch = images[i : i + self._embed_batch_size]
            tensor_list = [self._preprocess_val(Image.fromarray(img)) for img in batch]
            batched_image_tensor = torch.stack(tensor_list, dim=0).to(self._device)
            with torch.no_grad(), torch.amp.autocast(device_type=self._device_type):
                image_features = (
                    self._model.encode_image(batched_image_tensor, normalize=True)
                    .cpu()
                    .tolist()
                )
            results.extend(image_features)
        return results

    def get_text_embedding_batch(self, texts):
        """Generates embeddings for a list of texts.

        We iterate through the texts and give them to the model to encode in batches.

        Args:
            texts (list[str]): A list of string texts.

        Returns:
            results (list[list[float]]): It returns a list of embeddings. Each embedding
                is a python list of floats of length
                cfg.data.data_processing.embedding_size.
        """
        results = []
        for i in range(0, len(texts), self._embed_batch_size):
            batch = texts[i : i + self._embed_batch_size]
            batched_texts = self._tokenizer(batch).to(self._device)
            with torch.no_grad(), torch.amp.autocast(device_type=self._device_type):
                text_features = (
                    self._model.encode_text(batched_texts, normalize=True)
                    .cpu()
                    .tolist()
                )
            results.extend(text_features)
        return results

    def get_paired_embedding_batch(self, images, texts):
        """Generates embeddings for batched images and texts and returns them paired.

        If the images and the texts on corresponding indices are related (eg: one is
        the description of the other, we often want the embeddings to also be
        together). That is what this function does. It is mostly a utility function.

        Args:
            images (numpy ndarray): Array of images of shape [B, H, W, C].
                B = len(texts)
            texts (list[str]): A list of string texts. len(texts) = B

        Returns:
            results (list[tuple[list[float], list[float]]]): Returns a list of pairs
                of embeddings. Each emebdding is a list of floats of length given by
                cfg.data.data_processing.embedding_size.
        """
        img_results = self.get_image_embedding_batch(images)
        text_results = self.get_text_embedding_batch(texts)
        return list(zip(img_results, text_results))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default="/mnt/windows/Users/lordh/Documents/Svalbard/Data/fashion-gen/fashiongen_256_256_train.h5", help="Path to the hdf5 file for the fashion gen dataset")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    connector = QdrantConnector(
        url="http://localhost:6333", collection_name="fashion_gen"
    )
    embedder = FashionSigLIPEmbedding()
    server = ProductCatalogueMCPServer(connector=connector, embedder=embedder)
    mcp.add_tool(server.semantic_search)
    mcp.add_tool(server.get_product_categories)
    mcp.add_tool(server.get_datapoint_by_index)
    mcp.run(transport="http", port=9000)
