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

import atexit
import signal
import base64
import json
import threading
from collections import OrderedDict
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


mcp = FastMCP("Product Catalogue MCP Server")

_BUCKET = "frag-fashion-gen-dataset"
_S3_KEY = "fashion_gen.h5"
_NUM_DATAPOINTS = 260490
_CODEC = "latin-1"
_IMAGES_KEY = "input_image"
_PRICES_KEY = "input_msrpUSD"
_INDEX_KEY = "index_2"
_STRING_ATTRIBUTES = [
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

_vec_decode = np.vectorize(pyfunc=lambda x: x.decode(_CODEC))


# h5py is NOT thread-safe, so all access to _h5_file must be protected by _h5_lock.
_h5_file: h5py.File | None = None
_h5_s3_fh = None           # raw s3fs file object — must stay alive while h5 is open
_h5_lock = threading.Lock()


def _get_h5_file() -> h5py.File:
    """Return (or lazily initialise) the long-lived HDF5 file handle."""
    global _h5_file, _h5_s3_fh
    if _h5_file is None:
        with _h5_lock:
            if _h5_file is None:  # double-checked locking — safe for lazy init
                log.info("Opening HDF5 file from S3 (one-time initialisation)…")
                # Tuning: the default s3fs block_size is 5 MB. Our access pattern
                # is non-sequential random access to individual rows, so 5 MB blocks
                # means we download ~5 MB to use ~200 KB. 1 MB is a better trade-off
                # that reduces wasted bandwidth by ~5× without hurting latency.
                fs = s3fs.S3FileSystem(anon=False, default_block_size=1 * 1024 * 1024)
                _h5_s3_fh = fs.open(f"s3://{_BUCKET}/{_S3_KEY}", "rb")
                _h5_file = h5py.File(_h5_s3_fh, "r")
                log.info("HDF5 file open; handle will be reused for the process lifetime.")
    return _h5_file

# ── Graceful shutdown ──────────────────────────────────────────────────────────

def _close_h5_resources() -> None:
    """Close the HDF5 file handle and the underlying S3 file object in order.

    h5py must be closed before the s3fs file object it wraps, otherwise h5py
    will try to flush/read from an already-closed file descriptor.
    """
    global _h5_file, _h5_s3_fh
    if _h5_file is not None:
        try:
            _h5_file.close()
            log.info("HDF5 file closed.")
        except Exception as exc:
            log.warning("Error closing HDF5 file: {}", exc)
        finally:
            _h5_file = None
    if _h5_s3_fh is not None:
        try:
            _h5_s3_fh.close()
            log.info("S3 file handle closed.")
        except Exception as exc:
            log.warning("Error closing S3 file handle: {}", exc)
        finally:
            _h5_s3_fh = None


def _handle_sigterm(signum, frame) -> None:
    """Translate SIGTERM into a clean shutdown (atexit handlers will then run)."""
    log.info("Received SIGTERM — shutting down.")
    raise SystemExit(0)


# Covers: normal exit, sys.exit(), and KeyboardInterrupt (Ctrl-C).
atexit.register(_close_h5_resources)

# Covers: docker stop, systemd stop, kill <pid> — anything that sends SIGTERM.
# SIGINT (Ctrl-C) is already handled via KeyboardInterrupt → atexit above.
signal.signal(signal.SIGTERM, _handle_sigterm)

# ── Item-level LRU cache ───────────────────────────────────────────────────────
# Caches recently fetched items so that repeated lookups for the same row index
# (common in conversational MCP sessions) never hit S3 at all.
# Images are large, so cap at 256 items (~50–100 MB RAM depending on image size).
class _LRUCache:
    """Thread-safe LRU dict cache."""

    def __init__(self, maxsize: int = 256) -> None:
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: int) -> dict | None:
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def set(self, key: int, value: dict) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)


_item_cache = _LRUCache(maxsize=256)


# ── Core data-access functions ─────────────────────────────────────────────────

def get_fashion_gen_data_batch(indices: list[int]) -> list[dict]:
    """Fetch multiple datapoints from the HDF5 file in a single pass.

    This is the primary performance upgrade over the original design. The original
    get_image_matches loop called get_fashion_gen_data(item_id, item_id+1) once per
    Qdrant result — meaning N separate S3 connections and N HDF5 file opens for N
    matches. This function collects all row indices upfront, resolves the cache,
    and performs one batched HDF5 read for every uncached index, reducing N round-
    trips to S3 down to 1.

    Each returned dict mirrors the original get_fashion_gen_data shape exactly:
    every value is a 1-element list/array so all existing callers work unchanged.

    Args:
        indices: HDF5 row indices to fetch (order is preserved in output).

    Returns:
        List of data dicts, one per input index, in the same order.
        Out-of-range indices produce an empty dict {}.
    """
    if not indices:
        return []

    results: list[dict] = [{} for _ in indices]

    # ── Pass 1: resolve cache ──────────────────────────────────────────────────
    uncached_positions: list[int] = []   # position in `indices` list
    uncached_h5_indices: list[int] = []  # corresponding HDF5 row index

    for pos, idx in enumerate(indices):
        if not (0 <= idx < _NUM_DATAPOINTS):
            continue
        cached = _item_cache.get(idx)
        if cached is not None:
            results[pos] = cached
        else:
            uncached_positions.append(pos)
            uncached_h5_indices.append(idx)

    if not uncached_h5_indices:
        return results   # everything was cached — zero S3 requests needed

    h5 = _get_h5_file()

    # ── Pass 2: batched HDF5 read ──────────────────────────────────────────────
    # h5py fancy indexing requires indices in strictly increasing order.
    # We sort, read, then map results back to the original position order.
    sort_order = sorted(range(len(uncached_h5_indices)), key=lambda i: uncached_h5_indices[i])
    sorted_h5_idx = [uncached_h5_indices[i] for i in sort_order]

    with _h5_lock:
        raw_images  = h5[_IMAGES_KEY][sorted_h5_idx].astype("uint8")   # [N, H, W, C]
        raw_prices  = np.ravel(h5[_PRICES_KEY][sorted_h5_idx])         # [N]
        raw_idx_col = h5[_INDEX_KEY][sorted_h5_idx]                    # [N, ...]
        string_cols: dict[str, list[str]] = {
            key: _vec_decode(np.ravel(h5[key][sorted_h5_idx])).tolist()
            for key in _STRING_ATTRIBUTES
        }

    # Reconstruct per-item dicts and populate cache
    # sort_order[rank] gives us the position in uncached_positions/uncached_h5_indices
    # that corresponds to the rank-th sorted read result.
    reverse_sort = [0] * len(sort_order)
    for rank, orig_sort_pos in enumerate(sort_order):
        reverse_sort[orig_sort_pos] = rank

    for orig_sort_pos, orig_pos in enumerate(uncached_positions):
        rank = reverse_sort[orig_sort_pos]
        h5_idx = uncached_h5_indices[orig_sort_pos]
        entry: dict = {
            _IMAGES_KEY: raw_images[rank : rank + 1],   # shape [1, H, W, C] — matches original
            _PRICES_KEY: [float(raw_prices[rank])],
            _INDEX_KEY:  [raw_idx_col[rank]],
            **{key: [string_cols[key][rank]] for key in _STRING_ATTRIBUTES},
        }
        _item_cache.set(h5_idx, entry)
        results[orig_pos] = entry

    return results


def get_fashion_gen_data(from_idx: int, to_idx: int) -> dict:
    """Backward-compatible single-item wrapper around get_fashion_gen_data_batch.

    The original signature is preserved so callers like get_datapoint_by_index
    require no changes. Internally this now uses the singleton HDF5 handle and
    the LRU cache, so repeated calls for the same index are served from memory.
    """
    if from_idx >= _NUM_DATAPOINTS or from_idx >= to_idx:
        return {}
    from_idx = max(0, from_idx)
    batch = get_fashion_gen_data_batch([from_idx])
    return batch[0] if batch else {}


# ── Helpers ────────────────────────────────────────────────────────────────────

def encode_image(numpy_image) -> str:
    img = Image.fromarray(numpy_image)
    buffer = BytesIO()
    img.save(buffer, format="png")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ── MCP Server ─────────────────────────────────────────────────────────────────

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
        data = get_fashion_gen_data(from_idx=index, to_idx=index + 1)
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

        # BATCHING FIX:
        # Original code called get_fashion_gen_data(item_id, item_id+1) inside a
        # for-loop — one S3 connection + HDF5 file open per match. For 10 matches
        # that was 10× the overhead. Now we collect all IDs and scores first, then
        # fetch all data in a single batched HDF5 read (see get_fashion_gen_data_batch).
        item_ids = [p.payload[self._index_key] for p in query_response.points]
        scores   = [p.score for p in query_response.points]

        batch = get_fashion_gen_data_batch(item_ids)

        matches = []
        for img_data, score in zip(batch, scores):
            # Shallow-copy before mutation so cached dicts are never modified in-place.
            img_data = {**img_data, "score": score}
            matches.append(img_data)
        return matches


class FashionSigLIPEmbedding:
    """Create and return multi-modal embeddings."""

    def __init__(self):
        """Initialise device, model, image and text processor, and batch size."""
        self._device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(self._device_type)
        self._model, _, self._preprocess_val = open_clip.create_model_and_transforms(
            "hf-hub:Marqo/marqo-fashionSigLIP"
        )
        self._tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
        self._model.to(self._device)
        self._embed_batch_size = 512

    def get_image_embedding_batch(self, images):
        """Generates embeddings for an array of images."""
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
        """Generates embeddings for a list of texts."""
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
        """Generates embeddings for batched images and texts and returns them paired."""
        img_results = self.get_image_embedding_batch(images)
        text_results = self.get_text_embedding_batch(texts)
        return list(zip(img_results, text_results))


if __name__ == "__main__":
    connector = QdrantConnector(
        url="http://localhost:6333", collection_name="fashion_gen"
    )
    embedder = FashionSigLIPEmbedding()
    server = ProductCatalogueMCPServer(connector=connector, embedder=embedder)
    mcp.add_tool(server.semantic_search)
    mcp.add_tool(server.get_product_categories)
    mcp.add_tool(server.get_datapoint_by_index)
    mcp.run(transport="http", port=9000)
