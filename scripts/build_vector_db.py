"""Embed catalogue items with Amazon Nova and upsert them into a Qdrant collection.

A reproducible record of how the `nova_fashion_gen` collection (10,000 items; 1024-d
`images` and `texts` vectors, cosine) was built, over several runs; it doesn't need to
be run again (cleanup.md decision 1.12). Each item gets an
image embedding (the PNG on S3, read by Bedrock directly) and a text embedding of its
description, both with `embeddingPurpose=GENERIC_INDEX`. The point id is the item's
catalogue index, and the payload holds its metadata fields.

The SigLIP collection (`fashion_gen`) was built by a different script,
`vector_db_writer.py`, deleted in commit 2583639; recover it with
`git show 2583639^:<path>` (find the path with `git show --stat 2583639`).

Every Bedrock call is billed (cleanup.md §0.2C): the script prints how many requests it
will make and asks for a typed YES first. Run with
`uv run python scripts/build_vector_db.py --help`.
"""

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import boto3
import pyarrow.parquet as pq
from botocore.config import Config
from loguru import logger as log
from qdrant_client import QdrantClient, models

from frag.catalogue.store import build_s3_client
from frag.settings import Settings

from _common import confirm, start

VECTOR_SIZE = 1024
IMAGE_VECTOR = "images"
TEXT_VECTOR = "texts"
# metadata stored with each point; the app filters on the category and index fields
PAYLOAD_FIELDS = [
    "s3_keys",
    "index_2",
    "input_msrpUSD",
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
# rough size of a catalogue description, for the token estimate
TOKENS_PER_CHAR = 0.25


def parse_args(settings: Settings) -> argparse.Namespace:
    """Command-line arguments; defaults come from the settings."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--collection", default=settings.vector_db.collections["nova"])
    parser.add_argument("--qdrant-url", default=settings.vector_db.url)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument(
        "--ids-file", type=Path, help="text file with one item index per line"
    )
    which.add_argument(
        "--range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="item indexes START..END-1",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="delete and recreate the collection first (asks for YES)",
    )
    parser.add_argument(
        "--price-per-image", type=float, help="current $ per image embedding"
    )
    parser.add_argument(
        "--price-per-1k-tokens", type=float, help="current $ per 1k text tokens"
    )
    parser.add_argument("--yes", action="store_true", help="skip the cost question")
    return parser.parse_args()


def load_payloads(settings: Settings) -> dict[int, dict[str, Any]]:
    """Every item's payload fields from the metadata Parquet on S3, by index.

    `frag.catalogue.store.load_metadata` reads only the fields the app needs, under
    generic names; the collection's payload keeps all of them under their own names.
    """
    catalogue = settings.catalogue
    buffer = io.BytesIO()
    build_s3_client(settings.aws_region).download_fileobj(
        catalogue.s3_bucket, catalogue.metadata_key, buffer
    )
    buffer.seek(0)
    rows = pq.read_table(buffer, columns=PAYLOAD_FIELDS).to_pylist()
    return {row[catalogue.index_field]: row for row in rows}


class NovaEmbedder:
    """Image and text embeddings from Amazon Nova Multimodal Embeddings."""

    def __init__(self, settings: Settings) -> None:
        """Create the Bedrock client and find the account that owns the bucket."""
        self._settings = settings
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
            config=Config(retries={"total_max_attempts": 5, "mode": "adaptive"}),
        )
        # Bedrock reads the images from S3 itself and needs the bucket owner's account
        self._account_id = boto3.client(
            "sts", region_name=settings.aws_region
        ).get_caller_identity()["Account"]

    def _embed(self, params: dict[str, Any]) -> list[float]:
        body = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": VECTOR_SIZE,
                **params,
            },
        }
        response = self._client.invoke_model(
            body=json.dumps(body),
            modelId=self._settings.embedding.nova_model_id,
            accept="application/json",
            contentType="application/json",
        )
        return json.loads(response["body"].read())["embeddings"][0]["embedding"]

    def image(self, idx: int) -> list[float]:
        """Embedding of item `idx`'s PNG on S3."""
        catalogue = self._settings.catalogue
        key = catalogue.image_key_template.format(shard=idx // 1000, idx=idx)
        return self._embed(
            {
                "image": {
                    "detailLevel": "STANDARD_IMAGE",
                    "format": "png",
                    "source": {
                        "s3Location": {
                            "uri": f"s3://{catalogue.s3_bucket}/{key}",
                            "bucketOwner": self._account_id,
                        }
                    },
                }
            }
        )

    def text(self, description: str) -> list[float]:
        """Embedding of an item description."""
        return self._embed({"text": {"value": description, "truncationMode": "END"}})


def ensure_collection(client: QdrantClient, name: str, *, recreate: bool) -> None:
    """Create the collection if needed (after deleting it, with `recreate`)."""
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
        log.info("Deleted collection {}.", name)
    if client.collection_exists(name):
        log.info("Collection {} exists; adding to it.", name)
        return
    vector = models.VectorParams(
        size=VECTOR_SIZE, distance=models.Distance.COSINE, on_disk=True
    )
    client.create_collection(
        collection_name=name,
        vectors_config={IMAGE_VECTOR: vector, TEXT_VECTOR: vector},
        on_disk_payload=True,
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20_000),
        hnsw_config=models.HnswConfigDiff(on_disk=True),
    )
    log.info("Created collection {}.", name)


def cost_question(
    ids: list[int], payloads: dict[int, dict[str, Any]], args: argparse.Namespace
) -> str:
    """What the run will call on Bedrock, with a $ estimate if prices were given."""
    tokens = int(
        sum(len(payloads[i]["input_description"]) for i in ids) * TOKENS_PER_CHAR
    )
    text = (
        f"Embed {len(ids)} items with Nova: {len(ids)} image and {len(ids)} text "
        f"requests (about {tokens:,} text tokens) into '{args.collection}' at "
        f"{args.qdrant_url}."
    )
    if args.price_per_image is not None and args.price_per_1k_tokens is not None:
        cost = (
            len(ids) * args.price_per_image + tokens / 1000 * args.price_per_1k_tokens
        )
        return f"{text} Estimated Bedrock cost: ${cost:.2f}."
    return (
        f"{text} Check the Nova Multimodal Embeddings prices on the Bedrock pricing "
        "page (or pass --price-per-image and --price-per-1k-tokens for an estimate)."
    )


def main() -> None:
    """Embed the selected items and upsert them in batches."""
    settings = start("build-vector-db")
    args = parse_args(settings)
    if args.ids_file:
        ids = sorted({int(x) for x in args.ids_file.read_text().split()})
    else:
        ids = list(range(*args.range))
    payloads = load_payloads(settings)
    missing = [i for i in ids if i not in payloads]
    if missing:
        sys.exit(f"{len(missing)} ids are not in the catalogue, e.g. {missing[:5]}.")
    if args.recreate and not confirm(
        f"--recreate deletes the collection '{args.collection}' and all its points."
    ):
        sys.exit("Not confirmed; nothing changed.")
    if not confirm(cost_question(ids, payloads, args), assume_yes=args.yes):
        sys.exit("Not confirmed; nothing changed.")

    client = QdrantClient(url=args.qdrant_url)
    ensure_collection(client, args.collection, recreate=args.recreate)
    embedder = NovaEmbedder(settings)
    for first in range(0, len(ids), args.batch_size):
        batch = ids[first : first + args.batch_size]
        points = [
            models.PointStruct(
                id=idx,
                vector={
                    IMAGE_VECTOR: embedder.image(idx),
                    TEXT_VECTOR: embedder.text(payloads[idx]["input_description"]),
                },
                payload=payloads[idx],
            )
            for idx in batch
        ]
        client.upsert(collection_name=args.collection, points=points)
        log.info("Upserted {}/{} items.", first + len(batch), len(ids))


if __name__ == "__main__":
    main()
