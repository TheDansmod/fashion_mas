"""The catalogue bundle (data, vector DB, embedder, categories) and its lifespan.

Phase 6 of cleanup.md turns this into a service with `search()` / `get()` and moves
it to `frag/catalogue/`.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import boto3
from botocore.config import Config
from loguru import logger as log

from frag.data_manager.dataset_read_write import FashionGenData, load_metadata
from frag.data_manager.embedding import AWSEmbedder
from frag.data_manager.vector_db_read_write import QdrantConnector
from frag.settings import Settings


@dataclass(frozen=True)
class Catalogue:
    """Everything the catalogue tools need."""

    data: FashionGenData
    vector_db: QdrantConnector
    embedder: AWSEmbedder
    # derived from the metadata, so a catalogue subset never offers empty categories
    categories: list[str]


def _build_embedder(settings: Settings) -> AWSEmbedder:
    emb = settings.embedding
    if emb.backend == "nova":
        return AWSEmbedder(
            embedding_size=emb.nova_dim,
            model_id=emb.nova_model_id,
            region_name=settings.aws_region,
            purpose=emb.nova_purpose,
        )
    msg = (
        "The siglip embedder is restored in cleanup.md Phase 6; "
        "use EMBEDDING__BACKEND=nova."
    )
    raise NotImplementedError(msg)


@asynccontextmanager
async def open_catalogue(settings: Settings) -> AsyncIterator[Catalogue]:
    """Load metadata from S3, connect to Qdrant and build the embedder."""
    s3_client = boto3.client(
        "s3",
        region_name=settings.aws_region,
        config=Config(
            max_pool_connections=20, retries={"mode": "adaptive", "max_attempts": 3}
        ),
    )
    try:
        metadata = await load_metadata(s3_client, settings.catalogue)
        data = FashionGenData(s3_client, settings.catalogue, metadata)
        category_field = settings.catalogue.category_field
        categories = sorted(
            {
                item[category_field]
                for item in metadata.values()
                if isinstance(item[category_field], str)
            }
        )
        vector_db = QdrantConnector(
            url=settings.vector_db.url,
            prefer_grpc=settings.vector_db.prefer_grpc,
            collection_name=settings.collection,
            category_key=category_field,
            image_vectors_name=settings.vector_db.image_vector_name,
            index_key=settings.catalogue.index_field,
            data=data,
        )
        try:
            await vector_db.validate()
            catalogue = Catalogue(
                data=data,
                vector_db=vector_db,
                embedder=_build_embedder(settings),
                categories=categories,
            )
            log.info(
                "catalogue ready: {} items, {} categories, collection {}",
                len(metadata),
                len(categories),
                settings.collection,
            )
            yield catalogue
        finally:
            await vector_db.close()
    finally:
        s3_client.close()
