"""The catalogue service used by the tools: search by text, get by id."""

import asyncio
import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from loguru import logger as log
from qdrant_client import AsyncQdrantClient

from frag.catalogue.embedders import Embedder, build_embedder
from frag.catalogue.models import CatalogueItem
from frag.catalogue.store import (
    CATEGORY,
    DESCRIPTION,
    PRICE,
    S3ImageStore,
    build_s3_client,
    load_metadata,
)
from frag.catalogue.vector_db import QdrantConnector
from frag.settings import Settings

# upper bound on one search's results: every result is an image in the LLM's context
# (tokens, memory on a local model) and an S3 GET
MAX_SEARCH_MATCHES = 6


class InvalidCategoryError(ValueError):
    """A requested category is not in the catalogue."""


class Catalogue:
    """Metadata (in memory), images (S3), an embedder and a vector index."""

    def __init__(
        self,
        metadata: dict[int, dict[str, Any]],
        store: S3ImageStore,
        embedder: Embedder,
        vector_db: QdrantConnector,
    ) -> None:
        """Wire the parts together; `open_catalogue` is the usual way to build one."""
        self._metadata = metadata
        self._store = store
        self._embedder = embedder
        self._vector_db = vector_db
        # derived from the metadata, so a catalogue subset never offers empty categories
        self.categories: list[str] = sorted(
            {m[CATEGORY] for m in metadata.values() if isinstance(m.get(CATEGORY), str)}
        )

    def __len__(self) -> int:
        """Number of items with metadata."""
        return len(self._metadata)

    async def _item(self, item_id: int, score: float | None = None) -> CatalogueItem:
        png = await self._store.get_png_bytes(item_id)
        meta = self._metadata[item_id]
        return CatalogueItem(
            id=item_id,
            category=meta[CATEGORY],
            description=meta[DESCRIPTION],
            price=float(meta[PRICE]),
            image_b64=base64.b64encode(png).decode("ascii"),
            score=score,
        )

    async def search(
        self, description: str, categories: list[str], k: int
    ) -> list[CatalogueItem]:
        """Return up to `k` items matching `description`, within `categories`.

        `k` is clamped to 1..MAX_SEARCH_MATCHES.

        Raises:
            InvalidCategoryError: if any category is not in the catalogue.
        """
        invalid = [c for c in categories if c not in self.categories]
        if invalid:
            msg = f"Not valid categories: {', '.join(invalid)}."
            raise InvalidCategoryError(msg)
        k = max(1, min(k, MAX_SEARCH_MATCHES))
        vector = await self._embedder.embed_text(description)
        hits = await self._vector_db.search(vector, categories, k)
        # the index may hold ids outside the loaded metadata (e.g. a subset)
        hits = [(i, s) for i, s in hits if i in self._metadata]
        results = await asyncio.gather(
            *(self._item(i, s) for i, s in hits), return_exceptions=True
        )
        items = []
        for (item_id, _), result in zip(hits, results, strict=True):
            if isinstance(result, Exception):
                log.opt(exception=result).warning(
                    "Dropping search hit {}: image fetch failed.", item_id
                )
            else:
                items.append(result)
        return items

    async def get(self, item_id: int) -> CatalogueItem | None:
        """Return the item, or None if the id is not in the catalogue."""
        if item_id not in self._metadata:
            return None
        return await self._item(item_id)


@asynccontextmanager
async def open_catalogue(settings: Settings) -> AsyncIterator[Catalogue]:
    """Load metadata from S3, build the embedder, and open and validate Qdrant."""
    s3_client = build_s3_client(settings.aws_region)
    qdrant = AsyncQdrantClient(
        url=settings.vector_db.url, prefer_grpc=settings.vector_db.prefer_grpc
    )
    try:
        metadata = await load_metadata(s3_client, settings.catalogue)
        # SigLIP loads a model for a few seconds: keep the event loop free meanwhile
        embedder = await asyncio.to_thread(build_embedder, settings)
        vector_db = QdrantConnector(
            client=qdrant,
            collection=settings.collection,
            vector_name=settings.vector_db.image_vector_name,
            category_key=settings.catalogue.category_field,
            index_key=settings.catalogue.index_field,
        )
        await vector_db.validate(embedder.dim)
        store = S3ImageStore(
            settings.catalogue.s3_bucket,
            settings.catalogue.image_key_template,
            s3_client,
        )
        catalogue = Catalogue(metadata, store, embedder, vector_db)
        log.info(
            "catalogue ready: {} items, {} categories, collection {} ({})",
            len(catalogue),
            len(catalogue.categories),
            settings.collection,
            settings.embedding.backend,
        )
        yield catalogue
    finally:
        await qdrant.close()
        s3_client.close()
