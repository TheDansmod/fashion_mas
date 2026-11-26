"""Qdrant access for image similarity search."""

from typing import Any

from loguru import logger as log
from qdrant_client import AsyncQdrantClient, models

from frag.data_manager.dataset_read_write import FashionGenData


class QdrantConnector:
    """Searches the image vectors of one collection, optionally filtered by category."""

    def __init__(  # noqa: PLR0913  plain constructor; Phase 6 slims this down
        self,
        url: str,
        prefer_grpc: bool,  # noqa: FBT001  mirrors the settings field
        collection_name: str,
        category_key: str,
        image_vectors_name: str,
        index_key: str,
        data: FashionGenData,
    ) -> None:
        """Create the async client; call `validate()` before use."""
        self._client = AsyncQdrantClient(url=url, prefer_grpc=prefer_grpc)
        self._collection_name = collection_name
        self._category_key = category_key
        self._image_vectors_name = image_vectors_name
        self._index_key = index_key
        self._data = data

    async def validate(self) -> None:
        """Raise if the collection does not exist."""
        if not await self._client.collection_exists(self._collection_name):
            msg = f"Qdrant collection {self._collection_name!r} does not exist."
            raise ValueError(msg)
        log.debug("connected to qdrant collection {}", self._collection_name)

    async def close(self) -> None:
        """Close the client connection."""
        await self._client.close()

    async def get_image_matches(
        self, embedding: list[float], categories: list[str], num_matches: int
    ) -> list[dict[str, Any]]:
        """Return the closest catalogue items (with image and metadata) and scores."""
        log.debug("getting matching images from qdrant vector db")
        query_filter = None
        if categories:
            # "should" = OR: an item matches if it is in any of the categories
            query_filter = models.Filter(
                should=[
                    models.FieldCondition(
                        key=self._category_key, match=models.MatchValue(value=cat)
                    )
                    for cat in categories
                ]
            )
        query_response = await self._client.query_points(
            collection_name=self._collection_name,
            query=embedding,
            using=self._image_vectors_name,
            query_filter=query_filter,
            limit=num_matches,
        )
        matches = []
        for scored_point in query_response.points:
            img_data = await self._data.get(scored_point.payload[self._index_key])
            if not img_data:
                continue
            img_data["score"] = scored_point.score
            matches.append(img_data)
        return matches
