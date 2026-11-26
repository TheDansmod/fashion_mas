"""Qdrant image-vector search; returns catalogue ids and scores only."""

from loguru import logger as log
from qdrant_client import AsyncQdrantClient, models


class QdrantConnector:
    """Searches the image vectors of one collection, optionally filtered by category."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection: str,
        vector_name: str,
        category_key: str,
        index_key: str,
    ) -> None:
        """Wrap an open client; call `validate()` before searching."""
        self._client = client
        self._collection = collection
        self._vector_name = vector_name
        self._category_key = category_key
        self._index_key = index_key

    async def validate(self, expected_dim: int) -> None:
        """Raise if the collection is missing or its vectors don't match the embedder.

        This catches an embedding backend paired with the wrong collection at startup
        rather than at the first search.
        """
        if not await self._client.collection_exists(self._collection):
            msg = f"Qdrant collection {self._collection!r} does not exist."
            raise ValueError(msg)
        info = await self._client.get_collection(self._collection)
        vectors = info.config.params.vectors
        params = vectors.get(self._vector_name) if isinstance(vectors, dict) else None
        if params is None:
            msg = (
                f"Qdrant collection {self._collection!r} has no vector named "
                f"{self._vector_name!r}."
            )
            raise ValueError(msg)
        if params.size != expected_dim:
            msg = (
                f"Qdrant collection {self._collection!r} stores {params.size}-d "
                f"{self._vector_name!r} vectors, but the embedder produces "
                f"{expected_dim}-d vectors. Check EMBEDDING__BACKEND."
            )
            raise ValueError(msg)
        log.debug("connected to qdrant collection {}", self._collection)

    async def search(
        self, vector: list[float], categories: list[str], k: int
    ) -> list[tuple[int, float]]:
        """Return `(catalogue id, score)` of the `k` nearest items, best first."""
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
        response = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            using=self._vector_name,
            query_filter=query_filter,
            limit=k,
        )
        return [(int(p.payload[self._index_key]), p.score) for p in response.points]
