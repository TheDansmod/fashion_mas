from loguru import logger as log
from qdrant_client import AsyncQdrantClient, models

from frag.data_manager.dataset_read_write import get_fashion_gen_data

class QdrantConnector:
    def __init__(
        self,
        url,
        prefer_grpc,
        collection_name,
        category_key,
        image_vectors_name,
        index_key,
        fgen_args,
    ):
        self._client = AsyncQdrantClient(url=url, prefer_grpc=prefer_grpc)
        log.debug("connected to qdrant.")
        self._collection_name = collection_name
        self._category_key = category_key
        self._image_vectors_name = image_vectors_name
        self._index_key = index_key
        self._fgen_args = fgen_args

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
            img_data = await get_fashion_gen_data(item_id, *self._fgen_args)
            img_data["score"] = score
            matches.append(img_data)
        return matches
