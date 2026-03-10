"""Read / extract vectors from vector database."""

import logging

from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.data_manager.vector_db_writer import get_vector_db_client

log = logging.getLogger(__name__)


class VectorDbReader:
    """Reads / extracts vectors from vector database."""

    def __init__(self, cfg):
        """Setup the reader."""
        self._client = get_vector_db_client(cfg)
        self._collection_name = cfg.data.vector_db.collection_name
        self._image_vectors_name = cfg.data.vector_db.image_vectors_name
        self._category_key = cfg.data.fashion_gen.categories_key
        self._index_key = cfg.data.fashion_gen.index_key

    def get_image_matches(self, embedding, num_matches=1, categories=None):
        """Gets num_matches images that best match the embedding.

        We also filter on all the categories mentioned in the passed in list of
        categories. We use an OR filter so it is sufficient for a point to belong
        to any one of the categories in order to be matched.

        Returns:
            ids (list[int]): list of ids of the images from the collection that are
                the best matches to the provided embedding. The length of the list
                is num_matches.
        """
        ids = []
        should_filter = []
        if categories:
            for cat in categories:
                condition = FieldCondition(
                    key=self._category_key, match=MatchValue(value=cat)
                )
                should_filter.append(condition)
            query_filter = Filter(should=should_filter)
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
            item_cat = scored_points.payload[self._category_key]
            ids.append(item_id)
            log.debug(f"id: {item_id}; category: {item_cat}")
        return ids
