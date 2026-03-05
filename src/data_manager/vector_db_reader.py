"""Read / extract vectors from vector database."""

import logging

from src.data_manager.vector_db_writer import get_vector_db_client

log = logging.getLogger(__name__)

class VectorDbReader:
    """Reads / extracts vectors from vector database."""
    def __init__(self, cfg):
        self._client = get_vector_db_client(cfg)
        self._collection_name = cfg.data.vector_db.collection_name
        self._image_vectors_name = cfg.data.vector_db.image_vectors_name

    def get_image_matches(self, embedding, num_matches = 1):
        """Gets num_matches images that best match the embedding.

        Returns:
            ids (list[int]): list of ids of the images from the collection that are
                the best matches to the provided embedding. The length of the list
                is num_matches.
        """
        ids = []
        query_response = client.query_points(collection_name=self._collection_name, query=embedding, using=self._image_vectors_name, limit=num_matches)
        for scored_points in query_response.points:
            ids.append(scored_points.id)
        return ids
