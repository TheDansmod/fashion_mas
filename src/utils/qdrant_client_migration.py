import logging

from qdrant_client import QdrantClient

log = logging.getLogger(__name__)

def migrate_local_to_docker(cfg):
    src_client = QdrantClient(path=cfg.data.vector_db.vector_store_path)
    dst_client = QdrantClient(path=cfg.data.vector_db.vector_store_network_path, prefer_grpc=cfg.data.vector_db.prefer_grpc)

    src_client.migrate(
            dest_client=dst_client,
            collection_names=[cfg.data.vector_db.collection_name],
            recreate_on_collision=True,
            batch_size=cfg.data.vector_db.migration_batch_size
    )

