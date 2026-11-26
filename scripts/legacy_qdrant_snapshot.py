# this file will create the qdrant snapshot of the nova embedding vectors
# the purpose is to migrate this snapshot to AWS EBS instance and restore the collection there
import asyncio
from loguru import logger as log
from qdrant_client import AsyncQdrantClient, models

async def create_snapshot():
    client = AsyncQdrantClient(url="http://localhost:6333", prefer_grpc=True, timeout=300)
    log.info("connected to qdrant.")

    collection = await client.get_collection("nova_fashion_gen")

    log.info("Points count:          {} ", collection.points_count)
    log.info("Indexed vectors count: {} ", collection.indexed_vectors_count)
    log.info("Segments count:        {} ", collection.segments_count)
    log.info("Status:                {} ", collection.status)

    # don't wait until the snapshot creation completes and explicitly poll to check if it is completed since that ensures that gRPC does not trigger a timeout error (which it does by default after 5 secs)
    snapshot_description = await client.create_snapshot(collection_name="nova_fashion_gen", wait=False)
    log.info("Snapshot creation initiated: {}", snapshot_description)

    # poll to check if it is complete
    while True:
        snapshots = await client.list_snapshots(collection_name="nova_fashion_gen")
        if snapshots:
            log.info("Snapshot ready: {}", snapshots[-1].name)
            break
        await asyncio.sleep(10)

async def recover_snapshot():
    client = AsyncQdrantClient(url="http://localhost:6333", prefer_grpc=True)
    log.info("connected to qdrant")

    result = await client.recover_snapshot(
        collection_name="nova_fashion_gen",
        location="file:///qdrant/snapshots/my_collection/my_collection-2024-10-10.snapshot",
        priority=models.SnapshotPriority.SNAPSHOT,
    )


if __name__ == '__main__':
    asyncio.run(create_snapshot())
