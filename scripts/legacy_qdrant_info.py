# this file was required since I am resuming working on this after quite some time and I would like to remember what the size of the qdrant db is - how many images it contains - since I don't think it contains too many images

import asyncio
from loguru import logger as log
from qdrant_client import AsyncQdrantClient, models

async def main():
    client = AsyncQdrantClient(url="http://localhost:6333", prefer_grpc=True)
    log.debug("connected to qdrant.")

    collection = await client.get_collection("nova_fashion_gen")

    print("Points count:           ", collection.points_count)
    print("Indexed vectors count:  ", collection.indexed_vectors_count)
    print("Segments count:         ", collection.segments_count)
    print("Status:                 ", collection.status)

if __name__ == '__main__':
    asyncio.run(main())
