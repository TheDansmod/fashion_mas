# This file is meant to be run on an EC2 instance and is thus standalone
# the purpose is to connect to a running Qdrant Container, generate embeddings
# of images and text and then insert them into the vector db on an attached
# EBS instance.

import io
import json
import asyncio
import logging as log

import aioboto3
import pyarrow.parquet as pq
from qdrant_client import AsyncQdrantClient, models


log.basicConfig(
    level=log.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# globals
qdrant_host = "localhost"
qdrant_grpc_port = 6334
prefer_grpc = True
recreate = True
collection_name = "fashion_gen"

image_vectors_name = "images"
text_vectors_name = "texts"
embedding_size = 1024

vectors_on_disk = True
payload_on_disk = True
hnsw_on_disk = True
indexing_threshold = 0
final_indexing_threshold = 20_000

qdrant_client = None
data_bucket_name = 'frag-fashion-gen-dataset'
get_image_s3_key_from_index = lambda idx: f"images/{idx // 1000:03d}/{idx}.png"
account_id = "650539477368"
model_id = "amazon.nova-2-multimodal-embeddings-v1:0"
metadata_key = "metadata/fashion_gen_metadata.parquet"
descriptions_key = "input_description"
index_key = "index_2"

metadata_df = None
embedder = None
fetch_batch_size: int = 1024
aws_bedrock_simultaneous_queries = 20


payload_attributes = [
    "s3_keys", "index_2", "input_msrpUSD", "input_brand", "input_category",
    "input_composition", "input_department", "input_gender", "input_name",
    "input_season", "input_subcategory", "input_description",
]

async def setup_qdrant_client():
    global qdrant_client

    qdrant_client = AsyncQdrantClient(host=qdrant_host, grpc_port=qdrant_grpc_port, prefer_grpc=prefer_grpc)

    if recreate and await qdrant_client.collection_exists(collection_name):
        await qdrant_client.delete_collection(collection_name)
        log.info(f"Deleted existing collection '{collection_name}'.")

    if not await qdrant_client.collection_exists(collection_name):
        await qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                image_vectors_name: models.VectorParams(
                    size=embedding_size,
                    distance=models.Distance.COSINE,
                    on_disk=vectors_on_disk,
                ),
                text_vectors_name: models.VectorParams(
                    size=embedding_size,
                    distance=models.Distance.COSINE,
                    on_disk=vectors_on_disk,
                ),
            },
            on_disk_payload=payload_on_disk,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=indexing_threshold,
            ),
            hnsw_config=models.HnswConfigDiff(on_disk=hnsw_on_disk),
        )
        log.info(f"Created new collection '{collection_name}'...")
    else:
        log.info(f"Collection '{collection_name}' already exists. Appending.")

class AsyncEmbedder():
    def __init__(self, session):
        # Configure connection pool for native async boto
        boto_config = aioboto3.core.config.Config(max_pool_connections=aws_bedrock_simultaneous_queries)
        self.client = session.client("bedrock-runtime", config=boto_config)
        self._sem = asyncio.Semaphore(aws_bedrock_simultaneous_queries)
        
    async def __aenter__(self):
        self.client_context = await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client_context.__aexit__(exc_type, exc_val, exc_tb)

    async def _get_embedding(self, request_body):
        async with self._sem:
            try:
                response = await self.client_context.invoke_model(
                    body=json.dumps(request_body),
                    modelId=model_id,
                    accept="application/json",
                    contentType="application/json",
                )
                resp_body = json.loads(await response["body"].read())
                return resp_body["embeddings"][0]["embedding"]
            except Exception as e:
                log.exception("Exception while getting embedding from Bedrock.")
                raise

    async def get_text_embedding(self, description):
        request_body = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": embedding_size,
                "text": {"value": description, "truncationMode": "END"}
            }
        }
        return await self._get_embedding(request_body)

    async def get_image_embedding(self, s3_uri):
        request_body = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": embedding_size,
                "image": {
                    "detailLevel": "STANDARD_IMAGE",
                    "format": "png",
                    "source": {"s3Location": {"uri": s3_uri, "bucketOwner": account_id}}
                }
            }
        }
        return await self._get_embedding(request_body)

    async def get_paired_embedding(self, description, s3_uri):
        return await asyncio.gather(
            self.get_image_embedding(s3_uri),
            self.get_text_embedding(description),
        )

async def populate_vector_db(session):
    global data_bucket_name, metadata_key
    
    # Download parquet to buffer asynchronously
    s3_client = session.client("s3")
    buffer = io.BytesIO()
    async with s3_client as s3:
        await s3.download_fileobj(data_bucket_name, metadata_key, buffer)
    buffer.seek(0)
    
    # Read Parquet file sequentially in chunks to save RAM
    parquet_file = pq.ParquetFile(buffer)
    
    upsert_task = None

    async with AsyncEmbedder(session) as embedder:
        for batch_num, batch in enumerate(parquet_file.iter_batches(batch_size=fetch_batch_size)):
            df_chunk = batch.to_pandas()
            points = []
            
            # Prepare all embedder tasks for the current chunk
            tasks = []
            for _, row in df_chunk.iterrows():
                idx = row[index_key]
                desc = row[descriptions_key]
                s3_uri = f"s3://{data_bucket_name}/{get_image_s3_key_from_index(idx)}"
                tasks.append(embedder.get_paired_embedding(desc, s3_uri))
            
            # Fetch embeddings concurrently for the whole chunk
            results = await asyncio.gather(*tasks)
            
            for row_idx, (img_vec, text_vec) in enumerate(results):
                row = df_chunk.iloc[row_idx]
                points.append(
                    models.PointStruct(
                        id=row[index_key],
                        vector={image_vectors_name: img_vec, text_vectors_name: text_vec},
                        payload={key: row[key] for key in payload_attributes if key in row},
                    )
                )

            log.debug(f"Fetched embeddings for batch {batch_num + 1}")

            # Pipelining: Wait for previous upsert to finish before triggering the next one
            if upsert_task:
                await upsert_task

            # Fire off Qdrant upsert in the background and immediately move to the next embedding batch
            upsert_task = asyncio.create_task(qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
            ))

        # Await the final upsert task
        if upsert_task:
            await upsert_task

async def reset_qdrant_indexing_threshold():
    global qdrant_client, collection_name, final_indexing_threshold

    await qdrant_client.update_collection(
        collection_name=collection_name,
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=final_indexing_threshold,
        ),
    )

    log.info("Waiting for HNSW index to build...")
    while True:
        collection_info = await qdrant_client.get_collection(collection_name)
        if collection_info.optimizer_status == models.OptimizersStatusOneOf.OK:
            log.info("Indexing complete.")
            break
        await asyncio.sleep(5)

async def main():
    global embedder

    await setup_qdrant_client()
    session = aioboto3.Session()
    await populate_vector_db()
    await reset_qdrant_indexing_threshold()
    await qdrant_client.close()

if __name__ == '__main__':
    asyncio.run(main())
