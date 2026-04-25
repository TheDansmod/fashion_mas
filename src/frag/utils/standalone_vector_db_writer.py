# This file is meant to be run on an EC2 instance and is thus standalone
# the purpose is to connect to a running Qdrant Container, generate embeddings
# of images and text and then insert them into the vector db on an attached
# EBS instance.

import io
import json
import asyncio
import logging as log

import boto3
from botocore.config import Config
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

# this is the dictionary containing all the metadata for the fashion gen images.
# it can be accessed by doing metadata_df.at[index, key] if you want to lookup some key like "input_description"
metadata_df = None

# this is the instance of the class which generates (using amazon aws nova) the embeddings and returns them
embedder = None

# this is the total number of datapoints present in the fashion-gen dataset
num_datapoints = 260_490

# these are the attributes from the metadata that we want to capture in the vector db metadata
payload_attributes = [
    "s3_keys",
    "index_2",
    "input_msrpUSD",
    "input_brand",
    "input_category",
    "input_composition",
    "input_department",
    "input_gender",
    "input_name",
    "input_season",
    "input_subcategory",
    "input_description",
]

# this is the column in metadata_df that holds the index of the image
index_key = "index_2"

# this is the number of datapoints we fetch at one time from the fashion-gen dataset. Best for it to be some integer multiple of the batch size. Not so large that it does not fit in RAM. Without writing image to payload, the process is really fast.
fetch_batch_size: int = 1024

# this is the value to which the semaphore controlling how many simultaneous queries can be made to aws bedrock, is set
aws_bedrock_simultaneous_queries = 20

async def setup_qdrant_client():
    """Create the vector db client,and collection, and returns the client.

    Depending on the setup the existing collection might be deleted and a new
    collection created. Or if the collection already exists and the recreate flag is
    not enabled, then the existing collection is fetched.
    """
    global vector_db_url, prefer_grpc, recreate, collection_name, image_vectors_name, text_vectors_name, embedding_size, vectors_on_disk, payload_on_disk, hnsw_on_disk, indexing_threshold, qdrant_client

    qdrant_client = AsyncQdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True)

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

async def setup_metadata_df():
    global data_bucket_name, metadata_key, metadata_df

    s3_client = boto3.client("s3")
    log.debug("setting up metadata lookup")
    buffer = io.BytesIO()
    await asyncio.to_thread(s3_client.download_fileobj, data_bucket_name, metadata_key, buffer)
    buffer.seek(0)
    # read with pyarrow
    table = pq.read_table(buffer)
    metadata_df = table.to_pandas()

class Embedder():
    def __init__(self):
        boto_config = Config(max_pool_connections=aws_bedrock_simultaneous_queries)
        self.client = boto3.client("bedrock-runtime", config=boto_config)
        # we divide by two since we are making two queries within the semaphore
        self._sem = asyncio.Semaphore(aws_bedrock_simultaneous_queries // 2)

    def _get_text_request_body(self, index):
        global embedding_size, metadata_df, descriptions_key

        description = metadata_df.at[index, descriptions_key]
        request_body = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": embedding_size,
                "text": {
                    "value": description,
                    "truncationMode": "END"
                }
            }
        }
        return request_body

    def _get_image_request_body(self, index):
        global data_bucket_name, get_image_s3_key_from_index, account_id, embedding_size

        image_s3_uri  = f"s3://{data_bucket_name}/{get_image_s3_key_from_index(index)}"
        request_body = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": embedding_size,
                "image": {
                    "detailLevel": "STANDARD_IMAGE",
                    "format": "png",
                    "source": {
                        "s3Location": {
                            "uri": image_s3_uri,
                            "bucketOwner": account_id
                        }
                    }
                }
            }
        }
        return request_body

    def _get_embedding(self, request_body):
        global model_id

        try:
            response = self.client.invoke_model(
                body=json.dumps(request_body),
                modelId=model_id,
                accept="application/json",
                contentType="application/json",
            )
        except Exception as e:
            log.exception("There was an exception while getting embedding of an image on s3.")
            raise

        resp_body = json.loads(response["body"].read())
        embedding = resp_body["embeddings"][0]["embedding"]
        return embedding

    def _get_text_embedding(self, index):
        return self._get_embedding(self._get_text_request_body(index))

    def _get_image_embedding(self, index):
        return self._get_embedding(self._get_image_request_body(index))

    async def get_text_embedding(self, index):
        return await asyncio.to_thread(self._get_text_embedding, index)

    async def get_image_embedding(self, index):
        return await asyncio.to_thread(self._get_image_embedding, index)

    async def _get_single_paired_embedding(self, index):
        async with self._sem:
            image_vector, text_vector = await asyncio.gather(
                self._get_image_embedding(index),
                self._get_text_embedding(index),
            )
        return image_vector, text_vector

    async def get_paired_embedding_batch(self, from_idx, to_idx):
        global aws_bedrock_simultaneous_queries

        return await asyncio.gather(*[self._get_single_paired_embedding(i) for i in range(from_idx, to_idx)])


async def populate_vector_db():
    global image_vectors_name, text_vectors_name, embedder, num_datapoints, metadata_df, fetch_batch_size, payload_attributes, qdrant_client, collection_name

    for from_idx in range(0, num_datapoints, fetch_batch_size):
        to_idx = min(from_idx + fetch_batch_size, num_datapoints)
        points = []
        img_descr_pairs = await embedder.get_paired_embedding_batch(from_idx, to_idx)
        log.debug(f"Got embedding {from_idx} to {to_idx} out of {num_datapoints} datapoints.")
        for local_idx, (img_vec, text_vec) in enumerate(img_descr_pairs):
            abs_idx = from_idx + local_idx
            # construct the named vectors
            named_vectors = {image_vectors_name: img_vec, text_vectors_name: text_vec}
            # construct the point struct from the payload and named vectors
            points.append(
                models.PointStruct(
                    id=metadata_df.at[abs_idx, index_key],
                    vector=named_vectors,
                    payload={key: metadata_df.at[abs_idx, key] for key in payload_attributes},
                )
            )
        await qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
            wait=False,
        )

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
    await setup_metadata_df()
    embedder = Embedder()
    await populate_vector_db()
    await reset_qdrant_indexing_threshold()

if __name__ == '__main__':
    asyncio.run(main())
