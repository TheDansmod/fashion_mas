"""Create Local Vector DB using Nova Model.

this file is after two failed attempts:
- i first tried to create embeddings through batch inference - and that did not
    work out
- then i tried to create a script to run on EC2 directly, but that also did not
    go very well
- finally i am planning on creating the vector db locally and then uploading it
    to EBS later
since this file is not standalone we can use the configs setup by the DI
    framework
but that creates its own complication that i am just going to ignore and we'll
    just make this file be standalone as well
also, while we are simplifying things lets make stuff synchronous as well - no
    constraint on time or RAM, so we are good to go
"""

import json
import random
import logging as log

import boto3
from tqdm import tqdm
from botocore.config import Config
import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# globals
# stuff that is set in setup methods
qdrant_client = None
metadata_lookup = None
embedder = None

# config globals
qdrant_host = "localhost"
qdrant_grpc_port = 6334
prefer_grpc = True
recreate = False
collection_name = "nova_fashion_gen"

image_vectors_name = "images"
text_vectors_name = "texts"
embedding_size = 1024

vectors_on_disk = True
payload_on_disk = True
hnsw_on_disk = True
indexing_threshold = 20_000

metadata_file_path = r"/mnt/windows/Users/lordh/Documents/Svalbard/Data/fashion-gen/fashion_gen_metadata.parquet"
index_key = "index_2"
descriptions_key = "input_description"

data_bucket_name = "frag-fashion-gen-dataset"
account_id = "650539477368"
model_id = "amazon.nova-2-multimodal-embeddings-v1:0"

start_index = 100
end_index = 10_000
fetch_batch_size = 100
num_datapoints = 260_490
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


def get_image_s3_key_from_index(idx):
    """Random docstring."""
    return f"images/{idx // 1000:03d}/{idx}.png"


def setup_qdrant_client():
    """Random docstring."""
    global qdrant_client
    log.info("setting up qdrant client")

    qdrant_client = QdrantClient(
        host=qdrant_host, grpc_port=qdrant_grpc_port, prefer_grpc=prefer_grpc
    )
    log.info("Connection made with QdrantClient")

    if recreate and qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)
        log.info(f"Deleted existing collection '{collection_name}'.")

    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
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


def setup_metadata_lookup():
    """Random docstring."""
    global metadata_lookup

    log.info("setting up metadata lookup")
    table = pq.read_table(metadata_file_path)
    df = table.to_pandas()
    df.set_index(index_key, inplace=True, drop=False)
    metadata_lookup = df.to_dict(orient="index")
    log.info("metadata_lookup setup")
    log.info(f"{list(metadata_lookup[start_index].keys())}")


class Embedder:
    """Random docstring."""

    def __init__(self):
        """Random docstring."""
        config = Config(retries={"total_max_attempts": 5, "mode": "adaptive"})
        self.client = boto3.client("bedrock-runtime", config=config)

    def _get_text_request_body(self, index):
        """Random docstring."""
        description = metadata_lookup[index][descriptions_key]
        request_body = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": embedding_size,
                "text": {"value": description, "truncationMode": "END"},
            },
        }
        return request_body

    def _get_image_request_body(self, index):
        """Random docstring."""
        image_s3_uri = f"s3://{data_bucket_name}/{get_image_s3_key_from_index(index)}"
        request_body = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": embedding_size,
                "image": {
                    "detailLevel": "STANDARD_IMAGE",
                    "format": "png",
                    "source": {
                        "s3Location": {"uri": image_s3_uri, "bucketOwner": account_id}
                    },
                },
            },
        }
        return request_body

    def _get_embedding(self, request_body):
        """Random docstring."""
        try:
            response = self.client.invoke_model(
                body=json.dumps(request_body),
                modelId=model_id,
                accept="application/json",
                contentType="application/json",
            )
        except Exception:
            log.exception(
                "There was an exception while getting embedding from the nova model."
            )
            raise

        resp_body = json.loads(response["body"].read())
        embedding = resp_body["embeddings"][0]["embedding"]
        return embedding

    def get_paired_embedding_batch(self, from_idx, to_idx):
        """Random docstring."""
        results = []
        for index in range(from_idx, to_idx):
            image_embedding = self._get_embedding(self._get_image_request_body(index))
            text_embedding = self._get_embedding(self._get_text_request_body(index))
            results.append((index, image_embedding, text_embedding))
        return results


def populate_vector_db():
    """Random docstring."""
    for from_idx in tqdm(range(start_index, end_index, fetch_batch_size)):
        to_idx = min(from_idx + fetch_batch_size, end_index)
        points = []
        img_descr_pairs = embedder.get_paired_embedding_batch(from_idx, to_idx)
        log.info(f"Embed {from_idx}-{to_idx} of {end_index - start_index} datapoints")
        for abs_idx, img_vec, text_vec in img_descr_pairs:
            assert metadata_lookup[abs_idx][index_key] == abs_idx
            # construct the named vectors
            named_vectors = {image_vectors_name: img_vec, text_vectors_name: text_vec}
            # construct the point struct from the payload and named vectors
            points.append(
                models.PointStruct(
                    id=abs_idx,
                    vector=named_vectors,
                    payload={
                        key: metadata_lookup[abs_idx][key] for key in payload_attributes
                    },
                )
            )
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
        )

def check_collection_count():
    """Connect to Qdrant and report how many points are in the collection.

    Since point IDs are the absolute dataset indices (abs_idx), the count
    directly tells you where to resume:
        start_index = count
        end_index   = start_index + <however many you want next>
    """
    client = QdrantClient(
        host=qdrant_host, grpc_port=qdrant_grpc_port, prefer_grpc=prefer_grpc
    )

    if not client.collection_exists(collection_name):
        log.info(f"Collection '{collection_name}' does not exist yet.")
        return

    count_result = client.count(collection_name, exact=True)
    info = client.get_collection(collection_name)

    print("Printing some random fetch")
    random_id = random.randint(0, count_result.count)
    results = client.retrieve(
        collection_name=collection_name,
        ids=[random_id],
        with_vectors=True,
        with_payload=True,
    )

    if not results:
        log.info(f"No point found at id={random_id} (possible gap). Try again.")
        return

    point = results[0]

    print(f"\n--- Random Point (id={point.id}) ---")

    print("\n[Vectors]")
    for vec_name, vec in point.vector.items():
        preview = vec[:10]
        print(f"  {vec_name} (dim={len(vec)}): {[round(v, 5) for v in preview]} ...")

    print("\n[Payload]")
    for key, value in point.payload.items():
        # truncate long strings (e.g. descriptions)
        str_val = str(value)
        print(f"  {key}: {str_val}")

    log.info(f"--- Collection: '{collection_name}' ---")
    log.info(f"  Points (exact):   {count_result.count:,}")
    log.info(f"  Total datapoints: {num_datapoints}")
    log.info(f"  Progress:         {count_result.count / num_datapoints * 100:.2f}%")
    log.info(f"  Suggested next:   start_index = {count_result.count}")

def validate():
    log.info(f"START INDEX: {start_index}")
    log.info(f"END INDEX: {end_index}")

    assert end_index <= num_datapoints
    assert start_index < end_index
    assert fetch_batch_size <= end_index - start_index
    if recreate:
        cont = input("Recreate true. Enter YES if you want to continue: ")
        if not cont == "YES":
            raise ValueError("did not get permission to continue.")
    else:
        log.info("Recreate = False")
    is_ok = input("Proceed? YES/NO: ")
    if not is_ok == "YES":
        raise ValueError("Permisssion Denied")
    else:
        log.info("Continuing.")

def main():
    """Random docstring."""
    global embedder
    
    validate()
    setup_qdrant_client()
    setup_metadata_lookup()
    embedder = Embedder()
    populate_vector_db()


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        check_collection_count()
    else:
        main()
