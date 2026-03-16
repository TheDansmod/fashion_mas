"""This file will contain commonly re-used utility functions."""

import asyncio
import base64
import csv
import functools
import logging
from datetime import datetime
from io import BytesIO

import hydra
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables.graph import MermaidDrawMethod
from PIL import Image
from qdrant_client import QdrantClient, models

log = logging.getLogger(__name__)


def encode_image(image_path=None, numpy_image=None):
    """Encode an image to base64 from file path or numpy ndarray."""
    if (image_path is None) == (numpy_image is None):
        raise ValueError("Exactly 1 of image_path or numpy_image must be provided.")
    if image_path:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    if numpy_image is not None:
        img = Image.fromarray(numpy_image)
        buffer = BytesIO()
        img.save(buffer, format="png")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def validate_hydra_config(cfg):
    """Runs some checks to ensure validity of hydra config."""
    log.debug(f"Recreating the vector db: {cfg.data.vector_db.recreate}")
    log.debug(
        "Resuming from previous checkpoint: "
        f"{cfg.rag_pipeline.persistence.resume_from_checkpoint}"
    )
    log.debug(f"Running model: {cfg.models.vlm_agent.name}")
    if cfg.data.vector_db.recreate:
        confirmation = input(
            "Please enter `YES` if you want to re-create the vector db: "
        )
        if confirmation != "YES":
            raise ValueError("Cannot recreate vector db without confirmation.")
    if cfg.rag_pipeline.persistence.resume_from_checkpoint:
        confirmation = input(
            "Please enter `YES` if you wish to resume from previous checkpoint: "
        )
        if confirmation != "YES":
            raise ValueError("Cannot resume from checkpoint without confirmation.")
    if (
        cfg.data.data_processing.embedding_batch_size
        > cfg.data.data_processing.data_fetch_batch_size
    ):
        raise ValueError("The embedding_batch_size should be <= data_fetch_batch_size.")


def fetch_random_fashion_gen_images(cfg, num_images=3):
    """Fetches some randomly chosen images from fashion-gen.

    The primary use of this is to use those images as input for the agentic system.
    The images are saved in the path given by cfg.misc.random_image_save_path with the
    index of the image inserted into the name.
    """
    import random

    import h5py
    from PIL import Image

    with h5py.File(cfg.data.fashion_gen.hdf5_path, "r") as file:
        num_datapoints = file["index"].shape[0]
        for i in range(num_images):
            idx = random.randint(0, num_datapoints - 1)
            img = Image.fromarray(file["input_image"][idx].astype("uint8"))
            img.save(cfg.misc.random_image_save_path.format(idx))
            log.debug(f"Saved image {i}")


def fetch_fashion_gen_images(cfg, image_ids=None):
    """Fetches some randomly chosen images from fashion-gen.

    The primary use of this is to use those images as input for the agentic system.
    The images are saved in the path given by cfg.misc.random_image_save_path with the
    index of the image inserted into the name.

    Another big use of this is for debugging, since it lets me figure out why the
    recommendation was made in the first place, and diagnose if there is any error.
    """
    import h5py
    from PIL import Image

    if not image_ids:
        raise ValueError("image_ids should be a list of indices")
    with h5py.File(cfg.data.fashion_gen.hdf5_path, "r") as file:
        for idx in image_ids:
            img = Image.fromarray(file["input_image"][idx].astype("uint8"))
            description = file["input_description"][idx][0].decode("latin-1")
            category = file["input_category"][idx][0].decode("latin-1")
            img.save(cfg.misc.random_image_save_path.format(idx))
            log.debug(
                f"Saved image {idx}\nDescription: {description}\nCategory: {category}"
            )


def get_image_prompt_message(image_path=None, text_prompt=None, numpy_image=None):
    """Get langgraph compatible prompt containing an image and some text."""
    image_data = encode_image(image_path, numpy_image)
    message = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{image_data}",
                },
                {
                    "type": "text",
                    "text": text_prompt,
                },
            ]
        )
    ]
    return message

def get_multi_image_prompt_message(image_paths, text_prompt):
    """Get langgraph compatible prompt containing an image and some text."""
    content = []
    for image_path in image_paths:
        image_data = encode_image(image_path)
        content.append({
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{image_data}",
        })
    content.append({
            "type": "text",
            "text": text_prompt,
    })
    message = [HumanMessage(content=content)]
    return message


def draw_langraph_topology(app, path):
    r"""Given a langgraph app, draw the topology of the graph and save it to path."""
    png_bytes = app.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
    )
    with open(path, "wb") as f:
        f.write(png_bytes)


def get_categories_from_string(cfg, search_string):
    """Gets which categories might be mentioned in the search string."""
    categories = [cat.lower() for cat in cfg.data.fashion_gen.product_categories]
    search_string = search_string.lower()
    matched_categories = []
    for cat in categories:
        if cat in search_string:
            matched_categories.append(cat.upper())
    return matched_categories


def get_qdrant_points_by_id(cfg, ids=None):
    """Logs payload information for Qdrant points using provided IDs."""
    if len(ids) < 1:
        raise ValueError("ids should be a list of ids of length at least 1")
    collection_name = cfg.data.vector_db.collection_name
    client = QdrantClient(
        url=cfg.data.vector_db.vector_store_network_path,
        prefer_grpc=cfg.data.vector_db.prefer_grpc,
    )
    # returned values are list of Records, it has an attribute called payload
    points = client.retrieve(
        collection_name=collection_name, ids=ids, with_payload=True, with_vectors=False
    )
    for point in points:
        log.debug(point.payload["input_category"])
        log.debug(point.payload["input_description"])
        log.debug(point.payload.keys())


def migrate_local_to_docker(cfg):
    """Moves qdrant data from local setup to a docker setup."""
    log.debug("Starting migration process.")
    src_client = QdrantClient(
        url=cfg.data.vector_db.vector_store_network_path,
        prefer_grpc=cfg.data.vector_db.prefer_grpc,
    )
    log.debug("Loaded source client.")
    dst_client = QdrantClient(
        path=cfg.data.vector_db.vector_store_network_path,
        prefer_grpc=cfg.data.vector_db.prefer_grpc,
    )
    log.debug("Loaded destination client. Starting migration")

    src_client.migrate(
        dest_client=dst_client,
        collection_names=[cfg.data.vector_db.collection_name],
        recreate_on_collision=True,
        batch_size=cfg.data.data_processing.migration_batch_size,
    )


def batch_update_vector_db(cfg):
    """Update the index in the vector db.

    I have previously made the mistake of using the index key from the hdf5 dataset to
    index the qdrant database entries under the assumption of one-to-one equivalence
    between the sequential index and the value of the index key for any given element
    of the hdf5 dataset. I have later determined that this equivalence does not exist
    with the index key, but actually with the index_2 key. So, now I will be replacing
    the `"index": wrong-val` element in the qdrant payload with `"index_2": right-val`.

    Since we are using a scroll filter that restricts the updates only to those points
    which still have the wrong key, we can safely re-execute the code when there is a
    failure.
    """
    import h5py

    # get the mapping dictionary between old and new values
    log.debug("Getting old to new mappings.")
    num_datapoints = cfg.data.fashion_gen.num_datapoints
    index_val_to_index_2_val = dict()
    with h5py.File(cfg.data.fashion_gen.hdf5_path, "r") as file:
        for idx in range(num_datapoints):
            index = file["index"][idx][0].item()
            index_2 = file["index_2"][idx].item()  # don't need [0]
            index_val_to_index_2_val[index] = index_2
    log.debug("Obtained the mapping from old to new index values.")
    # get the qdrant client
    qdrant_url = cfg.data.vector_db.vector_store_network_path
    collection_name = cfg.data.vector_db.collection_name
    client = QdrantClient(url=qdrant_url, prefer_grpc=cfg.data.vector_db.prefer_grpc)
    # get the filter - only points which have the wrong key (index)
    scroll_filter = models.Filter(
        must_not=[models.IsEmptyCondition(is_empty=models.PayloadField(key="index"))]
    )
    # perform the batch updates sequentially
    offset = None
    batch_size = cfg.data.data_processing.payload_update_batch_size
    max_num_iter = (num_datapoints // batch_size) + 2  # 2 for safety
    log.debug("Performing batch updates.")
    for iter_num in range(max_num_iter):
        records, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        operations = []
        for record in records:
            new_value = index_val_to_index_2_val[record.payload.get("index")]
            # insertion - set payload only does update not overwrite
            operations.append(
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(
                        payload={"index_2": new_value}, points=[record.id]
                    )
                )
            )
            # deletion
            operations.append(
                models.DeletePayloadOperation(
                    delete_payload=models.DeletePayload(
                        keys=["index"], points=[record.id]
                    )
                )
            )
        # perform the update
        # wait=False ensures that the commands are sent to the container long before
        # it tries to perform the actual updates
        if operations:
            client.batch_update_points(
                collection_name=collection_name,
                update_operations=operations,
                wait=False,
            )
        if offset is None:
            break
        log.debug(f"Done iter {iter_num + 1} of at most {max_num_iter} iterations.")


def update_token_use(cfg, usage_metadata):
    """Updates the token usage tracking csv file with the data from the callback."""
    csv_path = hydra.utils.to_absolute_path(cfg.tracking.token_usage_tracker_path)
    log.info(
        f"Saving token use data for {len(usage_metadata)} models. "
        "Should be invoked just once every full run."
    )
    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        for model_name, metadata in usage_metadata.items():
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    model_name,
                    metadata["input_tokens"],
                    metadata["output_tokens"],
                    metadata["total_tokens"],
                ]
            )


def track_token_use(func):
    """Decorator to track token usage for LLM calls - useful for Mistral."""
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def wrapper(cfg, *args, **kwargs):
            callback = UsageMetadataCallbackHandler()
            callback_config = {"callbacks": [callback]}
            kwargs["callback_config"] = callback_config
            try:
                result = await func(cfg, *args, **kwargs)
            finally:
                update_token_use(cfg, callback.usage_metadata)
            return result

        return wrapper
    else:

        @functools.wraps(func)
        def wrapper(cfg, *args, **kwargs):
            callback = UsageMetadataCallbackHandler()
            callback_config = {"callbacks": [callback]}
            kwargs["callback_config"] = callback_config
            try:
                result = func(cfg, *args, **kwargs)
            finally:
                update_token_use(cfg, callback.usage_metadata)
            return result

        return wrapper


def get_rate_limiter(cfg):
    """Sets up a rate limiter for the LLM agent."""
    rps = cfg.models.rate_limiter.requests_per_second
    check_int = cfg.models.rate_limiter.check_every_n_seconds
    bucket_sz = cfg.models.rate_limiter.max_bucket_size
    if cfg.models.vlm_agent.use_rate_limiter:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=rps,
            check_every_n_seconds=check_int,
            max_bucket_size=bucket_sz,
        )
    else:
        rate_limiter = None
    return rate_limiter

def get_llm_model(cfg):
    """Creates and returns an LLM model for use with appropriate rate limits."""
    provider = hydra.utils.instantiate(cfg.models.vlm_agent)
    model = provider(
        model=cfg.models.vlm_agent.name,
        temperature=cfg.models.vlm_agent.temp,
        rate_limiter=get_rate_limiter(cfg),
    )
    return model

