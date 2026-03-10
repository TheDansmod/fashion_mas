"""This file will contain commonly re-used utility functions."""

import base64
import logging
from io import BytesIO

from langchain_core.messages import HumanMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from PIL import Image

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
    if cfg.data.vector_db.recreate and cfg.data.data_processing.insert_start_index > 0:
        raise ValueError(
            "When the insert_start_index > 0, we should have recreate be False."
        )
    if (
        cfg.data.data_processing.insert_start_index
        >= cfg.data.data_processing.insert_stop_index
    ):
        raise ValueError("The insert_start_index should be < insert_stop_index.")
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
        num_images = file["index"].shape[0]
        for i in range(num_images):
            idx = random.randint(0, num_images - 1)
            img = Image.fromarray(file["input_image"][idx].astype("uint8"))
            img.save(cfg.misc.random_image_save_path.format(i))
            log.debug(f"Saved image {i}")


def fetch_fashion_gen_images(cfg, image_ids=None):
    """Fetches some randomly chosen images from fashion-gen.

    The primary use of this is to use those images as input for the agentic system.
    The images are saved in the path given by cfg.misc.random_image_save_path with the
    index of the image inserted into the name.
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
