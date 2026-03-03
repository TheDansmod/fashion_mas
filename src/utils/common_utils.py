"""This file will contain commonly re-used utility functions."""

import base64
import logging

log = logging.getLogger(__name__)


def encode_image(image_path):
    """Encode a local image file to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def validate_hydra_config(cfg):
    """Runs some checks to ensure validity of hydra config."""
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
