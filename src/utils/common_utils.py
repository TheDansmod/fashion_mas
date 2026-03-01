"""This file will contain commonly re-used utility functions."""

import base64
import logging

log = logging.getLogger(__name__)


def encode_image(image_path):
    """Encode a local image file to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
