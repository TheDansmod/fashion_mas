"""Explore various things to see how they work before incorporation.

1. We are going to explore how qwen3-vl can be used to describe images.
2. Figure out how data is stored in the h5 file for fashion-gen, and
"""

import logging

import hydra
from langchain_core.messages import HumanMessage

from src.utils.common_utils import encode_image

log = logging.getLogger(__name__)


def test_qwen(cfg):
    """See how it describes an image."""
    provider = hydra.utils.instantiate(cfg.models.vlm_agent)
    model = provider(
        model=cfg.models.vlm_agent.name, temperature=cfg.models.vlm_agent.temp
    )
    image_data = encode_image(cfg.misc.test_image_path)
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{image_data}",
                },
                {
                    "type": "text",
                    "text": "Describe what you see in the image in great detail",
                },
            ]
        )
    ]
    result = model.invoke(messages)
    log.info(type(result))
    log.info(result.content)
