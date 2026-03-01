"""Explore various things to see how they work before incorporation.

1. We are going to explore how qwen3-vl can be used to describe images.
2. Figure out how data is stored in the h5 file for fashion-gen, and
"""

import logging
import random

import h5py
import hydra
from langchain_core.messages import HumanMessage
from PIL import Image

from src.utils.common_utils import encode_image

log = logging.getLogger(__name__)


def test_qwen(cfg):
    """See how qwen local model describes an image."""
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

def test_fashion_gen(cfg):
    """See how the fashion-gen dataset is setup, how to navigate it, and what content it has."""
    with h5py.File(cfg.data.fashion_gen_hdf5_path, 'r') as file:
        num_images = file['index'].shape[0]
        idx = random.randint(0, num_images - 1)
        img = Image.fromarray(file['input_image'][idx].astype('uint8'))
        img.show()
        for key in cfg.data.string_attributes:
            value = file[key][idx][0].decode('utf-8')
            log.info(f'{key} {value}')
