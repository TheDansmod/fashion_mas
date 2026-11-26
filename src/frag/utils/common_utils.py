"""This file will contain commonly re-used utility functions."""

import base64
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
from langchain_core.messages import HumanMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from PIL import Image


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


def get_image_prompt_message(image_path=None, text_prompt=None, numpy_image=None):
    """Get langgraph compatible prompt containing an image and some text."""
    image_data = encode_image(image_path, numpy_image)
    if image_path is None:
        img_format = "png"
    else:
        img_format = "png" if image_path.lower().endswith(".png") else "jpeg"
    message = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{img_format};base64,{image_data}",
                    }
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
        img_format = "png" if image_path.lower().endswith(".png") else "jpeg"
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{img_format};base64,{image_data}",
                }
            }
        )
    content.append(
        {
            "type": "text",
            "text": text_prompt,
        }
    )
    message = [HumanMessage(content=content)]
    return message


def get_multi_image_multi_prompt_message(prompts):
    """Gets a langgraph compatible prompt containing a sequence of images and text.

    The input should be a list of tuples in the order they should be present in
    the message. The first element of the tuple should be either `text` or `image`,
    and the second element, the actual text or image path.
    """
    content = []
    for key, value in prompts:
        if key == "text":
            content.append(
                {
                    "type": "text",
                    "text": value,
                }
            )
        elif key == "image":
            image_data = encode_image(value)
            img_format = "png" if value.lower().endswith(".png") else "jpeg"
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{img_format};base64,{image_data}",
                    }
                }
            )
        else:
            raise ValueError("Key should be either `text` or `image` only.")
    return [HumanMessage(content=content)]


def draw_langraph_topology(app, path):
    r"""Given a langgraph app, draw the topology of the graph and save it to path."""
    png_bytes = app.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
    )
    with open(path, "wb") as f:
        f.write(png_bytes)


def save_numpy_image_to_folder(folder_path: str, image_array: np.ndarray) -> Path:
    """Saves numpy image to temporary folder, returns path."""
    directory = Path(folder_path)
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4()}.png"
    file_path = directory / filename
    image = Image.fromarray(image_array)
    image.save(file_path)
    return file_path


def save_image_b64(folder_path: str, image_b64: str) -> str:
    """Write a base64-encoded PNG to a uniquely named file in `folder_path`."""
    directory = Path(folder_path)
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / f"{uuid.uuid4()}.png"
    file_path.write_bytes(base64.b64decode(image_b64))
    return str(file_path)

