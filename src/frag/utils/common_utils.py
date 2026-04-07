"""This file will contain commonly re-used utility functions."""

from pathlib import Path
import numpy as np
import uuid
import asyncio
import base64
import csv
import functools
from datetime import datetime
from io import BytesIO

from langchain_core.tools import StructuredTool
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from PIL import Image
from qdrant_client import QdrantClient, models
from loguru import logger as log
from dependency_injector.wiring import inject, Provide as PV

from frag.config.container import Container

cfg = Container.config.provided


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
        content.append(
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image_data}",
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
            content.append({
                "type": "text",
                "text": value,
            })
        elif key == "image":
            image_data = encode_image(value)
            content.append({
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image_data}",
            })
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


@inject
def get_qdrant_points_by_id(
    ids=None,
    collection_name: str = PV[cfg.data.vector_db.collection_name],
    url: str = PV[cfg.data.vector_db.vector_store_network_path],
    prefer_grpc: bool = PV[cfg.data.vector_db.prefer_grpc],
):
    """Logs payload information for Qdrant points using provided IDs."""
    if len(ids) < 1:
        raise ValueError("ids should be a list of ids of length at least 1")
    client = QdrantClient(url=url, prefer_grpc=prefer_grpc)
    # returned values are list of Records, it has an attribute called payload
    points = client.retrieve(
        collection_name=collection_name, ids=ids, with_payload=True, with_vectors=False
    )
    for point in points:
        log.debug(point.payload["input_category"])
        log.debug(point.payload["input_description"])
        log.debug(point.payload.keys())


@inject
def update_token_use(
    usage_metadata,
    tracker_path = PV[cfg.tracking.token_usage_tracker_path],
):
    """Updates the token usage tracking csv file with the data from the callback."""
    log.info(
        f"Saving token use data for {len(usage_metadata)} models. "
        "Should be invoked just once every full run."
    )
    with open(tracker_path, "a", newline="") as csv_file:
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
        async def wrapper(*args, **kwargs):
            callback = UsageMetadataCallbackHandler()
            callback_config = {"callbacks": [callback]}
            kwargs["callback_config"] = callback_config
            result = None
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                log.exception("Exception caught inside track_token_use function.")
            finally:
                update_token_use(callback.usage_metadata)
            return result
        return wrapper
    else:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            callback = UsageMetadataCallbackHandler()
            callback_config = {"callbacks": [callback]}
            kwargs["callback_config"] = callback_config
            result = None
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                log.exception("Exception caught inside track_token_use function.")
            finally:
                update_token_use(callback.usage_metadata)
            return result
        return wrapper


def get_tool_with_name(tools, search_name):
    """Given a list of mcp tools, returns the tool with the search name, or errors."""
    tool = None
    for t in tools:
        if t.name == search_name:
            tool = t
            break
    if not tool:
        raise ValueError(f"DB tool not found: {search_name}")
    return tool


def save_numpy_image_to_folder(folder_path: str, image_array: np.ndarray) -> Path:
    """Saves numpy image to temporary folder, returns path."""
    directory = Path(folder_path)
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4()}.png"
    file_path = directory / filename
    image = Image.fromarray(image_array)
    image.save(file_path)
    return file_path

def save_image_url_to_folder(folder_path: str, image_url: str) -> Path:
    """Saves image_url to temporary folder, returns path."""
    directory = Path(folder_path)
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4()}.png"
    file_path = directory / filename
    if "," in image_url:
        base64_string = image_url.split(",")[1]
        image_data = base64.b64decode(base64_string)
    else:
        image_data = base64.b64decode(image_url)
    with open(file_path, 'wb') as file:
        file.write(image_data)
    return str(file_path)

def make_mistral_compatible(tool):
    """Wraps an MCP tool to ensure it returns a plain string."""

    def sanitize_response(response):
        result = []
        if isinstance(response, list):
            for block in response:
                if isinstance(block, dict):
                    block_type = block["type"]
                    if block_type == "text":
                        result.append({"type": "text", "text": block["text"]})
                    elif block_type == "image":
                        result.append(
                            {
                                "type": "image_url",
                                "image_url": f"data:{block['mime_type']};base64,{block['base64']}",
                            }
                        )
                    else:
                        raise ValueError("unexpected type")
                else:
                    # we default to converting the whole thing to string if element of
                    # the list is not a dictionary
                    result.append({"type": "text", "text": str(block)})
            return result
        else:
            # we just default to string if response is not a list
            return str(response)

    def sync_wrapper(*args, **kwargs):
        tool_input = args[0] if args else kwargs
        response = tool.invoke(tool_input)
        return sanitize_response(response)

    async def async_wrapper(*args, **kwargs):
        tool_input = args[0] if args else kwargs
        response = await tool.ainvoke(tool_input)
        return sanitize_response(response)

    return StructuredTool.from_function(
        func=sync_wrapper,
        coroutine=async_wrapper,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )

