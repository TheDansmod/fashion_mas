"""Build multimodal chat messages from text and image files."""

import base64
import mimetypes
from pathlib import Path

from langchain_core.messages import HumanMessage

# an image path's type when its extension is unknown or missing
DEFAULT_IMAGE_TYPE = "image/png"


def image_block(path: Path) -> dict:
    """An `image_url` content block holding the file as a base64 data URL."""
    mime, _ = mimetypes.guess_type(path.name)
    if not (mime and mime.startswith("image/")):
        mime = DEFAULT_IMAGE_TYPE
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def build_message(parts: list[str | Path]) -> list[HumanMessage]:
    """One human message with the parts in order: a `str` is text, a `Path` an image.

    Image paths must be `Path` objects, so that a text that looks like a path is never
    read from disk.
    """
    content = [
        image_block(part) if isinstance(part, Path) else {"type": "text", "text": part}
        for part in parts
    ]
    return [HumanMessage(content=content)]
