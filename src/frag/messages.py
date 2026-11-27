"""Build multimodal chat messages from text and image files."""

import base64
import mimetypes
from pathlib import Path

from langchain_core.messages import HumanMessage

# an image's type when neither its content nor its extension identifies it
DEFAULT_IMAGE_TYPE = "image/png"


def sniff_image_type(data: bytes) -> str | None:
    """The image type from the file signature, for the formats Bedrock accepts.

    The content decides, not the name: files from the web are often misnamed (a WebP
    saved as `.jpg`), and Bedrock rejects an image whose declared type doesn't match
    its bytes.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def image_block(path: Path) -> dict:
    """An `image_url` content block holding the file as a base64 data URL."""
    raw = path.read_bytes()
    mime = sniff_image_type(raw)
    if mime is None:
        mime, _ = mimetypes.guess_type(path.name)
        if not (mime and mime.startswith("image/")):
            mime = DEFAULT_IMAGE_TYPE
    data = base64.b64encode(raw).decode("ascii")
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
