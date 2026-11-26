"""What the agent needs from the catalogue tools, and the shared content formats.

Both tool modes (in-process and MCP) give the LLM identical tool output: for each item
a text block with its metadata as JSON, then an `image_url` block with a PNG data URL.
"""

import json
from typing import Any, Protocol

from langchain_core.tools import BaseTool, StructuredTool

from frag.catalogue.models import CatalogueItem

# tool names: the LLM sees the first two; the MCP server also serves the lookup tool
SEMANTIC_SEARCH = "semantic_search"
GET_PRODUCT_CATEGORIES = "get_product_categories"
GET_DATAPOINT_BY_INDEX = "get_datapoint_by_index"
LLM_TOOL_NAMES = (SEMANTIC_SEARCH, GET_PRODUCT_CATEGORIES)

# the LLM reads these descriptions to decide how to call the tools
SEMANTIC_SEARCH_DOC = """\
Get num_matches images and their metadata that match the description and categories.

Given some text description of an item of clothing or an accessory, this
function returns `num_matches` images along with their metadata (which
includes id, price, category, description, and score) from within the images
that belong to the listed categories.

Args:
    description (str): The description to which the returned images are matched.
    categories (list[str]): The matched images will belong to one of these
        listed valid categories.
    num_matches (int): The number of matched images that should be returned.

Returns:
    Matched Images and their metadata. The metadata is a dictionary with the
    keys of `price`, `category`, `description`, `id`, and `score`. The score
    tells how good of a match (to the input text) the returned item is.
"""
GET_PRODUCT_CATEGORIES_DOC = "Returns a list of valid product categories."
GET_DATAPOINT_BY_INDEX_DOC = (
    "Get a datapoint, including image and metadata, using index in db."
)

Block = dict[str, Any]


class ToolsClient(Protocol):
    """Catalogue access for the agent, backed by in-process tools or an MCP server."""

    async def get_llm_tools(self) -> list[BaseTool]:
        """The tools offered to the LLM."""
        ...

    async def get_item(self, item_id: int) -> CatalogueItem | None:
        """Look up one item (None if the id is not in the catalogue)."""
        ...


def item_metadata(item: CatalogueItem) -> dict[str, Any]:
    """The metadata shown to the LLM next to an item's image."""
    return {
        "price": item.price,
        "category": item.category,
        "description": item.description,
        "id": item.id,
        "score": item.score,
    }


def item_to_blocks(item: CatalogueItem) -> list[Block]:
    """Format one item as LangChain content blocks (metadata text, then the image)."""
    return [
        {"type": "text", "text": json.dumps(item_metadata(item))},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{item.image_b64}"},
        },
    ]


def text_blocks(text: str) -> list[Block]:
    """A plain-text tool result (category lists, error messages)."""
    return [{"type": "text", "text": text}]


def _normalise_block(block: object) -> Block:
    if not isinstance(block, dict):
        return {"type": "text", "text": str(block)}
    match block.get("type"):
        case "text":
            return {"type": "text", "text": block["text"]}
        case "image" if "base64" in block:
            # LangChain standard block, as produced by langchain-mcp-adapters for
            # MCP ImageContent
            mime = block.get("mime_type", "image/png")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{block['base64']}"},
            }
        case "image_url":
            return block
        case other:
            msg = f"unexpected tool content block type: {other!r}"
            raise ValueError(msg)


def normalise_blocks(content: object) -> list[Block]:
    """Convert tool output into the blocks `item_to_blocks` produces."""
    if isinstance(content, list):
        return [_normalise_block(block) for block in content]
    return text_blocks(str(content))


def normalise_mcp_blocks(tool: BaseTool) -> StructuredTool:
    """Wrap an MCP tool so its output has the same shape as the in-process tools."""

    async def call(**kwargs: Any) -> list[Block]:  # noqa: ANN401  arguments follow the tool's own schema
        return normalise_blocks(await tool.ainvoke(kwargs))

    return StructuredTool.from_function(
        coroutine=call,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


def blocks_to_item(blocks: list[Block]) -> CatalogueItem | None:
    """Parse `item_to_blocks` output back into an item (None if it holds no item)."""
    metadata, image_b64 = None, None
    for block in normalise_blocks(blocks):
        if block["type"] == "image_url":
            image_b64 = block["image_url"]["url"].split("base64,", 1)[-1]
        elif metadata is None:
            try:
                parsed = json.loads(block["text"])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "id" in parsed:
                metadata = parsed
    if metadata is None or image_b64 is None:
        return None
    return CatalogueItem(
        id=int(metadata["id"]),
        category=metadata["category"],
        description=metadata["description"],
        price=float(metadata["price"]),
        image_b64=image_b64,
        score=metadata.get("score"),
    )
