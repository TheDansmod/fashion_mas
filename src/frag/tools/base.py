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
Search the catalogue for products that match a text description.

Returns up to `num_matches` products from the listed categories, best match first.
Each product is a JSON metadata block (`id`, `category`, `description`, `price`,
`score`) followed by its image. A higher `score` means a closer match to the text.

Args:
    description (str): What the product looks like, e.g. "black leather belt with a
        silver buckle".
    categories (list[str]): Only products in these categories are returned; use
        names exactly as get_product_categories lists them.
    num_matches (int): How many products to return (at most 6).
"""
GET_PRODUCT_CATEGORIES_DOC = "List the valid product categories for semantic_search."
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
