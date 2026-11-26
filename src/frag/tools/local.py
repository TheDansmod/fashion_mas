"""In-process catalogue tools (TOOLS__MODE=local)."""

import json

from langchain_core.tools import BaseTool, StructuredTool
from loguru import logger as log

from frag.catalogue.models import CatalogueItem
from frag.catalogue.service import Catalogue, InvalidCategoryError
from frag.tools.base import (
    GET_PRODUCT_CATEGORIES,
    GET_PRODUCT_CATEGORIES_DOC,
    SEMANTIC_SEARCH,
    SEMANTIC_SEARCH_DOC,
    Block,
    item_to_blocks,
    text_blocks,
)


class LocalToolsClient:
    """LangChain tools that call the catalogue directly."""

    def __init__(self, catalogue: Catalogue) -> None:
        """Build the tools over `catalogue`."""
        self._catalogue = catalogue

        async def semantic_search(
            description: str, categories: list[str], num_matches: int
        ) -> list[Block]:
            log.debug("semantic_search tool call: {!r} in {}", description, categories)
            try:
                items = await catalogue.search(description, categories, num_matches)
            except InvalidCategoryError as exc:
                return text_blocks(f"Error: {exc}")
            return [block for item in items for block in item_to_blocks(item)]

        def get_product_categories() -> list[Block]:
            log.debug("get_product_categories tool call")
            return text_blocks(json.dumps(catalogue.categories))

        self._tools: list[BaseTool] = [
            StructuredTool.from_function(
                coroutine=semantic_search,
                name=SEMANTIC_SEARCH,
                description=SEMANTIC_SEARCH_DOC,
            ),
            StructuredTool.from_function(
                func=get_product_categories,
                name=GET_PRODUCT_CATEGORIES,
                description=GET_PRODUCT_CATEGORIES_DOC,
            ),
        ]

    async def get_llm_tools(self) -> list[BaseTool]:
        """The tools offered to the LLM."""
        return self._tools

    async def get_item(self, item_id: int) -> CatalogueItem | None:
        """Look up one item in the catalogue."""
        return await self._catalogue.get(item_id)
