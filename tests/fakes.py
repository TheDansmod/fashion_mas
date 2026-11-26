"""Offline stand-ins for the LLM, the agent factory and the catalogue tools."""

import base64
import itertools
import json
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from loguru import logger as log

from frag.prompts import EXPLANATION_PROMPT, REQUEST_FOCUS_PROMPT

FIXTURE_IMAGE = Path(__file__).parent / "fixtures" / "shirt.jpg"
FIXTURE_IMAGE_B64 = base64.b64encode(FIXTURE_IMAGE.read_bytes()).decode()

# a tiny fake catalogue: id -> (category, description)
FAKE_ITEMS = {
    101: ("JEANS", "Dark wash slim jeans in stretch denim."),
    102: ("SHORTS", "Blue cotton chino shorts."),
    103: ("LACE UPS", "Black leather oxford shoes."),
}
UNKNOWN_ITEM_ID = 999_999

# stable substrings of the prompts, used to recognise which node is calling the LLM
_INTENT_MARKER = REQUEST_FOCUS_PROMPT.split("\n", 1)[0][:60]
_EXPLANATION_MARKER = EXPLANATION_PROMPT.strip()[-60:]


class MockStructuredOutput:
    """Returns a canned instance of the requested schema."""

    def __init__(self, schema):
        self.schema = schema

    def invoke(self, *_args, **_kwargs):
        match self.schema.__name__:
            case "NumRecommendations":
                return self.schema(num_recommendations=3)
            case "SingleImageDescription":
                return self.schema(
                    image_description="A blue cotton shirt with short sleeves."
                )
            case "RequiredClothes":
                return self.schema(
                    required_clothes_descriptions=[
                        "Dark wash denim jeans",
                        "Blue shorts",
                        "Black office shoes",
                    ]
                )
            case "CriticalEvaluation":
                # always unsatisfied, so the critique loop runs to its limit
                return self.schema(satisfactory="No", correction="Some correction")
            case name:
                raise NotImplementedError(f"no structured mock for {name}")

    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


class ChatMockLLM:
    """Recognises the calling node from its prompt and returns a canned answer."""

    def invoke(self, prompt, *_args, **_kwargs):
        if isinstance(prompt, str) and _INTENT_MARKER in prompt:
            return AIMessage(content="Focus on the lower-body garments.")
        if isinstance(prompt, list):
            texts = [c["text"] for c in prompt[0].content if c["type"] == "text"]
            if any(_EXPLANATION_MARKER in t for t in texts):
                return AIMessage(content="These items complement the uploaded shirt.")
        raise NotImplementedError(f"ChatMockLLM has no answer for: {prompt!r:.200}")

    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

    def with_structured_output(self, schema, **_kwargs):
        return MockStructuredOutput(schema)


class MockAgent:
    """Stands in for `langchain.agents.create_agent(...)`."""

    def __init__(self, schema, tools, chosen_ids, relevant_image_indexes):
        self.schema = schema
        self.tools = tools or []
        self._chosen_ids = chosen_ids
        self._relevant_image_indexes = relevant_image_indexes

    async def _exercise_tools(self):
        for tool in self.tools:
            if tool.name == "semantic_search":
                result = await tool.ainvoke(
                    {
                        "description": "Black pants",
                        "categories": ["JEANS"],
                        "num_matches": 2,
                    }
                )
                log.debug("mock agent semantic_search -> {} blocks", len(result))
            elif tool.name == "get_product_categories":
                log.debug("mock agent categories -> {}", await tool.ainvoke({}))

    async def ainvoke(self, *_args, **_kwargs):
        match self.schema.__name__:
            case "MatchedImageId":
                await self._exercise_tools()
                return {
                    "structured_response": self.schema(image_id=next(self._chosen_ids))
                }
            case "UpdatedUserRequest":
                return {
                    "structured_response": self.schema(
                        relevant_image_indexes=self._relevant_image_indexes,
                        updated_user_query="Show me similar items in red.",
                    )
                }
            case name:
                raise NotImplementedError(f"no agent mock for {name}")


def make_create_agent(chosen_ids=None, relevant_image_indexes=(0, 2)):
    """Build a `create_agent` replacement with controllable answers."""
    ids = itertools.cycle(chosen_ids or list(FAKE_ITEMS))

    def create_agent(model, response_format, tools=None, **_kwargs):  # noqa: ARG001
        return MockAgent(response_format, tools, ids, list(relevant_image_indexes))

    return create_agent


def _item_blocks(item_id: int, shape: Literal["local", "mcp"]) -> list[dict]:
    category, description = FAKE_ITEMS[item_id]
    text = {
        "type": "text",
        "text": json.dumps(
            {
                "id": item_id,
                "category": category,
                "description": description,
                "price": 10.0,
                "score": 0.5,
            }
        ),
    }
    if shape == "mcp":
        # what langchain-mcp-adapters returns for MCP ImageContent
        image = {"type": "image", "base64": FIXTURE_IMAGE_B64, "mime_type": "image/png"}
    else:
        image = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{FIXTURE_IMAGE_B64}"},
        }
    return [text, image]


class FakeToolsClient:
    """Same interface as LocalToolsClient / MCPToolsClient, over FAKE_ITEMS."""

    def __init__(self, shape: Literal["local", "mcp"] = "local"):
        self.shape = shape

        async def semantic_search(
            description: str,  # noqa: ARG001  mirrors the real tool's LLM-facing schema
            categories: list[str],
            num_matches: int,
        ):
            """Fake semantic search."""
            ids = [i for i, (cat, _) in FAKE_ITEMS.items() if cat in categories]
            return [b for i in ids[:num_matches] for b in _item_blocks(i, shape)]

        def get_product_categories() -> list[str]:
            """Fake categories."""
            return sorted({cat for cat, _ in FAKE_ITEMS.values()})

        async def get_datapoint_by_index(index: int):
            """Fake item lookup; unknown ids fail like a real catalogue miss."""
            if index not in FAKE_ITEMS:
                raise KeyError(index)
            return _item_blocks(index, shape)

        self._llm_tools = [
            StructuredTool.from_function(coroutine=semantic_search),
            StructuredTool.from_function(func=get_product_categories),
        ]
        self._db_tool = StructuredTool.from_function(coroutine=get_datapoint_by_index)

    async def get_llm_tools(self):
        return self._llm_tools

    async def get_db_tool(self):
        return self._db_tool
