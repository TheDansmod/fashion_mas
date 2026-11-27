"""Offline stand-ins for the LLM, the agent factory, the tools and the catalogue."""

import base64
import collections
import io
import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from loguru import logger as log
from qdrant_client import AsyncQdrantClient, models

from frag.agent.schemas import MatchedImageId
from frag.catalogue.models import CatalogueItem
from frag.catalogue.service import Catalogue
from frag.catalogue.store import CATEGORY, DESCRIPTION, PRICE, S3ImageStore
from frag.catalogue.vector_db import QdrantConnector
from frag.prompts import EXPLANATION_PROMPT, REQUEST_FOCUS_PROMPT
from frag.settings import CatalogueSettings
from frag.tools.base import item_to_blocks, text_blocks

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
    """Returns a canned instance of the requested schema (async only)."""

    def __init__(self, schema, llm):
        self.schema = schema
        self._llm = llm

    async def ainvoke(self, prompt, *_args, **_kwargs):
        self._llm.calls.append((self.schema.__name__, prompt))
        match self.schema.__name__:
            case "NumRecommendations":
                return self.schema(
                    num_recommendations=self._llm.num_recommendations,
                    item_types=self._llm.item_types,
                )
            case "SingleImageDescription":
                return self.schema(
                    image_description="A blue cotton shirt with short sleeves."
                )
            case "RequiredClothes":
                answer = (
                    self._llm.modifier_answers.pop(0)
                    if self._llm.modifier_answers
                    else [("", d) for d in DEFAULT_CHOICES]
                )
                return self.schema(
                    items=[{"item_type": t, "description": d} for t, d in answer]
                )
            case "UpdatedUserRequest":
                return self.schema(
                    relevant_image_indexes=self._llm.relevant_image_indexes,
                    updated_user_query="Show me similar items in red.",
                )
            case "CriticalEvaluation":
                if self._llm.critic_satisfied:
                    return self.schema(satisfactory="Yes")
                return self.schema(satisfactory="No", correction="Some correction")
            case name:
                raise NotImplementedError(f"no structured mock for {name}")


class ChatMockLLM:
    """Recognises the calling node from its prompt and returns a canned answer.

    Only async calls are implemented, so a node that calls the LLM synchronously
    fails. `calls` records `(schema name or "chat", prompt)` for every call. By
    default the critic is never satisfied, so the critique loop runs to its limit.
    """

    def __init__(
        self,
        *,
        critic_satisfied=False,
        num_recommendations=3,
        item_types=(),
        relevant_image_indexes=(),
        modifier_answers=(),
    ):
        self.calls = []
        self.item_types = list(item_types)
        # successive modifier answers as [(item_type, description), ...]; after
        # they run out, DEFAULT_CHOICES with no item types
        self.modifier_answers = list(modifier_answers)
        self.relevant_image_indexes = list(relevant_image_indexes)
        self.critic_satisfied = critic_satisfied
        self.num_recommendations = num_recommendations

    async def ainvoke(self, prompt, *_args, **_kwargs):
        self.calls.append(("chat", prompt))
        if isinstance(prompt, str) and _INTENT_MARKER in prompt:
            return AIMessage(content="Focus on the lower-body garments.")
        if isinstance(prompt, list):
            texts = [c["text"] for c in prompt[0].content if c["type"] == "text"]
            if any(_EXPLANATION_MARKER in t for t in texts):
                return AIMessage(content="These items complement the uploaded shirt.")
        raise NotImplementedError(f"ChatMockLLM has no answer for: {prompt!r:.200}")

    def with_structured_output(self, schema, **_kwargs):
        return MockStructuredOutput(schema, self)


# what the mock RequiredClothes answer asks for, and the item the mock agent picks
DEFAULT_CHOICES = {
    "Dark wash denim jeans": [101],
    "Blue shorts": [102],
    "Black office shoes": [103],
}


class MockAgent:
    """Stands in for the recommender's tool-using `create_agent(...)`.

    It calls the real `semantic_search` tool over every fake item (unless `search` is
    off), then answers from `choices`: item description -> ids returned on successive
    calls for that description (the last one repeats). Without a matching key it
    picks the first item the search returned. With `prose`, it answers in text
    instead of the structured response, like small local models often do.
    """

    def __init__(self, tools, choices, calls, *, search, prose):
        self.tools = {t.name: t for t in tools or []}
        self._choices = choices
        self._calls = calls
        self._search = search
        self._prose = prose

    async def _search_all(self):
        blocks = await self.tools["semantic_search"].ainvoke(
            {
                "description": "anything",
                "categories": sorted({cat for cat, _ in FAKE_ITEMS.values()}),
                "num_matches": 10,
            }
        )
        log.debug("mock agent semantic_search -> {} blocks", len(blocks))
        return ToolMessage(
            content=blocks, name="semantic_search", tool_call_id="call-search"
        )

    def _choose(self, prompt, tool_message):
        description = prompt.rsplit("Description:", 1)[-1].strip()
        if description in self._choices:
            ids = self._choices[description]
            n = self._calls[description]
            self._calls[description] += 1
            return ids[min(n, len(ids) - 1)]
        first_text = tool_message.content[0]["text"] if tool_message else "{}"
        return json.loads(first_text).get("id", UNKNOWN_ITEM_ID)

    async def ainvoke(self, agent_input, *_args, **_kwargs):
        messages = agent_input["messages"]
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]
        tool_message = await self._search_all() if self._search else None
        image_id = self._choose(messages[0].content, tool_message)
        messages = [*messages, *([tool_message] if tool_message else [])]
        if self._prose:
            answer = AIMessage(content=f"Out of 10 results, item {image_id} fits best.")
            return {"messages": [*messages, answer]}
        return {
            "structured_response": MatchedImageId(image_id=image_id),
            "messages": messages,
        }


def make_create_agent(choices=None, *, search=True, prose=False):
    """Build a `create_agent` replacement with controllable answers."""
    choices = DEFAULT_CHOICES if choices is None else choices
    calls = collections.Counter()

    def create_agent(model, response_format, tools=None, **_kwargs):  # noqa: ARG001
        assert response_format is MatchedImageId  # only the recommender uses agents
        return MockAgent(tools, choices, calls, search=search, prose=prose)

    return create_agent


def fake_item(item_id: int, score: float | None = None) -> CatalogueItem:
    category, description = FAKE_ITEMS[item_id]
    return CatalogueItem(
        id=item_id,
        category=category,
        description=description,
        price=10.0,
        image_b64=FIXTURE_IMAGE_B64,
        score=score,
    )


class FakeToolsClient:
    """Same interface as LocalToolsClient / MCPToolsClient, over FAKE_ITEMS.

    Ids in `failing_ids` raise on lookup, like an S3 or MCP error; unknown ids return
    None, like a real catalogue miss.
    """

    def __init__(self, failing_ids: frozenset[int] = frozenset()):
        self._failing_ids = failing_ids

        async def semantic_search(
            description: str,  # noqa: ARG001  mirrors the real tool's LLM-facing schema
            categories: list[str],
            num_matches: int,
        ):
            """Fake semantic search."""
            ids = [i for i, (cat, _) in FAKE_ITEMS.items() if cat in categories]
            return [b for i in ids[:num_matches] for b in item_to_blocks(fake_item(i))]

        def get_product_categories():
            """Fake categories."""
            return text_blocks(json.dumps(sorted({c for c, _ in FAKE_ITEMS.values()})))

        self._llm_tools = [
            StructuredTool.from_function(coroutine=semantic_search),
            StructuredTool.from_function(func=get_product_categories),
        ]

    async def get_llm_tools(self):
        return self._llm_tools

    async def get_item(self, item_id: int) -> CatalogueItem | None:
        if item_id in self._failing_ids:
            raise RuntimeError(f"lookup of {item_id} failed")
        return fake_item(item_id) if item_id in FAKE_ITEMS else None


# --- a real Catalogue over fake S3, a fake embedder and an in-memory Qdrant ----------

CATALOGUE_SETTINGS = CatalogueSettings()
# vectors per item: the fake embedder maps every text to [1, 0]
FAKE_VECTORS = {101: [1.0, 0.0], 102: [0.9, 0.1], 103: [0.0, 1.0]}
# in Qdrant but missing from the metadata (e.g. the index is larger than a subset)
ORPHAN_ID, ORPHAN_CATEGORY = 104, "JEANS"


class FakeS3:
    """Records requested keys; serves the fixture image and an optional Parquet."""

    def __init__(
        self, parquet: bytes = b"", missing_keys: frozenset[str] = frozenset()
    ):
        self.keys: list[tuple[str, str]] = []
        self._parquet = parquet
        self._missing = missing_keys

    def get_object(self, Bucket, Key):  # noqa: N803  boto3 parameter names
        self.keys.append((Bucket, Key))
        if Key in self._missing:
            raise KeyError(Key)
        return {"Body": io.BytesIO(FIXTURE_IMAGE.read_bytes())}

    def download_fileobj(self, bucket, key, buffer):
        self.keys.append((bucket, key))
        buffer.write(self._parquet)


class FakeEmbedder:
    dim = 2

    def __init__(self):
        self.texts: list[str] = []

    async def embed_text(self, text: str) -> list[float]:
        self.texts.append(text)
        return [1.0, 0.0]


async def fake_qdrant(collection: str = "test") -> AsyncQdrantClient:
    client = AsyncQdrantClient(location=":memory:")
    await client.create_collection(
        collection,
        vectors_config={
            "images": models.VectorParams(size=2, distance=models.Distance.COSINE)
        },
    )
    points = [(i, vec, FAKE_ITEMS[i][0]) for i, vec in FAKE_VECTORS.items()]
    points.append((ORPHAN_ID, [1.0, 0.0], ORPHAN_CATEGORY))
    await client.upsert(
        collection,
        [
            models.PointStruct(
                id=pid,
                vector={"images": vec},
                payload={
                    CATALOGUE_SETTINGS.index_field: pid,
                    CATALOGUE_SETTINGS.category_field: cat,
                },
            )
            for pid, vec, cat in points
        ],
    )
    return client


def fake_connector(
    client: AsyncQdrantClient, collection: str = "test"
) -> QdrantConnector:
    return QdrantConnector(
        client=client,
        collection=collection,
        vector_name="images",
        category_key=CATALOGUE_SETTINGS.category_field,
        index_key=CATALOGUE_SETTINGS.index_field,
    )


def build_fake_catalogue(
    client: AsyncQdrantClient, s3: FakeS3 | None = None
) -> Catalogue:
    metadata = {
        i: {CATEGORY: cat, DESCRIPTION: desc, PRICE: 10.0}
        for i, (cat, desc) in FAKE_ITEMS.items()
    }
    store = S3ImageStore(
        CATALOGUE_SETTINGS.s3_bucket,
        CATALOGUE_SETTINGS.image_key_template,
        s3 or FakeS3(),
    )
    return Catalogue(metadata, store, FakeEmbedder(), fake_connector(client))
