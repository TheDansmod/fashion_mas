"""Offline stand-ins for the LLM, the agent factory, the tools and the catalogue."""

import base64
import io
import itertools
import json
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from loguru import logger as log
from qdrant_client import AsyncQdrantClient, models

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
