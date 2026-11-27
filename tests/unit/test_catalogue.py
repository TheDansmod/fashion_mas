"""Catalogue store, vector search and service with fake S3 and in-memory Qdrant."""

import base64
import io
import json

import pandas as pd
import pytest

from frag.catalogue.embedders import NovaEmbedder, build_embedder
from frag.catalogue.service import MAX_SEARCH_MATCHES, InvalidCategoryError
from frag.catalogue.store import S3ImageStore, load_metadata
from frag.settings import EmbeddingSettings
from tests.fakes import (
    CATALOGUE_SETTINGS as CAT,
)
from tests.fakes import (
    FAKE_ITEMS,
    FIXTURE_IMAGE,
    ORPHAN_ID,
    FakeS3,
    build_fake_catalogue,
    fake_connector,
)


async def test_image_store_keys_and_bytes():
    s3 = FakeS3()
    store = S3ImageStore(CAT.s3_bucket, CAT.image_key_template, s3)
    assert store.key(12345) == "images/012/12345.png"
    assert await store.get_png_bytes(7) == FIXTURE_IMAGE.read_bytes()
    assert s3.keys == [(CAT.s3_bucket, "images/000/7.png")]


async def test_load_metadata_reads_only_needed_columns_with_canonical_keys():
    df = pd.DataFrame(
        {
            CAT.index_field: [7, 8],
            CAT.category_field: ["HATS", "BOOTS"],
            CAT.description_field: ["a hat", "a boot"],
            CAT.price_field: [1.0, 2.0],
            "input_composition": ["wool", "leather"],
        }
    )
    buffer = io.BytesIO()
    df.to_parquet(buffer)
    metadata = await load_metadata(FakeS3(buffer.getvalue()), CAT)
    assert metadata == {
        7: {"category": "HATS", "description": "a hat", "price": 1.0},
        8: {"category": "BOOTS", "description": "a boot", "price": 2.0},
    }
    assert all(type(k) is int for k in metadata)


async def test_validate_checks_collection_vector_name_and_dim(qdrant):
    await fake_connector(qdrant).validate(expected_dim=2)
    with pytest.raises(ValueError, match="does not exist"):
        await fake_connector(qdrant, collection="missing").validate(expected_dim=2)
    with pytest.raises(ValueError, match="768-d vectors"):
        await fake_connector(qdrant).validate(expected_dim=768)
    conn = fake_connector(qdrant)
    conn._vector_name = "text"  # noqa: SLF001
    with pytest.raises(ValueError, match="no vector named 'text'"):
        await conn.validate(expected_dim=2)


async def test_vector_search_filters_by_category(qdrant):
    hits = await fake_connector(qdrant).search([1.0, 0.0], ["SHORTS", "LACE UPS"], k=5)
    assert [i for i, _ in hits] == [102, 103]  # best first; JEANS filtered out
    assert hits[0][1] > hits[1][1]


async def test_catalogue_categories_come_from_metadata(catalogue):
    assert catalogue.categories == sorted({c for c, _ in FAKE_ITEMS.values()})


async def test_catalogue_search_returns_items_and_skips_ids_without_metadata(catalogue):
    items = await catalogue.search("dark jeans", ["JEANS"], k=5)
    # ORPHAN_ID (JEANS, same vector) is in Qdrant but not in the metadata
    assert [i.id for i in items] == [101]
    (item,) = items
    assert item.category == "JEANS"
    assert item.description == FAKE_ITEMS[101][1]
    assert base64.b64decode(item.image_b64) == FIXTURE_IMAGE.read_bytes()
    assert item.score == pytest.approx(1.0)
    assert ORPHAN_ID not in {i.id for i in await catalogue.search("x", [], k=10)}


async def test_catalogue_search_clamps_the_number_of_matches(catalogue, monkeypatch):
    requested = []
    search = catalogue._vector_db.search  # noqa: SLF001

    async def spy(vector, categories, k):
        requested.append(k)
        return await search(vector, categories, k)

    monkeypatch.setattr(catalogue._vector_db, "search", spy)  # noqa: SLF001
    await catalogue.search("x", [], k=0)
    await catalogue.search("x", [], k=50)
    assert requested == [1, MAX_SEARCH_MATCHES]


async def test_catalogue_search_rejects_invalid_categories(catalogue):
    with pytest.raises(InvalidCategoryError, match="HATS"):
        await catalogue.search("a hat", ["JEANS", "HATS"], k=3)


async def test_catalogue_search_drops_items_whose_image_fails(qdrant):
    s3 = FakeS3(missing_keys=frozenset({"images/000/102.png"}))
    catalogue = build_fake_catalogue(qdrant, s3)
    items = await catalogue.search("x", ["JEANS", "SHORTS"], k=5)
    assert [i.id for i in items] == [101]


async def test_catalogue_get(catalogue):
    item = await catalogue.get(103)
    assert item is not None
    assert (item.id, item.category, item.score) == (103, "LACE UPS", None)
    assert await catalogue.get(999) is None


def test_build_embedder_nova_makes_no_requests(clean_env):
    settings = clean_env(embedding=EmbeddingSettings(backend="nova"))
    embedder = build_embedder(settings)
    assert isinstance(embedder, NovaEmbedder)
    assert embedder.dim == 1024


async def test_nova_embedder_request_body():
    class FakeBedrock:
        def invoke_model(self, **kwargs):
            self.kwargs = kwargs
            body = json.dumps({"embeddings": [{"embedding": [0.1, 0.2]}]})
            return {"body": io.BytesIO(body.encode())}

    embedder = NovaEmbedder(
        "model-x", dim=256, purpose="GENERIC_INDEX", region="us-east-1"
    )
    embedder._client = FakeBedrock()  # noqa: SLF001  no Bedrock call in tests
    assert await embedder.embed_text("red dress") == [0.1, 0.2]
    sent = json.loads(embedder._client.kwargs["body"])  # noqa: SLF001
    assert embedder._client.kwargs["modelId"] == "model-x"  # noqa: SLF001
    params = sent["singleEmbeddingParams"]
    assert params["text"]["value"] == "red dress"
    assert params["embeddingDimension"] == 256
