"""Catalogue data access with fake S3 and an in-memory Qdrant (offline)."""

import io

import pandas as pd
import pytest
from qdrant_client import AsyncQdrantClient, models

from frag.data_manager.catalogue import derive_categories
from frag.data_manager.dataset_read_write import FashionGenData, load_metadata
from frag.data_manager.vector_db_read_write import QdrantConnector
from frag.settings import CatalogueSettings
from tests.fakes import FIXTURE_IMAGE

CAT = CatalogueSettings()


def _row(category, description, price=10.0):
    return {
        CAT.category_field: category,
        CAT.description_field: description,
        CAT.price_field: price,
    }


METADATA = {
    1: _row("JEANS", "jeans one"),
    2: _row("SHIRTS", "shirt two"),
    3: _row("JEANS", "jeans three"),
}


class FakeS3:
    """Records requested keys; serves the fixture image and a metadata parquet."""

    def __init__(self, parquet: bytes = b""):
        self.keys = []
        self._parquet = parquet

    def get_object(self, Bucket, Key):  # noqa: N803  boto3 parameter names
        self.keys.append((Bucket, Key))
        return {"Body": io.BytesIO(FIXTURE_IMAGE.read_bytes())}

    def download_fileobj(self, bucket, key, buffer):
        self.keys.append((bucket, key))
        buffer.write(self._parquet)


async def test_get_known_and_unknown_items():
    s3 = FakeS3()
    data = FashionGenData(s3, CAT, METADATA)
    item = await data.get(2)
    assert item["category"] == "SHIRTS"
    assert item["description"] == "shirt two"
    assert item["id"] == 2
    assert item["image"]  # base64 of the fixture
    assert s3.keys == [(CAT.s3_bucket, "images/000/2.png")]
    assert await data.get(12345) == {}  # membership check: unknown ids return {}
    assert len(s3.keys) == 1  # and make no S3 request


async def test_load_metadata_reads_only_needed_columns():
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
    assert set(metadata) == {7, 8}
    assert set(metadata[7]) == {
        CAT.category_field,
        CAT.description_field,
        CAT.price_field,
    }


def test_derive_categories_ignores_missing_values():
    metadata = {**METADATA, 4: _row(None, "no category")}
    assert derive_categories(metadata, CAT.category_field) == ["JEANS", "SHIRTS"]


@pytest.fixture
async def connector():
    data = FashionGenData(FakeS3(), CAT, METADATA)
    conn = QdrantConnector(
        url="http://localhost:6333",
        prefer_grpc=False,
        collection_name="test",
        category_key=CAT.category_field,
        image_vectors_name="images",
        index_key=CAT.index_field,
        data=data,
    )
    client = AsyncQdrantClient(location=":memory:")
    await client.create_collection(
        "test",
        vectors_config={
            "images": models.VectorParams(size=2, distance=models.Distance.COSINE)
        },
    )
    points = [
        (1, [1.0, 0.0], "JEANS"),
        (2, [0.9, 0.1], "SHIRTS"),
        (3, [0.8, 0.2], "JEANS"),
        (4, [1.0, 0.0], "JEANS"),
    ]  # id 4 is in Qdrant but not in the metadata
    await client.upsert(
        "test",
        [
            models.PointStruct(
                id=pid,
                vector={"images": vec},
                payload={CAT.index_field: pid, CAT.category_field: cat},
            )
            for pid, vec, cat in points
        ],
    )
    conn._client = client  # noqa: SLF001  swap in the in-memory client
    yield conn
    await client.close()


async def test_validate_requires_existing_collection(connector):
    await connector.validate()
    connector._collection_name = "missing"  # noqa: SLF001
    with pytest.raises(ValueError, match="does not exist"):
        await connector.validate()


async def test_image_matches_filter_by_category_and_skip_unknown_ids(connector):
    matches = await connector.get_image_matches([1.0, 0.0], ["JEANS"], num_matches=5)
    assert [m["id"] for m in matches] == [1, 3]  # 4 is skipped, 2 filtered out
    assert all(m["category"] == "JEANS" and "score" in m for m in matches)
