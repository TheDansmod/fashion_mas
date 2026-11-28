"""Catalogue data on S3 (in every mode): item images and the metadata Parquet."""

import asyncio
import io
from typing import Any

import boto3
import pyarrow.parquet as pq
from botocore.config import Config
from loguru import logger as log

from frag.settings import CatalogueSettings

# shared by every S3 call for the catalogue: searches fetch several images concurrently
S3_CONFIG = Config(
    max_pool_connections=20, retries={"mode": "adaptive", "max_attempts": 3}
)

# canonical metadata keys, independent of the Parquet column names
CATEGORY, DESCRIPTION, PRICE = "category", "description", "price"


def build_s3_client(region: str) -> Any:  # noqa: ANN401  boto3 clients are untyped
    """Create the catalogue S3 client (no requests are made here)."""
    return boto3.client("s3", region_name=region, config=S3_CONFIG)


class S3ImageStore:
    """Reads catalogue images from S3; keys are built from a template."""

    def __init__(
        self,
        bucket: str,
        key_template: str,
        client: Any,  # noqa: ANN401  boto3 clients are untyped
    ) -> None:
        """Store the bucket, key template and S3 client."""
        self._bucket = bucket
        self._key_template = key_template
        self._client = client

    def key(self, idx: int) -> str:
        """Return the S3 key of item `idx` (images are sharded into folders of 1000)."""
        return self._key_template.format(shard=idx // 1000, idx=idx)

    async def get_png_bytes(self, idx: int) -> bytes:
        """Download the image of item `idx`."""
        # boto3 is synchronous: run it in a thread so the shared event loop (Chainlit
        # or the MCP server) is not blocked during the download
        response = await asyncio.to_thread(
            self._client.get_object, Bucket=self._bucket, Key=self.key(idx)
        )
        return await asyncio.to_thread(response["Body"].read)


async def load_metadata(
    client: Any,  # noqa: ANN401  boto3 clients are untyped
    settings: CatalogueSettings,
) -> dict[int, dict[str, Any]]:
    """Download the metadata Parquet and index it by catalogue index.

    Only the needed columns are read, which keeps memory low for 260k rows. The
    returned dicts use the canonical keys `category`, `description` and `price`.
    """
    log.debug(
        "loading catalogue metadata from s3://{}/{}",
        settings.s3_bucket,
        settings.metadata_key,
    )
    buffer = io.BytesIO()
    await asyncio.to_thread(
        client.download_fileobj, settings.s3_bucket, settings.metadata_key, buffer
    )
    buffer.seek(0)
    renames = {
        settings.category_field: CATEGORY,
        settings.description_field: DESCRIPTION,
        settings.price_field: PRICE,
    }
    table = pq.read_table(buffer, columns=[settings.index_field, *renames])
    df = table.to_pandas().rename(columns=renames).set_index(settings.index_field)
    metadata = {int(idx): row for idx, row in df.to_dict(orient="index").items()}
    log.info("loaded metadata for {} catalogue items", len(metadata))
    return metadata
