"""Read catalogue items (image from S3 + metadata) by index."""

import asyncio
import base64
import io
from typing import Any

import pyarrow.parquet as pq
from loguru import logger as log

from frag.settings import CatalogueSettings


async def load_metadata(
    s3_client: Any, settings: CatalogueSettings  # noqa: ANN401  boto3 clients are untyped
) -> dict[int, dict[str, Any]]:
    """Download the metadata Parquet from S3 and index it by catalogue index.

    Only the columns the app uses are read, which keeps memory low for 260k rows.
    """
    log.debug(
        "loading catalogue metadata from s3://{}/{}",
        settings.s3_bucket,
        settings.metadata_key,
    )
    buffer = io.BytesIO()
    await asyncio.to_thread(
        s3_client.download_fileobj, settings.s3_bucket, settings.metadata_key, buffer
    )
    buffer.seek(0)
    columns = [
        settings.index_field,
        settings.category_field,
        settings.description_field,
        settings.price_field,
    ]
    df = pq.read_table(buffer, columns=columns).to_pandas()
    df = df.set_index(settings.index_field)
    metadata = df.to_dict(orient="index")
    log.info("loaded metadata for {} catalogue items", len(metadata))
    return metadata


class FashionGenData:
    """Fetches catalogue items: the image from S3 and metadata from the lookup."""

    def __init__(
        self,
        s3_client: Any,  # noqa: ANN401  boto3 clients are untyped
        settings: CatalogueSettings,
        metadata: dict[int, dict[str, Any]],
    ) -> None:
        """Store the S3 client, catalogue settings and the preloaded metadata."""
        self._s3 = s3_client
        self._settings = settings
        self.metadata = metadata

    async def get(self, fetch_index: int) -> dict[str, Any]:
        """Return `{image (base64 PNG), price, category, description, id}`.

        Returns an empty dict if the index is not in the catalogue (membership, not a
        range check, so catalogue subsets work).
        """
        if fetch_index not in self.metadata:
            return {}
        # boto3 is synchronous - run it in a thread so the event loop (shared by
        # Chainlit / the MCP server) is not blocked during the download
        response = await asyncio.to_thread(
            self._s3.get_object,
            Bucket=self._settings.s3_bucket,
            Key=self._settings.image_key(fetch_index),
        )
        image_bytes = await asyncio.to_thread(response["Body"].read)
        item = self.metadata[fetch_index]
        return {
            "image": base64.b64encode(image_bytes).decode("utf-8"),
            "price": item[self._settings.price_field],
            "category": item[self._settings.category_field],
            "description": item[self._settings.description_field],
            "id": fetch_index,
        }
