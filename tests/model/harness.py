"""Harness for the model-quality tests (`pytest -m ollama`).

These run the real agent: the configured Ollama model, the local SigLIP embedder,
Qdrant, and catalogue images and metadata from S3. They are slow (minutes) and
opt-in. To keep S3 use at free-tier scale, the metadata
Parquet and every catalogue image are cached on disk under CACHE_DIR, and a run stops
if it needs more than FRAG_MODEL_TEST_MAX_S3_GETS uncached downloads.

Before running, start Ollama with the memory-capped command in the README
and Qdrant on the SigLIP collection; VECTOR_DB__URL selects the Qdrant instance.
"""

import asyncio
import io
import os
import time
import uuid
from pathlib import Path
from typing import Any

import boto3
import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from loguru import logger as log
from qdrant_client import AsyncQdrantClient, QdrantClient

from frag.agent.graph import FashionAgent
from frag.catalogue.embedders import build_embedder
from frag.catalogue.service import Catalogue
from frag.catalogue.store import (
    CATEGORY,
    DESCRIPTION,
    S3ImageStore,
    build_s3_client,
    load_metadata,
)
from frag.catalogue.vector_db import QdrantConnector
from frag.llm import build_llm
from frag.settings import Settings
from frag.tools.local import LocalToolsClient
from tests.model.rubric import Grade, Item, Turn

CACHE_DIR = Path.home() / ".cache" / "frag-model-tests"
REPORT_DIR = Path("data/model_test_runs")
MAX_S3_GETS = int(os.environ.get("FRAG_MODEL_TEST_MAX_S3_GETS", "300"))
REPEATS = int(os.environ.get("FRAG_MODEL_TEST_REPEATS", "1"))
WARNING_LEVEL = 30

# filled by the tests, written to REPORT_DIR at the end of the session
REPORT: list[dict[str, Any]] = []


class CachedImageStore(S3ImageStore):
    """S3ImageStore with a disk cache and a cap on uncached downloads."""

    downloads = 0

    async def get_png_bytes(self, idx: int) -> bytes:
        """Return the image from the cache, downloading it the first time."""
        path = CACHE_DIR / "images" / f"{idx}.png"
        if path.exists():
            return path.read_bytes()
        if CachedImageStore.downloads >= MAX_S3_GETS:
            msg = f"S3 GET budget of {MAX_S3_GETS} images used up"
            raise RuntimeError(msg)
        CachedImageStore.downloads += 1
        data = await super().get_png_bytes(idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return data


class _CachedParquetClient:
    """Serves `download_fileobj` of the metadata Parquet from the disk cache."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def download_fileobj(self, bucket: str, key: str, buffer: io.BytesIO) -> None:
        path = CACHE_DIR / "metadata.parquet"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(bucket, key, str(path))
        buffer.write(path.read_bytes())


def _skip_unless_ready(settings: Settings) -> None:
    if settings.llm.provider != "ollama":
        # a Bedrock run is billed, so these tests never use it
        pytest.skip("model tests run only with LLM__PROVIDER=ollama")
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        tags = httpx.get(f"{host}/api/tags", timeout=5).json()
    except httpx.HTTPError:
        pytest.skip(f"Ollama is not reachable at {host}")
    if settings.llm.model_id not in {m["name"] for m in tags.get("models", [])}:
        pytest.skip(f"run `ollama pull {settings.llm.model_id}` first")
    try:
        QdrantClient(url=settings.vector_db.url, prefer_grpc=False).get_collection(
            settings.collection
        )
    except Exception as exc:  # noqa: BLE001  any failure means "not available"
        pytest.skip(f"Qdrant collection {settings.collection} unavailable: {exc}")
    if boto3.Session().get_credentials() is None:
        pytest.skip("no AWS credentials for the S3 catalogue")


def load_model_env() -> tuple:
    """Settings, S3 client, metadata, description->category map and the embedder.

    Skips the calling test if the environment is not ready.
    """
    settings = Settings()
    _skip_unless_ready(settings)
    s3 = build_s3_client(settings.aws_region)
    metadata = asyncio.run(load_metadata(_CachedParquetClient(s3), settings.catalogue))
    # description -> category, to grade products by catalogue ground truth
    categories = {m[DESCRIPTION]: m[CATEGORY] for m in metadata.values()}
    embedder = build_embedder(settings)
    return settings, s3, metadata, categories, embedder


class ModelHarness:
    """Builds a fresh agent per test and runs conversations through it."""

    def __init__(self, env: tuple, tmp_path: Path) -> None:
        self.settings, self._s3, self.metadata, self._categories, self._embedder = env
        self._tmp = tmp_path
        self._store = CachedImageStore(
            self.settings.catalogue.s3_bucket,
            self.settings.catalogue.image_key_template,
            self._s3,
        )

    async def catalogue_image(self, item_id: int) -> str:
        """Save a catalogue product's image to a file, to upload as user input."""
        path = self._tmp / f"input_{item_id}.png"
        path.write_bytes(await self._store.get_png_bytes(item_id))
        return str(path)

    async def converse(self, messages: list[tuple[str, list[str]]]) -> list[Turn]:
        """Send the messages on one new thread; return a Turn per message."""
        qdrant = AsyncQdrantClient(
            url=self.settings.vector_db.url,
            prefer_grpc=self.settings.vector_db.prefer_grpc,
        )
        try:
            vector_db = QdrantConnector(
                client=qdrant,
                collection=self.settings.collection,
                vector_name=self.settings.vector_db.image_vector_name,
                category_key=self.settings.catalogue.category_field,
                index_key=self.settings.catalogue.index_field,
            )
            catalogue = Catalogue(self.metadata, self._store, self._embedder, vector_db)
            agent_settings = self.settings.agent.model_copy(
                update={"temp_dir": str(self._tmp / "recommended")}
            )
            agent = FashionAgent(
                build_llm(self.settings.llm, self.settings.aws_region),
                LocalToolsClient(catalogue),
                InMemorySaver(),
                agent_settings,
            )
            thread = str(uuid.uuid4())
            turns = []
            for text, images in messages:
                start = time.monotonic()
                notes: list[str] = []
                sink = log.add(
                    lambda m, notes=notes: notes.append(m.record["message"]),
                    filter=lambda r: (
                        r["level"].no >= WARNING_LEVEL
                        or r["message"].startswith(("Skipping", "Critique:"))
                    ),
                    level="DEBUG",
                )
                try:
                    state = await agent.run_turn(text, images, thread)
                finally:
                    log.remove(sink)
                if CachedImageStore.downloads >= MAX_S3_GETS:
                    pytest.exit(f"S3 GET budget ({MAX_S3_GETS}) reached; stopping")
                turn = self._turn(text, state, time.monotonic() - start)
                turn.notes = notes
                turns.append(turn)
            return turns
        finally:
            await qdrant.close()

    def _turn(self, text: str, state: dict[str, Any], seconds: float) -> Turn:
        items = [
            Item(self._categories.get(d, "UNKNOWN"), d)
            for d in state["recommended_clothes_descriptions"]
        ]
        return Turn(
            message=text,
            request=state["input_text"],
            items=items,
            explanation=state["recommended_clothes_explanation"],
            reference_text=" ".join(state["input_images_descriptions"]),
            seconds=round(seconds, 1),
        )


def record(case: str, run: int, turns: list[Turn], grades: list[Grade]) -> None:
    """Add one graded conversation to the session report."""
    REPORT.append(
        {
            "case": case,
            "run": run,
            "passed": all(g.passed for g in grades),
            "turns": [
                {
                    "message": t.message,
                    "request": t.request,
                    "seconds": t.seconds,
                    "score": g.score,
                    "passed": g.passed,
                    "checks": g.results,
                    "items": [[i.category, i.description] for i in t.items],
                    "explanation": t.explanation,
                    # agent warnings, skipped items and critiques during the turn
                    "notes": t.notes,
                }
                for t, g in zip(turns, grades, strict=True)
            ],
        }
    )
