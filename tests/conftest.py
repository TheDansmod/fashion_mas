import contextlib
import os

import pytest
from _pytest.logging import LogCaptureFixture
from langgraph.checkpoint.memory import InMemorySaver
from loguru import logger

from frag.rag_pipeline.rag_agent import FashionAgent
from frag.settings import AgentSettings, Settings
from tests.fakes import (
    ChatMockLLM,
    FakeToolsClient,
    build_fake_catalogue,
    fake_qdrant,
    make_create_agent,
)


@pytest.fixture
def mocked_create_agent(monkeypatch):
    """Replace create_agent inside the agent module with the offline mock."""
    monkeypatch.setattr("frag.rag_pipeline.rag_agent.create_agent", make_create_agent())


@pytest.fixture
def make_agent(tmp_path):
    """Factory for a FashionAgent wired entirely to fakes."""

    def _make(tools=None, **agent_settings):
        return FashionAgent(
            llm=ChatMockLLM(),
            tools=tools or FakeToolsClient(),
            checkpointer=InMemorySaver(),
            settings=AgentSettings(
                temp_dir=str(tmp_path / "temp_images"), **agent_settings
            ),
        )

    return _make


@pytest.fixture
async def qdrant():
    """An in-memory Qdrant with the fake catalogue's vectors in collection "test"."""
    client = await fake_qdrant()
    yield client
    await client.close()


@pytest.fixture
def catalogue(qdrant):
    """A real Catalogue over fake S3, a fake embedder and the in-memory Qdrant."""
    return build_fake_catalogue(qdrant)


@pytest.fixture
def clean_env(monkeypatch):
    """Remove every environment variable Settings would read; ignore .env."""
    for key in list(os.environ):
        if key.lower().split("__", 1)[0] in Settings.model_fields:
            monkeypatch.delenv(key)

    def _settings(**kwargs):
        return Settings(_env_file=None, **kwargs)

    return _settings


# loguru does not use the stdlib logging system, so route it into pytest's caplog
@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,
    )
    yield caplog
    with contextlib.suppress(ValueError):
        logger.remove(handler_id)


# opt-in debug log of a test run: FRAG_TEST_LOG_FILE=logs/pytest.log uv run pytest
@pytest.fixture(autouse=True, scope="session")
def _optional_log_file():
    path = os.environ.get("FRAG_TEST_LOG_FILE")
    handler_id = (
        logger.add(path, level="DEBUG", backtrace=True, diagnose=True) if path else None
    )
    yield
    if handler_id is not None:
        logger.remove(handler_id)
