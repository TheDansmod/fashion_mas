"""LangGraph checkpointer selection: memory, sqlite, postgres or dynamodb."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from loguru import logger as log

from frag.settings import CheckpointerSettings


@asynccontextmanager
async def open_checkpointer(
    settings: CheckpointerSettings, aws_region: str
) -> AsyncIterator[BaseCheckpointSaver]:
    """Open the configured checkpointer and close its connections on exit.

    Backend packages are imported lazily so that only the one in use must be importable.
    """
    log.debug("opening {} checkpointer", settings.backend)
    match settings.backend:
        case "memory":
            yield InMemorySaver()
        case "sqlite":
            from langgraph.checkpoint.sqlite.aio import (
                AsyncSqliteSaver,
            )

            Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            async with AsyncSqliteSaver.from_conn_string(settings.sqlite_path) as saver:
                yield saver
        case "postgres":
            from langgraph.checkpoint.postgres.aio import (
                AsyncPostgresSaver,
            )
            from psycopg_pool import (
                AsyncConnectionPool,
            )

            # autocommit: each checkpoint read/write is its own transaction, which the
            # LangGraph checkpointer's SQL relies on.
            # prepare_threshold=0: disable psycopg prepared-statement caching, which
            # breaks when statements run on different pooled connections.
            async with AsyncConnectionPool(
                conninfo=settings.postgres_dsn,
                max_size=settings.postgres_pool_size,
                kwargs={"autocommit": True, "prepare_threshold": 0},
                open=False,
            ) as pool:
                saver = AsyncPostgresSaver(pool)
                # creates the tables if missing; idempotent
                await saver.setup()
                yield saver
        case "dynamodb":
            from botocore.config import Config
            from langgraph_checkpoint_aws import (
                DynamoDBSaver,
            )

            # adaptive retries add client-side rate limiting, which keeps bursts of
            # agent checkpoints from turning throttling into a retry storm
            yield DynamoDBSaver(
                table_name=settings.dynamodb_table,
                region_name=aws_region,
                ttl_seconds=settings.ttl_seconds,
                enable_checkpoint_compression=settings.compress,
                boto_config=Config(
                    retries={"mode": "adaptive", "max_attempts": 3},
                    max_pool_connections=20,
                ),
            )
