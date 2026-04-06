"""Checkpointer provider abstractions and concrete implementations."""

from abc import ABC, abstractmethod

import aiosqlite
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

class CheckpointerProvider(ABC):
    """Abstract lifecycle manager for a LangGraph checkpointer.

    Mirrors the pattern of get_llm_model: callers depend only on this
    abstract type and never on a concrete backend.
    """

    @abstractmethod
    async def start(self) -> BaseCheckpointSaver:
        """Initialise backend resources and return a ready checkpointer."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Tear down all held resources cleanly."""
        ...


class SqliteCheckpointerProvider(CheckpointerProvider):
    """Single-connection SQLite backend. Suitable for local / dev use."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._connection: aiosqlite.Connection | None = None

    async def start(self) -> BaseCheckpointSaver:
        self._connection = await aiosqlite.connect(self._db_path)
        return AsyncSqliteSaver(self._connection)

    async def stop(self) -> None:
        if self._connection:
            await self._connection.close()
            self._connection = None


class PostgresCheckpointerProvider(CheckpointerProvider):
    """Async connection-pool Postgres backend. Supports simultaneous R/W."""

    def __init__(self, conn_string: str, max_size: int) -> None:
        self._conn_string = conn_string
        self._max_size = max_size
        self._pool: AsyncConnectionPool | None = None

    async def start(self) -> BaseCheckpointSaver:
        # autocommit: True - each checkpoint read/write operates as its own auto-committed transaction, which is required by the LangGraph checkpointer's internal SQL logic
        # prepare_threshold: 0 - disables psycopg3's prepared-statement caching, which is necessary when using a connection pool (different connections would otherwise lose each other's prepared statements)
        self._pool = AsyncConnectionPool(
            conninfo=self._conn_string,
            max_size=self._max_size,
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=False,
        )
        await self._pool.open()
        checkpointer = AsyncPostgresSaver(self._pool)
        # Call setup() once to create the required Postgres tables.
        # It is idempotent - safe to call on every startup, but only hits the DB
        # if the tables don't already exist.
        await checkpointer.setup()
        return checkpointer

    async def stop(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None


# class DynamoDBCheckpointerProvider(CheckpointerProvider):
#     """Async AWS checkpointer"""
#     def __init__(self, table_name: str, 

def create_checkpointer_provider(
    backend: str,
    sqlite_db_path: str,
    postgres_dsn: str,
    postgres_max_pool_size: int,
) -> CheckpointerProvider:
    """Factory: returns the correct provider based on configured backend.

    backend: Literal['sqlite', 'postgres', 'dynamodb']

    For sqlite: need db_path

    For postgres: need max_pool_size, dsn
    """
    match backend:
        case "sqlite":
            return SqliteCheckpointerProvider(db_path=sqlite_db_path)
        case "postgres":
            return PostgresCheckpointerProvider(
                conn_string=postgres_dsn,
                max_size=postgres_max_pool_size,
            )
        case _:
            raise ValueError(
                f"Unsupported checkpointer backend: {backend!r}. "
                "Expected 'sqlite' or 'postgres'."
            )
