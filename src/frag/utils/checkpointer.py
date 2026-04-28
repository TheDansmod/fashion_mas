"""Checkpointer provider abstractions and concrete implementations."""

from abc import ABC, abstractmethod

from langgraph.checkpoint.base import BaseCheckpointSaver


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
        self._connection = None

    async def start(self) -> BaseCheckpointSaver:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        import aiosqlite

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
        self._pool = None

    async def start(self) -> BaseCheckpointSaver:
        from psycopg_pool import AsyncConnectionPool
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

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


class DynamoDBCheckpointerProvider(CheckpointerProvider):
    """Async AWS checkpointer"""

    def __init__(
        self,
        region_name: str,
        table_name: str,
        ttl_seconds: int,
        do_compression: bool,
        max_pool_size: int,
        retry_mode: str,
        max_retry_attempts: int,
    ):
        import boto3

        self.session = boto3.Session(
            region_name=region_name,
        )
        self.table_name = table_name
        self.ttl_seconds = ttl_seconds
        self.do_compression = do_compression
        self.max_pool_size = max_pool_size
        self.retry_mode = retry_mode
        self.max_retry_attempts = max_retry_attempts

    async def start(self) -> BaseCheckpointSaver:
        from langgraph_checkpoint_aws import DynamoDBSaver
        from botocore.config import Config

        checkpointer = DynamoDBSaver(
            table_name=self.table_name,
            session=self.session,
            ttl_seconds=self.ttl_seconds,
            enable_checkpoint_compression=self.do_compression,
            boto_config=Config(
                retries={
                    "mode": self.retry_mode,
                    "max_attempts": self.max_retry_attempts,
                },
                max_pool_connections=self.max_pool_size,
            ),
        )
        return checkpointer

    async def stop(self) -> None:
        pass


def create_checkpointer_provider(
    backend: str,
    sqlite_config,
    postgres_config,
    dynamodb_config,
) -> CheckpointerProvider:
    """Factory: returns the correct provider based on configured backend.

    backend: Literal['sqlite', 'postgres', 'dynamodb']

    For sqlite: need db_path

    For postgres: need max_pool_size, dsn
    """
    match backend:
        case "sqlite":
            return SqliteCheckpointerProvider(db_path=sqlite_config.db_path)
        case "postgres":
            return PostgresCheckpointerProvider(
                conn_string=postgres_config.dsn,
                max_size=postgres_config.max_pool_size,
            )
        case "dynamodb":
            return DynamoDBCheckpointerProvider(
                region_name=dynamodb_config.region_name,
                table_name=dynamodb_config.table_name,
                ttl_seconds=dynamodb_config.ttl_seconds,
                do_compression=dynamodb_config.do_compression,
                max_pool_size=dynamodb_config.max_pool_size,
                retry_mode=dynamodb_config.retry_mode,
                max_retry_attempts=dynamodb_config.max_retry_attempts,
            )
        case _:
            raise ValueError(
                f"Unsupported checkpointer backend: {backend!r}. "
                "Expected 'sqlite' or 'postgres'."
            )
