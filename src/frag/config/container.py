from contextlib import asynccontextmanager, contextmanager

from loguru import logger
from dependency_injector import containers, providers

from frag.config.app_config import ApplicationConfig
from frag.utils.model_factory import get_llm_model
from frag.utils.checkpointer import create_checkpointer_provider
from frag.utils.logger_setup import setup_logging


@asynccontextmanager
async def checkpointer_connection(
    backend: str, sqlite_db_path: str, postgres_dsn: str, postgres_max_pool_size: int
):
    checkpointer_provider = create_checkpointer_provider(
        backend, sqlite_db_path, postgres_dsn, postgres_max_pool_size
    )
    checkpointer = await checkpointer_provider.start()
    try:
        yield checkpointer
    finally:
        await checkpointer_provider.stop()


@asynccontextmanager
async def manage_logging(log_cfg):
    setup_logging(log_cfg)
    yield
    await logger.complete()


class Container(containers.DeclarativeContainer):
    # general config
    config = providers.Singleton(ApplicationConfig)

    # llm model
    llm_model = providers.Singleton(get_llm_model, cfg=config.provided)

    # checkpointer
    checkpointer = providers.Resource(
        checkpointer_connection,
        backend=config.provided.orchestration.checkpointer.backend,
        sqlite_db_path=config.provided.orchestration.checkpointer.sqlite.db_path,
        postgres_dsn=config.provided.orchestration.checkpointer.postgres.dsn,
        postgres_max_pool_size=config.provided.orchestration.checkpointer.postgres.max_pool_size,
    )

    # logging
    _logger = providers.Resource(
        manage_logging,
        log_cfg=config.provided.logs,
    )

    wiring_config = containers.WiringConfiguration(
        packages=[
            "frag.data_manager",
            "frag.evaluation",
            "frag.exploration",
            "frag.rag_pipeline",
            "frag.utils",
        ],
    )
