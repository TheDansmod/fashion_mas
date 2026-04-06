from contextlib import contextmanager

from dependency_injector import containers, providers

from src.config.app_config import ApplicationConfig
from src.utils.model_factory import get_llm_model
from src.utils.checkpointer import create_checkpointer_provider

@contextmanager
async def checkpointer_connection(backend: str, sqlite_db_path: str, postgres_dsn: str, postgres_max_pool_size: int):
    checkpointer_provider = create_checkpointer_provider(backend, sqlite_db_path, postgres_dsn, postgres_max_pool_size)
    checkpointer = await checkpointer_provider.start()
    try:
        yield checkpointer
    finally:
        await checkpointer.stop()

class Container(containers.DeclarativeContainer):
    # general config
    config = providers.Singleton(ApplicationConfig)

    # llm model
    llm_model = providers.Singleton(get_llm_model, cfg=config.provided)

    # checkpointer
    checkpointer = providers.Resource(
        create_checkpointer_provider,
        backend = config.provided.orchestration.checkpointer.backend,
        sqlite_db_path = config.provided.orchestration.checkpointer.sqlite.db_path,
        postgres_dsn = config.provided.orchestration.checkpointer.postgres.dsn,
        postgres_max_pool_size = config.provided.orchestration.checkpointer.postgres.max_pool_size,
    )

    wiring_config = containers.WiringConfiguration(
        packages=[
            "src.data_manager",
            "src.evaluation",
            "src.exploration",
            "src.rag_pipeline",
            "src.utils",
        ],
    )
