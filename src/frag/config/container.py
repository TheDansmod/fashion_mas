from contextlib import asynccontextmanager

from loguru import logger
from dotenv import load_dotenv
from dependency_injector import containers, providers

from frag.config.app_config import ApplicationConfig
from frag.utils.model_factory import get_llm_model
from frag.utils.checkpointer import create_checkpointer_provider
from frag.utils.logger_setup import setup_logging

# this is at the top level to ensure it runs only once,
# and is inside the container.py file since that ensures it is run early in any entrypoint
load_dotenv()


@asynccontextmanager
async def checkpointer_connection(
    backend: str, sqlite_config, postgres_config, dynamodb_config,
):
    checkpointer_provider = create_checkpointer_provider(
        backend,
        sqlite_config,
        postgres_config,
        dynamodb_config,
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
        sqlite_config=config.provided.orchestration.checkpointer.sqlite,
        postgres_config=config.provided.orchestration.checkpointer.postgres,
        dynamodb_config=config.provided.orchestration.checkpointer.dynamodb,
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
            "frag.mcp_server",
            "frag.rag_pipeline",
            "frag.utils",
        ],
    )
