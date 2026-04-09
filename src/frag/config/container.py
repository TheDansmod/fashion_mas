from contextlib import asynccontextmanager, contextmanager

from loguru import logger
from dependency_injector import containers, providers

from frag.config.app_config import ApplicationConfig
from frag.utils.model_factory import get_llm_model
from frag.utils.checkpointer import create_checkpointer_provider
from frag.utils.logger_setup import setup_logging
from frag.utils.aws_ssm_bootstrap import _bootstrap_ssm


@asynccontextmanager
async def checkpointer_connection(
    backend: str, sqlite_config, postgres_config, dynamodb_config
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

@contextmanager
def _ssm_resource():
    _bootstrap_ssm()
    yield

def _manage_config(_bootstrap_done=None):
    # the only purpose of this function is to have DI resolve the bootstrap resource before setting up the config
    return ApplicationConfig()

class Container(containers.DeclarativeContainer):

    # bootstrapping the config values from AWS, or fallback to local
    _ssm = providers.Resource(_ssm_resource)

    # general config
    config = providers.Singleton(_manage_config, _bootstrap_done=_ssm)

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
            "frag.rag_pipeline",
            "frag.utils",
        ],
    )
