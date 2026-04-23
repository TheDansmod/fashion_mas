from contextlib import asynccontextmanager

import boto3
from botocore.config import Config
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

# this allows one to create a flag that can either start or not start the checkpointer
# this is useful for mcp server - when you don't want to setup the checkpointer
@asynccontextmanager
async def _conditional_checkpointer(use_checkpointer, backend, sqlite_config, postgres_config, dynamodb_config):
    if use_checkpointer:
        async with checkpointer_connection(backend, sqlite_config, postgres_config, dynamodb_config) as cp:
            yield cp
    else:
        yield None

@asynccontextmanager
async def manage_logging(log_cfg, for_mcp_server):
    setup_logging(log_cfg, for_mcp_server)
    yield
    await logger.complete()

@asynccontextmanager
async def manage_s3_connection(max_pool_size, retry_mode, max_retry_attempts, setup_connection):
    if setup_connection:
        s3_client = boto3.client(
            "s3",
            config=Config(
                max_pool_connections=max_pool_size,
                retries={
                    "max_attempts": max_retry_attempts,
                    "mode": retry_mode
                }
            )
        )
        yield s3_client
    else:
        yield None

class Container(containers.DeclarativeContainer):
    # general config
    config = providers.Singleton(ApplicationConfig)

    # llm model
    llm_model = providers.Singleton(get_llm_model, cfg=config.provided)

    # checkpointer
    # by default, the checkpointer is created
    use_checkpointer = providers.Object(True)
    checkpointer = providers.Resource(
        _conditional_checkpointer,
        use_checkpointer=use_checkpointer,
        backend=config.provided.orchestration.checkpointer.backend,
        sqlite_config=config.provided.orchestration.checkpointer.sqlite,
        postgres_config=config.provided.orchestration.checkpointer.postgres,
        dynamodb_config=config.provided.orchestration.checkpointer.dynamodb,
    )

    # logging
    # by default, the logging is setup for rag_agent, not mcp_server
    mcp_server_logger = providers.Object(False)
    _logger = providers.Resource(
        manage_logging,
        log_cfg=config.provided.logs,
        for_mcp_server=mcp_server_logger,
    )

    # s3 connection
    # by default, we don't setup an s3 connection
    setup_s3_connection = providers.Object(False)
    s3_client = providers.Resource(
        manage_s3_connection,
        max_pool_size=config.provided.orchestration.mcp.max_pool_size,
        retry_mode=config.provided.orchestration.mcp.retry_mode,
        max_retry_attempts=config.provided.orchestration.mcp.max_retry_attempts,
        setup_connection=setup_s3_connection,
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
