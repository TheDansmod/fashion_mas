from contextlib import asynccontextmanager

import h5py
import s3fs
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
async def s3fs_file_handler(bucket, s3_key, block_size, setup_connection):
    # s3fs is internally async but the open call does not support async
    if setup_connection:
        fs = s3fs.S3FileSystem(anon=False, default_block_size=block_size)
        with fs.open(f"s3://{bucket}/{s3_key}", "rb") as fh:
            yield fh
    else:
        yield None

@asynccontextmanager
async def h5py_file_handler(s3_fh, setup_connection):
    # h5py is not async safe and we can't use async with on it
    if setup_connection:
        with h5py.File(s3_fh, "r") as f:
            yield f
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
    # be default, the logging is setup for rag_agent, not mcp_server
    mcp_server_logger = providers.Object(False)
    _logger = providers.Resource(
        manage_logging,
        log_cfg=config.provided.logs,
        for_mcp_server=mcp_server_logger,
    )

    # s3fs and h5py
    # by default we don't setup s3fs and hdf5
    setup_dataset_connection = providers.Object(False)
    s3_file_handle = providers.Resource(
        s3fs_file_handler,
        bucket=config.provided.data.aws_fashion_gen.s3_bucket_name,
        s3_key=config.provided.data.aws_fashion_gen.dataset_object_name,
        block_size=config.provided.data.aws_fashion_gen.s3fs_block_size,
        setup_connection=setup_dataset_connection,
    )

    h5_file = providers.Resource(
        h5py_file_handler,
        s3_fh=s3_file_handle,
        setup_connection=setup_dataset_connection,
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
