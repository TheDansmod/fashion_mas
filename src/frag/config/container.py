import io
import asyncio
from contextlib import asynccontextmanager

import boto3
from botocore.config import Config
import pyarrow.parquet as pq
from loguru import logger
from dotenv import load_dotenv
from dependency_injector import containers, providers

from frag.config.app_config import ApplicationConfig
from frag.data_manager.vector_db_read_write import QdrantConnector
from frag.data_manager.embedding import FashionSigLIPEmbedding
from frag.utils.model_factory import get_llm_model
from frag.utils.checkpointer import create_checkpointer_provider
from frag.utils.logger_setup import setup_logging
from frag.utils.llm_tools_setup import get_tools_client

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

@asynccontextmanager
async def manage_metadata_lookup(s3_client, bucket_name, metadata_key, index_key, setup_connection):
    if setup_connection:
        buffer = io.BytesIO()
        await asyncio.to_thread(s3_client.download_fileobj, bucket_name, metadata_key, buffer)
        buffer.seek(0)
        # read with pyarrow
        table = pq.read_table(buffer)
        df = table.to_pandas()
        df.set_index(index_key, inplace=True)
        metadata_lookup = df.to_dict(orient='index')
        yield metadata_lookup
    else:
        yield None

@asynccontextmanager
async def manage_qdrant_connection(url, prefer_grpc, collection_name, category_key, image_vectors_name, index_key, fgen_args):
    qdrant_connector = QdrantConnector(url, prefer_grpc, collection_name, category_key, image_vectors_name, index_key, fgen_args)
    await qdrant_connector.validate()
    yield qdrant_connector

@asynccontextmanager
async def manage_embedder(embedding_model, embedding_batch_size):
    embedder = FashionSigLIPEmbedding(embedding_model, embedding_batch_size)
    yield embedder

@asynccontextmanager
async def manage_tools_client(use_mcp_server, connector, embedder, product_categories, llm_tool_names, db_tool_name, mcp_client_transport, mcp_url, fgen_args):
    tools_client = get_tools_client(use_mcp_server, connector, embedder, product_categories, llm_tool_names, db_tool_name, mcp_client_transport, mcp_url, fgen_args)
    yield tools_client

# this is required for resolving the values
def _make_fgen_args(num_datapoints, prices_key, categories_key, descriptions_key, s3_client, bucket_name, metadata_lookup):
    return (num_datapoints, prices_key, categories_key, descriptions_key, s3_client, bucket_name, metadata_lookup)

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

    # metadata lookup - also relies on the setup_s3_connection to be setup
    metadata_lookup = providers.Resource(
        manage_metadata_lookup,
        s3_client=s3_client,
        bucket_name=config.provided.data.aws_fashion_gen.s3_bucket_name,
        metadata_key=config.provided.data.aws_fashion_gen.fashion_gen_metadata_s3_key,
        index_key=config.provided.data.fashion_gen.index_key,
        setup_connection=setup_s3_connection,
    )

    # TODO: this is being created since I am calling get_fashion_gen_data downstream from the container itself
    # and it can't be wired - probably should figure out a better solution later
    _fgen_args = providers.Singleton(
        _make_fgen_args,
        num_datapoints=config.provided.data.fashion_gen.num_datapoints,
        prices_key=config.provided.data.fashion_gen.prices_key,
        categories_key=config.provided.data.fashion_gen.categories_key,
        descriptions_key=config.provided.data.fashion_gen.descriptions_key,
        s3_client=s3_client,
        bucket_name=config.provided.data.aws_fashion_gen.s3_bucket_name,
        metadata_lookup=metadata_lookup,
    )

    # qdrant connection
    qdrant_connector = providers.Resource(
        manage_qdrant_connection,
        url=config.provided.data.vector_db.vector_store_network_path,
        prefer_grpc=config.provided.data.vector_db.prefer_grpc,
        collection_name=config.provided.data.vector_db.collection_name,
        category_key=config.provided.data.fashion_gen.categories_key,
        image_vectors_name=config.provided.data.vector_db.image_vectors_name,
        index_key=config.provided.data.fashion_gen.index_key,
        fgen_args=_fgen_args,
    )

    # embedder
    multimodal_embedder = providers.Resource(
        manage_embedder,
        embedding_model=config.provided.data.vector_db.embedding_model,
        embedding_batch_size=config.provided.data.data_processing.embedding_batch_size,
    )

    # tools client
    tools_client = providers.Resource(
        manage_tools_client,
        use_mcp_server=config.provided.orchestration.use_mcp_server,
		connector=qdrant_connector,
		embedder=multimodal_embedder,
		product_categories=config.provided.data.fashion_gen.product_categories,
		llm_tool_names=config.provided.orchestration.mcp.llm_tool_names,
		db_tool_name=config.provided.orchestration.mcp.db_tool_name,
		mcp_client_transport=config.provided.orchestration.mcp.client_transport_method,
		mcp_url=config.provided.orchestration.mcp.url,
        fgen_args=_fgen_args,
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
