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
async def _conditional_checkpointer(backend, sqlite_config, postgres_config, dynamodb_config, in_mcp_server_process):
    if not in_mcp_server_process:
        async with checkpointer_connection(backend, sqlite_config, postgres_config, dynamodb_config) as cp:
            yield cp
    else:
        yield None

@asynccontextmanager
async def manage_logging(log_cfg, in_mcp_server_process):
    setup_logging(log_cfg, in_mcp_server_process)
    yield
    await logger.complete()

@asynccontextmanager
async def manage_s3_connection(max_pool_size, retry_mode, max_retry_attempts, in_mcp_server_process, use_mcp_server):
    # setup an s3 connection if you are in mcp_server_process or if you are not using mcp server
    if in_mcp_server_process or not use_mcp_server:
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
async def manage_metadata_lookup(s3_client, bucket_name, metadata_key, index_key, in_mcp_server_process, use_mcp_server):
    # setup an s3 connection if you are in mcp_server_process or if you are not using mcp server
    if in_mcp_server_process or not use_mcp_server:
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
async def manage_qdrant_connection(url, prefer_grpc, collection_name, category_key, image_vectors_name, index_key, in_mcp_server_process, use_mcp_server):
    # setup an qdrant connection if you are in mcp_server_process or if you are not using mcp server
    if in_mcp_server_process or not use_mcp_server:
        qdrant_connector = QdrantConnector(url, prefer_grpc, collection_name, category_key, image_vectors_name, index_key)
        await qdrant_connector.validate()
        yield qdrant_connector
    else:
        yield None

@asynccontextmanager
async def manage_embedder(embedding_model, embedding_batch_size, in_mcp_server_process, use_mcp_server):
    # setup an embedder if you are in mcp_server_process or if you are not using mcp server
    if in_mcp_server_process or not use_mcp_server:
        embedder = FashionSigLIPEmbedding(embedding_model, embedding_batch_size)
        yield embedder
    else:
        yield None

@asynccontextmanager
async def manage_tools_client(use_mcp_server, connector, embedder, product_categories, llm_tool_names, db_tool_name, mcp_client_transport, mcp_url):
    tools_client = get_tools_client(use_mcp_server, connector, embedder, product_categories, llm_tool_names, db_tool_name, mcp_client_transport, mcp_url)
    yield tools_client

class Container(containers.DeclarativeContainer):
    # general config
    config = providers.Singleton(ApplicationConfig)

    # llm model
    llm_model = providers.Singleton(get_llm_model, cfg=config.provided)

    # this flag tells us if we are in the mcp server process (when we are setting up the mcp server separately)
    in_mcp_server_process = providers.Object(False)

    # checkpointer
    # by default, the checkpointer is created
    checkpointer = providers.Resource(
        _conditional_checkpointer,
        backend=config.provided.orchestration.checkpointer.backend,
        sqlite_config=config.provided.orchestration.checkpointer.sqlite,
        postgres_config=config.provided.orchestration.checkpointer.postgres,
        dynamodb_config=config.provided.orchestration.checkpointer.dynamodb,
        in_mcp_server_process=in_mcp_server_process,
    )

    # logging
    # by default, the logging is setup for rag_agent, not mcp_server
    _logger = providers.Resource(
        manage_logging,
        log_cfg=config.provided.logs,
        in_mcp_server_process=in_mcp_server_process,
    )

    # s3 connection
    s3_client = providers.Resource(
        manage_s3_connection,
        max_pool_size=config.provided.orchestration.mcp.max_pool_size,
        retry_mode=config.provided.orchestration.mcp.retry_mode,
        max_retry_attempts=config.provided.orchestration.mcp.max_retry_attempts,
        in_mcp_server_process=in_mcp_server_process,
        use_mcp_server=config.provided.env.use_mcp_server,
    )

    # metadata lookup - also relies on the setup_s3_connection to be setup
    metadata_lookup = providers.Resource(
        manage_metadata_lookup,
        s3_client=s3_client,
        bucket_name=config.provided.data.aws_fashion_gen.s3_bucket_name,
        metadata_key=config.provided.data.aws_fashion_gen.fashion_gen_metadata_s3_key,
        index_key=config.provided.data.fashion_gen.index_key,
        in_mcp_server_process=in_mcp_server_process,
        use_mcp_server=config.provided.env.use_mcp_server,
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
        in_mcp_server_process=in_mcp_server_process,
        use_mcp_server=config.provided.env.use_mcp_server,
    )

    # embedder
    multimodal_embedder = providers.Resource(
        manage_embedder,
        embedding_model=config.provided.data.vector_db.embedding_model,
        embedding_batch_size=config.provided.data.data_processing.embedding_batch_size,
        in_mcp_server_process=in_mcp_server_process,
        use_mcp_server=config.provided.env.use_mcp_server,
    )

    # tools client
    tools_client = providers.Resource(
        manage_tools_client,
        use_mcp_server=config.provided.env.use_mcp_server,
		connector=qdrant_connector,
		embedder=multimodal_embedder,
		product_categories=config.provided.data.fashion_gen.product_categories,
		llm_tool_names=config.provided.orchestration.mcp.llm_tool_names,
		db_tool_name=config.provided.orchestration.mcp.db_tool_name,
		mcp_client_transport=config.provided.orchestration.mcp.client_transport_method,
		mcp_url=config.provided.orchestration.mcp.url,
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
