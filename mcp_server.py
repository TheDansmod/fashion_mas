"""Shim for launching the mcp server."""
import asyncio

from loguru import logger as log
from dependency_injector.wiring import inject, Provide as PV
from dependency_injector import providers

# dependency wiring must be done before frag imports
from frag.config.container import Container

container = Container()
# we don't want to setup checkpointer when using mcp server
container.use_checkpointer.override(providers.Object(False))
# we want to log to a different file than the main process
container.mcp_server_logger.override(providers.Object(True))

from frag.mcp_server.server import main as server_main

async def main():
    try:
        await container.init_resources()
        await server_main()
    except Exception as e:
        log.exception("Some Exception in mcp_server.py main function.")
    finally:
        await container.shutdown_resources()

if __name__ == '__main__':
    asyncio.run(main())
