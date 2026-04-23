"""Shim for launching the mcp server."""
import asyncio

from loguru import logger as log
from dependency_injector.wiring import inject, Provide as PV
from dependency_injector import providers

# dependency wiring must be done before frag imports
from frag.config.container import Container

container = Container()
# we are in the mcp server process
container.in_mcp_server_process.override(providers.Object(True))

from frag.mcp_server.server import main as server_main

@inject
async def main(use_mcp_server: bool = PV[Container.config.provided.env.use_mcp_server]):
    if not use_mcp_server:
        raise ValueError("Did you mean to run the MCP server? The USE_MCP_SERVER environment variable is set to False!")
    try:
        await container.init_resources()
        await server_main()
    except Exception as e:
        log.exception("Some Exception in mcp_server.py main function.")
    finally:
        await container.shutdown_resources()

if __name__ == '__main__':
    asyncio.run(main())
