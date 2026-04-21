"""Shim for launching the mcp server."""
import asyncio

from loguru import logger as log
from dependency_injector.wiring import inject, Provide as PV

# dependency wiring must be done before frag imports
from frag.config.container import Container

container = Container()

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
