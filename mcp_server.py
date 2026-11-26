"""MCP server entrypoint: `uv run mcp_server.py`.

Serves the catalogue tools over streamable HTTP; the agent uses them when
TOOLS__MODE=mcp. Replaced by `frag mcp` in cleanup.md Phase 8.
"""

import asyncio

from dotenv import load_dotenv
from loguru import logger as log

from frag.mcp_server.server import main as server_main
from frag.runtime import open_resources
from frag.settings import get_settings
from frag.utils.logger_setup import setup_logging


async def main() -> None:
    """Open the catalogue and serve the MCP tools until interrupted."""
    settings = get_settings()
    setup_logging(
        settings.logging,
        level=settings.log_level,
        is_dev=settings.is_dev,
        service="mcp",
    )
    try:
        async with open_resources(settings, "mcp") as resources:
            await server_main(resources.catalogue, settings.tools)
    except Exception:  # noqa: BLE001  top-level: log any failure before exiting
        log.exception("Unhandled exception in the MCP server.")
    finally:
        await log.complete()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
