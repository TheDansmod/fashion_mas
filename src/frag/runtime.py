"""Explicit composition of the long-lived resources each process needs.

Each entrypoint (Chainlit app, CLI, MCP server) opens resources once with
`open_resources(settings, role)` and passes them on explicitly. Nothing is shared
between processes (cleanup.md §3).
"""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from frag.checkpointer import open_checkpointer
from frag.data_manager.catalogue import Catalogue, open_catalogue
from frag.llm import build_llm
from frag.settings import Settings
from frag.utils.llm_tools_setup import ToolsClient, get_tools_client

Role = Literal["agent", "eval", "mcp"]


@dataclass(frozen=True)
class Resources:
    """Resources opened for one process; fields a role doesn't need are None."""

    settings: Settings
    llm: BaseChatModel | None = None
    checkpointer: BaseCheckpointSaver | None = None
    tools: ToolsClient | None = None
    catalogue: Catalogue | None = None


@asynccontextmanager
async def open_resources(settings: Settings, role: Role) -> AsyncIterator[Resources]:
    """Open what `role` needs and close everything on exit, including on error.

    - "mcp": only the catalogue (the server process serves the tools).
    - "agent": LLM, the configured checkpointer, tools (local tools also open the
      catalogue in-process).
    - "eval": like "agent" but with an in-memory checkpointer, since evaluation runs
      need no persistence and must not share a SQLite file with the app.
    """
    async with AsyncExitStack() as stack:
        catalogue = None
        if role == "mcp" or settings.tools.mode == "local":
            catalogue = await stack.enter_async_context(open_catalogue(settings))

        if role == "mcp":
            yield Resources(settings=settings, catalogue=catalogue)
        else:
            checkpointer_settings = settings.checkpointer
            if role == "eval":
                checkpointer_settings = checkpointer_settings.model_copy(
                    update={"backend": "memory"}
                )
            checkpointer = await stack.enter_async_context(
                open_checkpointer(checkpointer_settings, settings.aws_region)
            )
            yield Resources(
                settings=settings,
                llm=build_llm(settings.llm, settings.aws_region),
                checkpointer=checkpointer,
                tools=get_tools_client(settings.tools, catalogue),
                catalogue=catalogue,
            )
