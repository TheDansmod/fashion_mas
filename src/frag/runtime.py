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

from frag.catalogue.service import Catalogue, open_catalogue
from frag.checkpointer import open_checkpointer
from frag.llm import build_llm
from frag.settings import Settings
from frag.tools.base import ToolsClient
from frag.tools.local import LocalToolsClient
from frag.tools.mcp_client import MCPToolsClient

Role = Literal["agent", "eval", "mcp"]


@dataclass(frozen=True)
class Resources:
    """Resources opened for one process; fields a role doesn't need are None."""

    settings: Settings
    llm: BaseChatModel | None = None
    # role "eval" only: the judge (the agent's `llm` unless evaluation.judge_llm is set)
    judge_llm: BaseChatModel | None = None
    checkpointer: BaseCheckpointSaver | None = None
    tools: ToolsClient | None = None
    catalogue: Catalogue | None = None


def build_tools_client(settings: Settings, catalogue: Catalogue | None) -> ToolsClient:
    """Return MCP-backed or in-process tools, depending on `settings.tools.mode`."""
    if settings.tools.mode == "mcp":
        return MCPToolsClient(settings.tools.mcp_url)
    if catalogue is None:
        msg = "local tools need an open catalogue"
        raise ValueError(msg)
    return LocalToolsClient(catalogue)


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
            llm = build_llm(settings.llm, settings.aws_region)
            judge_llm = None
            if role == "eval":
                judge_settings = settings.evaluation.judge_llm
                judge_llm = (
                    build_llm(judge_settings, settings.aws_region)
                    if judge_settings
                    else llm
                )
            yield Resources(
                settings=settings,
                llm=llm,
                judge_llm=judge_llm,
                checkpointer=checkpointer,
                tools=build_tools_client(settings, catalogue),
                catalogue=catalogue,
            )
