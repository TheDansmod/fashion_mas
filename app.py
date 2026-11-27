"""Chainlit entrypoint: `uv run chainlit run app.py`.

Sign-in is through AWS Cognito, and chat history is kept in DynamoDB and S3 (always
on, cleanup.md decision 1.5). Resources are opened once per process at app startup
and closed at shutdown (cleanup.md §3.2).
"""

import shutil
from contextlib import AsyncExitStack
from pathlib import Path

import chainlit as cl
from chainlit.data.dynamodb import DynamoDBDataLayer
from chainlit.data.storage_clients.s3 import S3StorageClient
from chainlit.types import ThreadDict
from dotenv import load_dotenv
from loguru import logger as log

from frag.agent.graph import FashionAgent
from frag.runtime import Resources, open_resources
from frag.settings import get_settings
from frag.tokens import track_tokens
from frag.ui import chainlit_patches
from frag.ui.node_updates import NODE_META
from frag.utils.logger_setup import setup_logging

# make .env visible to libraries that read os.environ themselves (LangSmith, Chainlit
# OAuth, boto3, ...); idempotent and never overrides variables that are already set
load_dotenv()
# before any request is served, since the OAuth redirect depends on one of them
chainlit_patches.apply()

# Chainlit's staging folder for uploads, removed at shutdown
CHAINLIT_FILES_DIR = Path(".files")


class _AppState:
    """Process-wide resources, opened at app startup and closed at shutdown."""

    exit_stack: AsyncExitStack | None = None
    resources: Resources | None = None


_app = _AppState()


@cl.data_layer
def get_data_layer() -> DynamoDBDataLayer:
    """Chat history in DynamoDB, with uploaded and generated files in S3."""
    # Chainlit calls this before on_app_startup; that is fine since settings are pure
    settings = get_settings()
    history = settings.chat_history
    if not (history.dynamodb_table and history.s3_bucket):
        msg = "Set CHAT_HISTORY__DYNAMODB_TABLE and CHAT_HISTORY__S3_BUCKET in .env."
        raise ValueError(msg)
    storage_client = S3StorageClient(
        bucket=history.s3_bucket, region_name=settings.aws_region
    )
    return DynamoDBDataLayer(
        table_name=history.dynamodb_table, storage_provider=storage_client
    )


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,  # noqa: ARG001  required by Chainlit's callback signature
    raw_user_data: dict[str, str],
    default_user: cl.User,
) -> cl.User | None:
    """Identify Cognito users by their `sub` claim, which never changes."""
    if provider_id != "aws-cognito":
        return default_user
    sub = raw_user_data.get("sub")
    if not sub:
        log.error("Cognito user data has no 'sub' claim; refusing the sign-in.")
        return None
    return cl.User(
        identifier=sub,
        metadata={
            "email": raw_user_data.get("email"),
            "provider": provider_id,
            "username": raw_user_data.get("username"),
            **default_user.metadata,
        },
    )


@cl.on_app_startup
async def startup() -> None:
    """Set up logging and open the agent's resources for this process."""
    settings = get_settings()
    setup_logging(
        settings.logging,
        level=settings.log_level,
        is_dev=settings.is_dev,
        service="agent",
    )
    _app.exit_stack = AsyncExitStack()
    _app.resources = await _app.exit_stack.enter_async_context(
        open_resources(settings, "agent")
    )


@cl.on_app_shutdown
async def shutdown() -> None:
    """Close the resources and remove Chainlit's upload staging folder."""
    if _app.exit_stack is not None:
        await _app.exit_stack.aclose()
    if CHAINLIT_FILES_DIR.is_dir():  # noqa: ASYNC240  once, at shutdown
        shutil.rmtree(CHAINLIT_FILES_DIR, ignore_errors=True)
        log.info("Removed {}.", CHAINLIT_FILES_DIR)
    await log.complete()


def _init_session(thread_id: str) -> None:
    """Give the chat session an agent, sharing the process-wide resources."""
    resources = _app.resources
    agent = FashionAgent(
        llm=resources.llm,
        tools=resources.tools,
        checkpointer=resources.checkpointer,
        settings=resources.settings.agent,
    )
    cl.user_session.set("agent", agent)
    # the thread id (not the websocket session id) is stable across reconnects and
    # resumes, and keys the agent's checkpointed state
    cl.user_session.set("thread_id", thread_id)


@cl.on_chat_start
async def start_chat() -> None:
    """A new conversation."""
    _init_session(cl.context.session.thread_id)
    await cl.Message(
        content="Ask for recommendations. You can attach images to your messages!"
    ).send()


@cl.on_chat_resume
async def chat_resume(thread: ThreadDict) -> None:
    """A conversation reopened from the history; its agent state is checkpointed."""
    log.info("Resuming thread {}.", thread["id"])
    _init_session(thread["id"])


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Run one agent turn, showing each node as a step, then the recommendations."""
    agent: FashionAgent = cl.user_session.get("agent")
    thread_id: str = cl.user_session.get("thread_id")
    image_paths = [
        element.path
        for element in (message.elements or [])
        if "image" in (getattr(element, "mime", None) or "")
    ]
    state = {}
    try:
        with track_tokens(get_settings().token_usage_csv, "agent", thread_id) as usage:
            async for chunk in agent.stream_turn(
                message.content, image_paths, thread_id, callbacks=[usage]
            ):
                for node_name, update in chunk.items():
                    state.update(update or {})
                    label, summarise = NODE_META.get(
                        node_name, (f"⚙️ {node_name}", lambda _u: "")
                    )
                    async with cl.Step(name=label) as step:
                        step.output = summarise(update or {})
    except Exception:  # noqa: BLE001  any failure is logged and reported to the user
        log.exception("The agent failed on thread {}.", thread_id)
        await cl.Message(
            content="Sorry, something went wrong while handling your request."
        ).send()
        return

    paths = state.get("recommended_clothes_image_paths", [])
    explanation = state.get("recommended_clothes_explanation", "")
    if not (paths and explanation):
        await cl.Message(
            content="No recommendations could be found for your request."
        ).send()
        return
    images = [
        cl.Image(path=path, name=f"image {i + 1}", display="inline")
        for i, path in enumerate(paths)
    ]
    await cl.Message(content=explanation, elements=images).send()


@cl.on_chat_end
async def end_chat() -> None:
    """Nothing to clean up per chat.

    Token usage is recorded per message, and the temporary image folder is shared by
    all sessions, so it is not deleted here.
    """
    log.info("Chat ended.")
