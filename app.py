"""Chainlit entrypoint: `chainlit run app.py`."""

import shutil
import string
from contextlib import AsyncExitStack
from pathlib import Path

import chainlit as cl
from chainlit import secret as _cl_secret
from chainlit.data.dynamodb import DynamoDBDataLayer
from chainlit.data.storage_clients.s3 import S3StorageClient
from dotenv import load_dotenv
from loguru import logger as log

from frag.agent.graph import FashionAgent
from frag.runtime import Resources, open_resources
from frag.settings import get_settings
from frag.tokens import track_tokens
from frag.ui.node_updates import NODE_META
from frag.utils.logger_setup import setup_logging

# make .env visible to libraries that read os.environ themselves (LangSmith, Chainlit
# OAuth, boto3, ...); idempotent and never overrides variables that are already set
load_dotenv()

# START MONKEYPATCH (moves to frag/ui/chainlit_patches.py in cleanup.md Phase 8)
# the redirect `state` parameter chainlit generates uses characters that AWS Cognito
# rejects; restrict the alphabet. Applied at import, before any request is served.
# https://github.com/Chainlit/chainlit/issues/2707
# https://github.com/Chainlit/chainlit/issues/972
_cl_secret.chars = string.ascii_letters + string.digits + "-_"


# Chainlit awaits the synchronous boto3 close(); call it without awaiting instead
async def patched_s3_close(self):
    try:
        if hasattr(self, "client") and self.client:
            self.client.close()
    except Exception as e:
        log.warning(f"Failed to close S3 client: {e}")


S3StorageClient.close = patched_s3_close
# END MONKEYPATCH


class _AppState:
    """Process-wide resources, opened at app startup and closed at shutdown."""

    exit_stack: AsyncExitStack | None = None
    resources: Resources | None = None


_app = _AppState()


@cl.data_layer
def get_data_layer():
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


def _start_session(thread_id: str) -> None:
    """One agent per chat session, sharing the process-wide resources."""
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


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: dict[str, str],
    default_user: cl.User,
) -> cl.User | None:
    if provider_id == "aws-cognito":
        sub = raw_user_data.get("sub")  # this is mandatory
        email = raw_user_data.get("email")
        username = raw_user_data.get("username")
        # if no sub, fail auth
        if not sub:
            log.error("Cognito missing 'sub' claim.")
            return None
        # replace email with sub as the identifier
        return cl.User(
            identifier=sub,  # stable, unique per user even if they change email etc
            metadata={
                "email": email,
                "provider": provider_id,
                "username": username,
                **default_user.metadata,
            },
        )
    return default_user


@cl.on_app_startup
async def startup():
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
async def shutdown():
    if _app.exit_stack is not None:
        await _app.exit_stack.aclose()
    await log.complete()

    # .files/ folder is created by chainlit as a temp staging area during execution
    temp_dir_path = Path(".files")
    if temp_dir_path.exists() and temp_dir_path.is_dir():
        shutil.rmtree(temp_dir_path, ignore_errors=True)
        log.info("removed .files folder")


@cl.on_chat_start
async def start_chat():
    _start_session(cl.context.session.thread_id)
    await cl.Message(
        content="Ask for recommendations. You can attach images to your messages!"
    ).send()


@cl.on_chat_resume
async def chat_resume(thread):
    log.info("resuming thread {}", thread["id"])
    _start_session(thread["id"])


@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    thread_id = cl.user_session.get("thread_id")
    image_paths = [
        el.path
        for el in (message.elements or [])
        if "image" in (getattr(el, "mime", None) or "")
    ]
    accumulated_state = {}
    csv_path = get_settings().token_usage_csv
    try:
        with track_tokens(csv_path, "agent", thread_id) as usage:
            async for chunk in agent.stream_turn(
                message.content, image_paths, thread_id, callbacks=[usage]
            ):
                for node_name, update in chunk.items():
                    accumulated_state.update(update or {})
                    label, summary_fn = NODE_META.get(
                        node_name, (f"⚙️ {node_name}", lambda _u: "")
                    )
                    async with cl.Step(name=label) as step:
                        step.output = summary_fn(update or {})
    except Exception as e:
        log.exception("Agent failed during streaming.")
        await cl.Message(content=f"An error occurred {e}").send()
        return

    paths = accumulated_state.get("recommended_clothes_image_paths", [])
    expl = accumulated_state.get("recommended_clothes_explanation", "")

    if paths and expl:
        output_images = []
        for idx, path in enumerate(paths):
            output_images.append(
                cl.Image(path=path, name=f"image {idx + 1}", display="inline")
            )
        await cl.Message(content=expl, elements=output_images).send()
    else:
        await cl.Message(
            content="No recommendations could be found for your request."
        ).send()


@cl.on_chat_end
async def end_chat():
    # token usage is recorded per message; the temp image folder is shared by all
    # sessions, so it is not deleted here
    log.info("chat ended")
