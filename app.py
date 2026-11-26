"""Chainlit entrypoint: `chainlit run app.py`."""

import shutil
import string
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Optional

import chainlit as cl
from chainlit import secret as _cl_secret
from chainlit.data.dynamodb import DynamoDBDataLayer
from chainlit.data.storage_clients.s3 import S3StorageClient
from dotenv import load_dotenv
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langgraph.types import Command
from loguru import logger as log

from frag.rag_pipeline.rag_agent import FashionAgent
from frag.runtime import Resources, open_resources
from frag.settings import get_settings
from frag.tokens import append_usage_csv
from frag.utils.logger_setup import setup_logging
from frag.utils.ui_node_updates import NODE_META

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


def _new_agent(metadata_callback: UsageMetadataCallbackHandler) -> FashionAgent:
    """One agent per chat session, sharing the process-wide resources."""
    resources = _app.resources
    return FashionAgent(
        llm=resources.llm,
        tools=resources.tools,
        checkpointer=resources.checkpointer,
        settings=resources.settings.agent,
        callbacks=[metadata_callback],
    )


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    if provider_id == "aws-cognito":
        sub = raw_user_data.get("sub")  # this is mandatory
        email = raw_user_data.get("email", None)
        username = raw_user_data.get("username", None)
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
    log.info("in start")
    # we want to have a separate callback handler and fashion agent object per connection
    metadata_callback = UsageMetadataCallbackHandler()
    agent = _new_agent(metadata_callback)
    cl.user_session.set("metadata_callback", metadata_callback)
    cl.user_session.set("agent", agent)

    # we use thread_id since that is stable across reconnects - the session.id is just the websocket id which will change on refresh / re-connect - when we switch to adding users - we should use user_id:conversation_id
    config = {"configurable": {"thread_id": cl.context.session.thread_id}}
    cl.user_session.set("config", config)

    await cl.Message(
        content="Starting chat loop... You can attach images directly to your messages!"
    ).send()

    try:
        await agent.ainvoke({"is_chat_start": True}, config=config)
    except Exception as e:
        log.exception("failed to start graph.")
        await cl.Message(content="Failed to start the Agent, please try again.").send()

@cl.on_chat_resume
async def chat_resume(thread):
    log.info("In chat resume")

    metadata_callback = UsageMetadataCallbackHandler()
    agent = _new_agent(metadata_callback)
    cl.user_session.set("metadata_callback", metadata_callback)
    cl.user_session.set("agent", agent)

    # we use thread_id since that is stable across reconnects - the session.id is just the websocket id which will change on refresh / re-connect - when we switch to adding users - we should use user_id:conversation_id
    config = {"configurable": {"thread_id": thread["id"]}}
    cl.user_session.set("config", config)

@cl.on_message
async def on_message(message: cl.Message):
    config = cl.user_session.get("config")
    agent = cl.user_session.get("agent")

    input_images = [
        el for el in (message.elements or []) if "image" in getattr(el, "mime", "")
    ]
    resume_payload = {
        "input_images_path": [img.path for img in input_images],
        "input_text": message.content,
    }
    accumulated_state = {}
    try:
        async for chunk in agent.astream(Command(resume=resume_payload), config=config):
            for node_name, update in chunk.items():
                if node_name == "__interrupt__":
                    continue
                accumulated_state.update(update)
                if node_name in NODE_META:
                    label, summary_fn = NODE_META[node_name]
                    async with cl.Step(name=label) as step:
                        step.output = summary_fn(update)
                else:
                    async with cl.Step(name=f"⚙️ {node_name}") as step:
                        step.output = f"unknown node {node_name}"
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
    log.info("in end chat")

    metadata_callback = cl.user_session.get("metadata_callback")
    if metadata_callback is not None:
        csv_path = get_settings().token_usage_csv
        if csv_path:
            append_usage_csv(csv_path, metadata_callback.usage_metadata)
    else:
        log.warning("metadata callback not set; skipping token update.")
    log.info("updated token use")

    # don't delete the tempdir folder since it might be in use by multiple sessions
