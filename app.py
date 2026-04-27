"""This is the starting point for the project."""

from loguru import logger as log

# START MONKEYPATCH
# monkey patch since in the redirect state parameter chainlit uses characters that are invalid for aws cognito
# https://github.com/Chainlit/chainlit/issues/2707
# https://github.com/Chainlit/chainlit/issues/972
from chainlit import secret as _cl_secret  # the secret.py file in the .venv folder
import string

_cl_secret.chars = string.ascii_letters + string.digits + "-_"
# END MONKEYPATCH

from chainlit.data.storage_clients.s3 import S3StorageClient

# START S3 CLOSE MONKEYPATCH
# Fixes Chainlit bug where it tries to await a synchronous boto3 close method
from chainlit.data.storage_clients.s3 import S3StorageClient

async def patched_s3_close(self):
    try:
        # Call the synchronous close without awaiting
        if hasattr(self, 'client') and self.client:
            self.client.close()
    except Exception as e:
        log.warning(f"Failed to close S3 client: {e}")

S3StorageClient.close = patched_s3_close
# END S3 CLOSE MONKEYPATCH

from typing import Optional
from pathlib import Path
import shutil

import chainlit as cl
import chainlit.data as cl_data
from chainlit.data.dynamodb import DynamoDBDataLayer
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langgraph.types import Command

# dependency wiring must be done before frag imports
from frag.config.container import Container

container = Container()

from frag.rag_pipeline.rag_agent import FashionAgent
from frag.utils.common_utils import update_token_use
from frag.utils.ui_node_updates import NODE_META

cfg = Container.config.provided


@cl.data_layer
def get_data_layer():
    # chainlit first dynamically launches the app before registering with sys.modules registry.
    # this interferes with the DI using sys.modules to perform its wiring.
    # thus, instead of doing inject, we directly use container.config()
    app_config = container.config()
    s3_bucket_name = app_config.data.chainlit_persistence.s3_bucket_name
    s3_region = app_config.data.chainlit_persistence.s3_region
    dynamodb_table_name = app_config.data.chainlit_persistence.dynamodb_table_name

    storage_client = S3StorageClient(bucket=s3_bucket_name, region_name=s3_region)
    return DynamoDBDataLayer(
        table_name=dynamodb_table_name, storage_provider=storage_client
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
    await container.init_resources()


@cl.on_app_shutdown
async def shutdown():
    await container.shutdown_resources()

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
    agent = FashionAgent({"callbacks": [metadata_callback]})
    cl.user_session.set("metadata_callback", metadata_callback)
    cl.user_session.set("agent", agent)

    await agent.compile_graph()

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
    agent = FashionAgent({"callbacks": [metadata_callback]})
    cl.user_session.set("metadata_callback", metadata_callback)
    cl.user_session.set("agent", agent)

    await agent.compile_graph()

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
        update_token_use(metadata_callback.usage_metadata)
    else:
        log.warning("metadata callback not set; skipping token update.")
    log.info("updated token use")

    # don't delete the tempdir folder since it might be in use by multiple sessions
