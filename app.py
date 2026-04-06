"""This is the starting point for the project."""

from pathlib import Path
import shutil

import chainlit as cl
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langgraph.types import Command
from loguru import logger as log
from dependency_injector.wiring import inject, Provide as PV

# dependency wiring must be done before src imports
from src.config.container import Container
container = Container()

from src.rag_pipeline.rag_agent import FashionAgent
from src.utils.common_utils import update_token_use
from src.utils.ui_node_updates import NODE_META

cfg = Container.config.provided

@cl.on_app_startup
async def startup():
    await container.init_resources()

@cl.on_app_shutdown
async def shutdown():
    await container.shutdown_resources()

@cl.on_chat_start
async def start_chat():
    log.info('in start')
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

    await agent.ainvoke({"is_chat_start": True}, config=config)

@cl.on_message
async def on_message(message: cl.Message):
    config = cl.user_session.get("config")
    agent = cl.user_session.get("agent")

    input_images = [el for el in (message.elements or []) if "image" in getattr(el, "mime", "")]
    resume_payload = {
            "input_images_path": [img.path for img in input_images],
            "input_text": message.content,
    }
    accumulated_state = {}
    try:
        async for chunk in agent.astream(Command(resume=resume_payload), config=config):
            for node_name, update in chunk.items():
                if node_name == '__interrupt__':
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
        await cl.Message(content=f"An error occured {e}").send()

    paths = accumulated_state.get("recommended_clothes_image_paths", [])
    expl = accumulated_state.get("recommended_clothes_explanation", "")

    if paths and expl:
        output_images = []
        for idx, path in enumerate(paths):
            output_images.append(cl.Image(path=path, name=f'image {idx+1}', display='inline'))
        await cl.Message(content=expl, elements=output_images).send()
    else:
        await cl.Message(content="No recommendations could be found for your request.").send()

@cl.on_chat_end
@inject
async def end_chat(temp_dir_path = PV[cfg.orchestration.temporary_images_folder]):
    log.info('in end chat')

    metadata_callback = cl.user_session.get("metadata_callback")
    update_token_use(metadata_callback.usage_metadata)
    log.info("updated token use")

    temp_dir_path = Path(temp_dir_path)
    if temp_dir_path.exists() and temp_dir_path.is_dir():
        shutil.rmtree(temp_dir_path, ignore_errors=True)
