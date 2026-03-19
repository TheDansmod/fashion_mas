"""This is the starting point for the project."""

import asyncio
import logging
from pathlib import Path
import shutil

import chainlit as cl
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langgraph.types import Command
from src.utils.ui_node_updates import NODE_META

from src.rag_pipeline.rag_agent import FashionAgent
from src.utils.common_utils import validate_hydra_config
from src.utils.common_utils import update_token_use

# The .env file should contain `HYDRA_FULL_ERROR=1` to see a full stacktrace in case
# of error.
# The .env file should also have the HF_TOKEN value from huggingface for vision model
# access - not mandatory.
# The .env file should populate langsmith endpoints like `LANGSMITH_TRACING=true`,
# `LANGSMITH_PROJECT=<project_name>`, `LANGSMITH_API_KEY`, `LANGSMITH_ENDPOINT=<eu/us>`.
# The .env file should have a key for mistral api key: `MISTRAL_API_KEY=<key>`

# having to do this since chainlit and hydra both want to start the app and I have
# decided to get chainlit to do the startup. Chainlit loads the run file as a module
# which means __name__ != '__main__', thus, the hydra initialization is global
cfg = None
if not GlobalHydra.instance().is_initialized():
    load_dotenv()
    with hydra.initialize(version_base=None, config_path="config/"):
        cfg: DictConfig = hydra.compose(
            config_name="config", overrides=[], return_hydra_config=True
        )
    hydra.core.utils.configure_log(cfg.hydra.job_logging, cfg.hydra.verbose)

    # code start - this is here since we want to execute it just once
    validate_hydra_config(cfg)

    metadata_callback = UsageMetadataCallbackHandler()
    callback_config = {"callbacks": [metadata_callback]}
    agent = FashionAgent(cfg, callback_config)
# this is to prevent a loop of watch files creating a log and hydra logging it and watchfiles logging that
logging.getLogger("watchfiles.main").setLevel(logging.WARNING)


@cl.on_chat_start
async def start_chat():
    print('in start')
    await agent.compile_graph(cfg.rag_pipeline.persistence.db_path)

    config = {"configurable": {"thread_id": cl.context.session.id}}
    cl.user_session.set("config", config)

    await cl.Message(
        content="Starting chat loop... type 'quit' to exit. You can attach images directly to your messages!"
    ).send()

    result = await agent.ainvoke({"is_chat_start": True}, config=config)

@cl.on_message
async def on_message(message: cl.Message):
    config = cl.user_session.get("config")

    images = [el for el in (message.elements or []) if "image" in getattr(el, "mime", "")]
    resume_payload = {
            "input_images_path": [img.path for img in images],
            "input_text": message.content,
    }
    accumulated_state = {}
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

    paths = accumulated_state.get("recommended_clothes_image_paths", [])
    expl = accumulated_state.get("recommended_clothes_explanation", "")

    if paths and expl:
        images = []
        for path in paths:
            images.append(cl.Image(path=path, name='image 1', display='inline'))
        await cl.Message(content=expl, elements=images).send()
    else:
        await cl.Message(content="No recommendations could be found for your request.").send()

@cl.on_chat_end
async def end_chat():
    print('in end chat')
    update_token_use(cfg, metadata_callback.usage_metadata)
    print("updated token use")
    if agent:
        print("closed connection")
        await agent.close_connection()

    temp_dir_path = Path(cfg.rag_pipeline.temporary_images_folder)
    if temp_dir_path.exists() and temp_dir_path.is_dir():
        shutil.rmtree(temp_dir_path)
