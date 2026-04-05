import logging
from uuid import uuid4
from typing import Annotated

from fastapi import FastAPI, status, UploadFile
from pydantic import BaseModel
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langgraph.types import Command

from src.utils.common_utils import get_global_config
from src.rag_pipeline.checkpointer import create_checkpointer_provider
from src.rag_pipeline.rag_agent import FashionAgent
from src.data_manager.session_data_table import SessionData

cfg = get_global_config()
log = logging.getLogger(__name__)
app = FastAPI()
session_data_table = SessionData(cfg)

class SessionDataResponse(BaseModel):
    session_id: str

@app.post("/new-session/", status_code=status.HTTP_201_CREATED)
async def new_session() -> SessionDataResponse:
    metadata_callback = UsageMetadataCallbackHandler()
    checkpointer_provider = create_checkpointer_provider(cfg)
    agent = FashionAgent(cfg, {"callbacks": [metadata_callback]}, checkpointer_provider)
    session_id = str(uuid4())
    config = {"configurable": {"thread_id": session_id}}
    session_data_table.add_entry(session_id=session_id, agent=agent, metadata_callback=metadata_callback)
    await agent.ainvoke({"is_chat_start": True}, config=config)
    return SessionDataResponse(session_id=session_id)

@app.post("/message/{session_id}")
async def message_from_user(
    session_id: str,
    text: Annotated[str, Form()],
    images: Annotated[list[UploadFile] | None, File()] = None
):
    pass
