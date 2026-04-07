# frag/api/server.py
import asyncio, json, uuid, shutil, logging
from contextlib import asynccontextmanager
from pathlib import Path

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langgraph.types import Command
import hydra
from dotenv import load_dotenv

from frag.rag_pipeline.rag_agent import FashionAgent
from frag.utils.common_utils import update_token_use, validate_hydra_config

log = logging.getLogger(__name__)

# ── in-memory session store (replace with Redis for multi-worker deployments) ──
_sessions: dict[str, dict] = {}
cfg = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global cfg
    load_dotenv()
    with hydra.initialize(version_base=None, config_path="../../config/"):
        cfg = hydra.compose(config_name="config", overrides=[])
    hydra.core.utils.configure_log(cfg.hydra.job_logging, cfg.hydra.verbose)
    validate_hydra_config(cfg)
    yield
    # graceful shutdown: close every open SQLite connection
    for s in list(_sessions.values()):
        await s["agent"].close_connection()
    _sessions.clear()

app = FastAPI(lifespan=lifespan)

# ── serve agent-saved images as static files ──────────────────────────────────
# Chainlit can then build a URL like http://localhost:8000/images/<session>/<file>
# instead of using a local path it cannot reach.
TEMP_ROOT = None  # set during first request once cfg is available

def _temp_dir(session_id: str) -> Path:
    root = Path(cfg.rag_pipeline.temporary_images_folder)
    d = root / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d



@app.post("/sessions", status_code=201)
async def create_session():
    """Creates an agent instance, compiles the graph, fires the initial ainvoke
    (which immediately hits the human_node interrupt), and returns a session_id."""
    session_id = str(uuid.uuid4())
    callback = UsageMetadataCallbackHandler()
    agent = FashionAgent(cfg, {"callbacks": [callback]})
    await agent.compile_graph(cfg.rag_pipeline.persistence.db_path)
    langgraph_config = {"configurable": {"thread_id": session_id}}

    _sessions[session_id] = {
        "agent": agent,
        "config": langgraph_config,
        "callback": callback,
    }

    # Runs START → human_node → interrupt(); returns immediately at the interrupt
    await agent.ainvoke({"is_chat_start": True}, config=langgraph_config)
    return {"session_id": session_id}



@app.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: str,
    input_text: str = Form(...),
    images: list[UploadFile] = File(default=[]),
):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    agent: FashionAgent = session["agent"]
    config = session["config"]

    # ── persist uploaded images to a per-session temp folder ──────────────────
    image_paths: list[str] = []
    temp_dir = _temp_dir(session_id)
    for img in images:
        dest = temp_dir / img.filename
        async with aiofiles.open(dest, "wb") as f:
            content = await img.read()
            await f.write(content)
        image_paths.append(str(dest))

    resume_payload = {"input_text": input_text, "input_images_path": image_paths}

    # ── SSE generator: one event per LangGraph node update ────────────────────
    async def event_stream():
        try:
            async for chunk in agent.astream(
                Command(resume=resume_payload), config=config
            ):
                for node_name, update in chunk.items():
                    if node_name == "__interrupt__":
                        continue
                    # Convert image paths → server-accessible URLs before sending
                    update = _localise_image_paths(update, session_id)
                    payload = json.dumps({"node": node_name, "update": update},
                                         default=str)
                    yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            log.exception("Agent stream failed")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _localise_image_paths(update: dict, session_id: str) -> dict:
    """Replaces absolute server paths with relative URL paths the UI can fetch.
    
    Converts e.g. /tmp/imgs/abc123/shirt.jpg
             → /images/abc123/shirt.jpg
    which FastAPI serves via the StaticFiles mount below.
    """
    url_base = f"/images/{session_id}"
    temp_dir = str(_temp_dir(session_id))

    def fix(val):
        if isinstance(val, str) and val.startswith(temp_dir):
            return url_base + "/" + Path(val).name
        if isinstance(val, list):
            return [fix(v) for v in val]
        return val

    return {k: fix(v) for k, v in update.items()}




@app.delete("/sessions/{session_id}")
async def end_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions.pop(session_id)
    update_token_use(cfg, session["callback"].usage_metadata)
    await session["agent"].close_connection()

    temp_dir = _temp_dir(session_id)
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

    return {"status": "ok"}


# Mount AFTER route definitions so the route takes precedence
@app.on_event("startup")
async def _mount_static():
    root = Path(cfg.rag_pipeline.temporary_images_folder)
    root.mkdir(parents=True, exist_ok=True)
    app.mount("/images", StaticFiles(directory=str(root)), name="images")
