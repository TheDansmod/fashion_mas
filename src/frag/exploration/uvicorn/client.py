import json, logging
from pathlib import Path
import httpx
from httpx_sse import aconnect_sse
import chainlit as cl
from frag.utils.ui_node_updates import NODE_META

log = logging.getLogger(__name__)
API_BASE = "http://localhost:8000"  # or read from env


@cl.on_chat_start
async def start_chat():
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{API_BASE}/sessions")
        resp.raise_for_status()
        session_id = resp.json()["session_id"]

    cl.user_session.set("session_id", session_id)
    await cl.Message(content="Ready! Attach images directly to your messages.").send()


@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    input_images = [
        el for el in (message.elements or []) if "image" in getattr(el, "mime", "")
    ]

    # Build multipart payload: text field + one file entry per image
    form_data = {"input_text": message.content}
    files = [
        ("images", (Path(img.path).name, open(img.path, "rb"), img.mime))
        for img in input_images
    ]

    accumulated_state: dict = {}

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with aconnect_sse(
                client,
                "POST",
                f"{API_BASE}/sessions/{session_id}/messages",
                data=form_data,
                files=files or None,
            ) as event_source:
                async for sse in event_source.aiter_sse():
                    if sse.data == "[DONE]":
                        break
                    payload = json.loads(sse.data)

                    if "error" in payload:
                        await cl.Message(
                            content=f"An error occurred: {payload['error']}"
                        ).send()
                        return

                    node_name: str = payload["node"]
                    update: dict = payload["update"]
                    accumulated_state.update(update)

                    if node_name in NODE_META:
                        label, summary_fn = NODE_META[node_name]
                        async with cl.Step(name=label) as step:
                            step.output = summary_fn(update)
                    else:
                        async with cl.Step(name=f"⚙️ {node_name}") as step:
                            step.output = f"unknown node: {node_name}"
    except Exception as exc:
        log.exception("Streaming failed")
        await cl.Message(content=f"An error occurred: {exc}").send()
        return
    finally:
        for _, (_, fobj, _) in files or []:
            fobj.close()

    url_paths = accumulated_state.get("recommended_clothes_image_paths", [])
    expl = accumulated_state.get("recommended_clothes_explanation", "")

    if url_paths and expl:
        output_images = []
        # Fetch each image from the API server and hand to Chainlit
        async with httpx.AsyncClient() as client:
            for idx, url_path in enumerate(url_paths):
                resp = await client.get(f"{API_BASE}{url_path}")
                resp.raise_for_status()
                # Write to a local temp file Chainlit can read
                local = Path(f"/tmp/cl_img_{session_id}_{idx}.jpg")
                local.write_bytes(resp.content)
                output_images.append(
                    cl.Image(path=str(local), name=f"image {idx + 1}", display="inline")
                )
        await cl.Message(content=expl, elements=output_images).send()
    else:
        await cl.Message(
            content="No recommendations could be found for your request."
        ).send()


@cl.on_chat_end
async def end_chat():
    session_id = cl.user_session.get("session_id")
    if session_id:
        async with httpx.AsyncClient() as client:
            await client.delete(f"{API_BASE}/sessions/{session_id}")
