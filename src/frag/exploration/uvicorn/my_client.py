import logging

import chainlit as cl

from frag.utils.common_utils import get_global_config

cfg = get_global_config()

log = logging.getLogger(__name__)

@cl.on_chat_start
async def start_chat():
    async with httpx.AsyncClient() as client:
        response = await client.get()
