import logging

log = logging.getLogger(__name__)

import hydra
import asyncio
from fastmcp import Client
from langchain.agents import create_agent
from src.utils.common_utils import track_token_use, get_rate_limiter
from langchain_mcp_adapters.client import MultiServerMCPClient  
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

async def stringify_interceptor(request: MCPToolCallRequest, handler):
    log.debug(f"In the interceptor.\n{request=}\n{handler=}")
    result = await handler(request)
    if isinstance(result, list):
        return "\n".join(block.get("text", "") for block in result if isinstance(block, dict) and block.get("type") == "text")
    return str(result)

@track_token_use
async def ship_order(cfg, callback_config):
    # model setup
    provider = hydra.utils.instantiate(cfg.models.vlm_agent)
    model = provider(
        model=cfg.models.vlm_agent.name,
        temperature=cfg.models.vlm_agent.temp,
        rate_limiter=get_rate_limiter(cfg),
    )
    log.debug('setup model')
    # client setup
    client = MultiServerMCPClient(
        {
            "shipping_server": {
                "transport": "streamable_http",
                "url": "http://localhost:8000/mcp",
            },
        },
        tool_interceptors=[stringify_interceptor],
    )
    log.debug('setup mcp client')
    # agent setup
    tools = await client.get_tools()

    agent = create_agent(
        model=model,
        tools=tools,
        debug=True,
    )
    log.debug('setup mcp server')
    # prompt setup
    prompt = await client.get_prompt("shipping_server", "fulfill_order", arguments={"order_id": "9942"})
    log.debug('setup prompt, invoking agent')
    # when doing invocation the input must be a dictionary with messages key
    response = await agent.ainvoke({"messages": prompt[0].content}, config=callback_config)
    log.debug(response)
