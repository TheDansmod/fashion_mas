import logging

log = logging.getLogger(__name__)

import hydra
import asyncio
from langchain.agents import create_agent
from src.utils.common_utils import track_token_use, get_rate_limiter
from langchain_mcp_adapters.client import MultiServerMCPClient  
from langchain_core.tools import StructuredTool

def make_mistral_compatible(tool):
    """Wraps an MCP tool to ensure it returns a plain string."""
    def stringify_invoke(*args, **kwargs):
        tool_input = args[0] if args else kwargs
        response = tool.invoke(tool_input)
        result = []
        if isinstance(response, list):
            for block in response:
                if isinstance(block, dict):
                    result.append({k: v for k, v in block.items() if k in ["type", "text"]})
                else:
                    result.append({"type": "text", "text": str(block)})
            return result
        else:
            return str(response)

    async def stringify_ainvoke(*args, **kwargs):
        tool_input = args[0] if args else kwargs
        response = await tool.ainvoke(tool_input)
        result = []
        if isinstance(response, list):
            for block in response:
                if isinstance(block, dict):
                    result.append({k: v for k, v in block.items() if k in ["type", "text"]})
                else:
                    result.append({"type": "text", "text": str(block)})
            return result
        else:
            return str(response)

    return StructuredTool.from_function(
        func=stringify_invoke,
        coroutine=stringify_ainvoke,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )

def get_llm_model(cfg):
    provider = hydra.utils.instantiate(cfg.models.vlm_agent)
    model = provider(
        model=cfg.models.vlm_agent.name,
        temperature=cfg.models.vlm_agent.temp,
        rate_limiter=get_rate_limiter(cfg),
    )
    return model

def get_mcp_client():
    client = MultiServerMCPClient(
        {
            "shipping_server": {
                "transport": "streamable_http",
                "url": "http://localhost:8000/mcp",
            },
        },
    )
    return client

async def do_debugging(tools):
    tool_name = 'get_order_details'
    tool = None
    for t in tools:
        if t.name == tool_name:
            tool = t
    if tool:
        response = await tool.ainvoke({"order_id": "9942"})
        log.debug(response)
    else:
        log.debug('tool not found')


@track_token_use
async def ship_order(cfg, callback_config):
    model = get_llm_model(cfg)
    log.debug('setup model')

    client = get_mcp_client()
    log.debug('setup mcp client')

    # agent setup
    mcp_tools = await client.get_tools()
    tools = [make_mistral_compatible(t) for t in mcp_tools]

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
