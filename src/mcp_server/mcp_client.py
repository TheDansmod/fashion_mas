import logging

from langchain_mcp_adapters.client import MultiServerMCPClient  
from langchain_core.tools import StructuredTool
import json

log = logging.getLogger(__name__)

def make_mistral_compatible(tool):
    """Wraps an MCP tool to ensure it returns a plain string."""
    def stringify_invoke(*args, **kwargs):
        tool_input = args[0] if args else kwargs
        response = tool.invoke(tool_input)
        log.debug(f'{response=}')
        if isinstance(response, list):
            result = "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in response)
        else:
            result = str(response)
        return result

    async def stringify_ainvoke(*args, **kwargs):
        tool_input = args[0] if args else kwargs
        response = await tool.ainvoke(tool_input)
        log.debug(f'{response=}')
        if isinstance(response, list):
            result = "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in response)
        else:
            result = str(response)
        return result

    return StructuredTool.from_function(
        func=stringify_invoke,
        coroutine=stringify_ainvoke,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )

def get_mcp_client():
    client = MultiServerMCPClient(
        {
            "product_catalogue_server": {
                "transport": "streamable_http",
                "url": "http://localhost:8000/mcp",
            },
        },
    )
    return client

async def run_client(cfg):
    log.debug('running client')
    client = get_mcp_client()
    tools = await client.get_tools()
    tools = [make_mistral_compatible(t) for t in tools]
    tool_name = 'semantic_search'
    tool = None
    for t in tools:
        if t.name == tool_name:
            tool = t
    if tool:
        description = "Black formal round-toed shoes."
        categories = ["BOOTS", "BOAT SHOES & MOCCASINS"]
        num_matches = 3
        # response is a list of dicts
        response = await tool.ainvoke({"description": description, "categories": categories, "num_matches": num_matches})
        log.debug(response)
        log.debug(type(response))
        # for resp in response:
        #     for key, value in resp.items():
        #         if key == 'text':
        #             value = json.dumps(json.loads(value), indent=2)
        #         log.debug(f'{key}: {value}')
    else:
        log.debug('tool not found')
