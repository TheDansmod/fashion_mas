import logging

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

log = logging.getLogger(__name__)


def make_mistral_compatible(tool):
    """Wraps an MCP tool to ensure it returns a plain string."""

    def sanitize_response(response):
        result = []
        if isinstance(response, list):
            for block in response:
                if isinstance(block, dict):
                    block_type = block["type"]
                    if block_type == "text":
                        result.append({"type": "text", "text": block["text"]})
                    elif block_type == "image":
                        result.append(
                            {
                                "type": "image_url",
                                "image_url": f"data:{block['mime_type']};base64,{block['base64']}",
                            }
                        )
                    else:
                        raise ValueError("unexpected type")
                else:
                    # we default to converting the whole thing to string if element of
                    # the list is not a dictionary
                    result.append({"type": "text", "text": str(block)})
            return result
        else:
            # we just default to string if response is not a list
            return str(response)

    def sync_wrapper(*args, **kwargs):
        tool_input = args[0] if args else kwargs
        response = tool.invoke(tool_input)
        return sanitize_response(response)

    async def async_wrapper(*args, **kwargs):
        tool_input = args[0] if args else kwargs
        response = await tool.ainvoke(tool_input)
        log.debug(f"{len(response)=}\n{response=}")
        return sanitize_response(response)

    return StructuredTool.from_function(
        func=sync_wrapper,
        coroutine=async_wrapper,
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


async def test_semantic_search(tools):
    tool_name = "semantic_search"
    tool = None
    for t in tools:
        if t.name == tool_name:
            tool = t
    if tool:
        description = "Black formal round-toed shoes."
        categories = ["BOOTS", "BOAT SHOES & MOCCASINS"]
        num_matches = 3
        # response is a list of dicts
        response = await tool.ainvoke(
            {
                "description": description,
                "categories": categories,
                "num_matches": num_matches,
            }
        )
        log.debug(response)
    else:
        log.debug("tool not found")


async def test_product_categories(tools):
    tool_name = "get_product_categories"
    tool = None
    for t in tools:
        if t.name == tool_name:
            tool = t
    if tool:
        response = await tool.ainvoke(dict())
        log.debug(response)
    else:
        log.debug("tool not found")


async def run_client(cfg):
    log.debug("running client")
    client = get_mcp_client()
    tools = await client.get_tools()
    tools = [make_mistral_compatible(t) for t in tools]
    await test_semantic_search(tools)
    await test_product_categories(tools)
