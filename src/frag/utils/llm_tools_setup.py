from abc import ABC, abstractmethod

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import StructuredTool

from frag.rag_pipeline.local_tools import ProductCatalogueTools

class ToolsClient(ABC):
    @abstractmethod
    async def get_llm_tools(self):
        """Get all the tools for use by LLM models."""
        ...

    @abstractmethod
    async def get_db_tool(self):
        """Get the single tool to interface with the db."""
        ...

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
        return sanitize_response(response)

    return StructuredTool.from_function(
        func=sync_wrapper,
        coroutine=async_wrapper,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )

def get_tool_with_name(tools, search_name):
    """Given a list of mcp tools, returns the tool with the search name, or errors."""
    tool = None
    for t in tools:
        if t.name == search_name:
            tool = t
            break
    if not tool:
        raise ValueError(f"DB tool not found: {search_name}")
    return tool

class MCPToolsClient(ToolsClient):
    def __init__(self, mcp_client_transport, mcp_url, llm_tool_names, db_tool_name):
        self.client = MultiServerMCPClient(
            {
                "product_catalogue_server": {
                    "transport": mcp_client_transport,
                    "url": mcp_url,
                },
            }
        )
        self.llm_tool_names = llm_tool_names
        self.db_tool_name = db_tool_name

    async def get_llm_tools(self):
        tools = await self.client.get_tools()
        llm_tools = [make_mistral_compatible(tool) for tool in tools if tool.name in self.llm_tool_names]
        return llm_tools

    async def get_db_tool(self):
        tools = await self.client.get_tools()
        tool = get_tool_with_name(tools, self.db_tool_name)
        return tool

class LocalToolsClient(ToolsClient):
    def __init__(self, connector, embedder, product_categories, llm_tool_names, db_tool_name, fgen_args):
        self.client = ProductCatalogueTools(connector, embedder, product_categories, fgen_args)
        self.llm_tool_names = llm_tool_names
        self.db_tool_name = db_tool_name
        self.tools = self.client.get_tools()

    async def get_llm_tools(self):
        llm_tools = [t for t in self.tools if t.name in self.llm_tool_names]
        return llm_tools

    async def get_db_tool(self):
        tool = get_tool_with_name(self.tools, self.db_tool_name)
        return tool

def get_tools_client(use_mcp_server, connector, embedder, product_categories, llm_tool_names, db_tool_name, mcp_client_transport, mcp_url, fgen_args):
    if use_mcp_server:
        return MCPToolsClient(mcp_client_transport, mcp_url, llm_tool_names, db_tool_name)
    else:
        return LocalToolsClient(connector, embedder, product_categories, llm_tool_names, db_tool_name, fgen_args)
