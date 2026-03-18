import asyncio
import base64
from langchain_core.tools import StructuredTool
from pathlib import Path
import uuid
from PIL import Image
import json
from langchain_mcp_adapters.client import MultiServerMCPClient

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

async def main():
    client = MultiServerMCPClient(
        {
            "product_catalogue_server": {
                "transport": "streamable_http",
                "url": "http://localhost:8000/mcp",
            },
        }
    )
    tools = await client.get_tools()
    tools = [make_mistral_compatible(t) for t in tools]
    tool = None
    for t in tools:
        if t.name == 'get_datapoint_by_index':
            tool = t
            break
    if not tool:
        raise ValueError("DB tool not found")
    response = await tool.ainvoke({"index": 10})
    for block in response:
        if 'image_url' in block:
            image_url = block['image_url']
            break
        else:
            print(block)
    else:
        raise ValueError("Could not find image")
    folder_path = r'/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas/data/temp_images/'
    directory = Path(folder_path)
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4()}.png"
    file_path = directory / filename
    if "," in image_url:
        base64_string = image_url.split(",")[1]
        image_data = base64.b64decode(base64_string)
    else:
        image_data = base64.b64decode(image_url)
    with open(file_path, 'wb') as file:
        file.write(image_data)
    print(file_path)


if __name__ == '__main__':
    asyncio.run(main())
