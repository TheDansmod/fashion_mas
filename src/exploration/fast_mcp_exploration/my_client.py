import asyncio
from fastmcp import Client

async def log_handler(msg):
    print(msg)

client = Client("http://localhost:8000/mcp", log_handler=log_handler)

async def call_tool(name: str):
    async with client:
        result = await client.call_tool("greet", {"name": name})
        print(result)

asyncio.run(call_tool("Ford"))
