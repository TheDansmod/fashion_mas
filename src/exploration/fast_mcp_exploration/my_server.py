from fastmcp import FastMCP
from fastmcp.dependencies import CurrentContext
from fastmcp.server.context import Context

mcp = FastMCP("My MCP Server")

@mcp.tool
async def greet(name: str, ctx: Context = CurrentContext()) -> str:
    await ctx.info(f"Inside the greet function with {name=}")
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run(transport='http', port=8000)
