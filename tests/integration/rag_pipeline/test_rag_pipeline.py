import pytest

from frag.rag_pipeline.rag_agent import run_fashion_agent

@pytest.mark.asyncio
async def test_run_fashion_agent(container, mock_input):
    mock_input("Please provide a pant that matches the uploaded shirt", "1", "/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas/data/trendy-check-shirt-10212-2.jpg", "quit")
    await run_fashion_agent()
