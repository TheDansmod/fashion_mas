import logging

import pytest

from frag.rag_pipeline.rag_agent import run_fashion_agent


@pytest.mark.asyncio
async def test_run_fashion_agent(container, mock_input, mocked_create_agent, caplog):
    mock_input(
        "Please provide a pant that matches the uploaded shirt",
        "1",
        "/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas/data/trendy-check-shirt-10212-2.jpg",
        "I don't like the suggested items please suggest again.",
        "0",
        "quit",
    )

    expected_logs = [
        "Entered human node",
        "Entered quantifier node",
        "Entered intent node",
        "Entered vision node",
        "Entered modifier node.",
        "Entered recommender node.",
        "Entered critique node.",
        "Entered explanation node.",
        "Exiting gracefully.",
    ]
    with caplog.at_level(logging.DEBUG):
        await run_fashion_agent()

    for expected_msg in expected_logs:
        assert any(expected_msg in msg for msg in caplog.messages), (
            f"Expected log message not found: {expected_msg!r}"
        )
