import pytest
from tests.mocks.mock_llm_agent import ChatMockLLM, mock_create_agent

@pytest.fixture
def mocked_llm():
    return ChatMockLLM()

@pytest.fixture
def mocked_create_agent():
    return mock_create_agent
