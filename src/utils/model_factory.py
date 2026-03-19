"""Given some model name, provide a unified interface to access the model."""

import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama

# from src.utils.mock_llm_agent import ChatMockLLM
from src.utils.mock_llm_agent import ChatMockLLM

log = logging.getLogger(__name__)


def get_llm_provider(name, *args, **kwargs):
    """Get the right LLM provider based on the model name."""
    if name in ["qwen3-vl:8b-thinking", "qwen3-vl:4b-thinking"]:
        return ChatOllama
    elif "gemma" in name:
        return ChatGoogleGenerativeAI
    elif "mock" in name.lower():
        return ChatMockLLM
    elif "mistral" in name:
        return ChatMistralAI
    else:
        raise ValueError("Unable to map name to LLM Provider")
