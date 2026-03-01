"""Given some model name, provide a unified interface to access the model."""

import logging

from langchain_ollama import ChatOllama

log = logging.getLogger(__name__)


def get_llm_provider(name, *args, **kwargs):
    """Get the right LLM provider based on the model name."""
    if name in ["qwen3-vl:8b-thinking"]:
        return ChatOllama
    else:
        raise ValueError("Unable to map name to LLM Provider")
