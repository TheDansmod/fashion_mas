"""Given some model name, provide a unified interface to access the model."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from langchain_core.rate_limiters import InMemoryRateLimiter
from loguru import logger as log


def get_llm_provider(name, *args, **kwargs):
    """Get the right LLM provider based on the model name."""
    if name in ["qwen3-vl:8b-thinking", "qwen3-vl:4b-thinking"]:
        return ChatOllama
    elif "gemma" in name:
        return ChatGoogleGenerativeAI
    elif "mistral" in name:
        return ChatMistralAI
    else:
        raise ValueError("Unable to map name to LLM Provider")

# we allow use of cfg here - this comes from the Container - since this is setup - not a leaf function
def get_rate_limiter(cfg):
    """Sets up a rate limiter for the LLM agent."""
    rps = cfg.models.rate_limiter.requests_per_second
    check_int = cfg.models.rate_limiter.check_every_n_seconds
    bucket_sz = cfg.models.rate_limiter.max_bucket_size
    if cfg.models.vlm_agent.use_rate_limiter:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=rps,
            check_every_n_seconds=check_int,
            max_bucket_size=bucket_sz,
        )
    else:
        rate_limiter = None
    return rate_limiter


# we allow use of cfg here - this comes from the Container - since this is setup - not a leaf function
def get_llm_model(cfg):
    """Creates and returns an LLM model for use with appropriate rate limits."""
    model = cfg.models.llm_provider(
        model=cfg.models.vlm_agent.name,
        temperature=cfg.models.vlm_agent.temp,
        rate_limiter=get_rate_limiter(cfg),
    )
    return model

