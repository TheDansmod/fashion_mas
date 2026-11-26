"""Build the chat model from settings."""

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from frag.settings import LLMSettings


def build_llm(settings: LLMSettings, aws_region: str) -> BaseChatModel:
    """Return the configured chat model, with an optional client-side rate limiter.

    The provider is chosen explicitly by `settings.provider`, never inferred from the
    model id. Provider packages are imported lazily so only the one in use is loaded.
    """
    rate_limiter = None
    if settings.rate_limit_rps:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=settings.rate_limit_rps,
            check_every_n_seconds=settings.rate_limit_check_every_s,
            max_bucket_size=settings.rate_limit_max_bucket,
        )
    match settings.provider:
        case "bedrock":
            from langchain_aws import (
                ChatBedrockConverse,
            )

            return ChatBedrockConverse(
                model_id=settings.model_id,
                temperature=settings.temperature,
                rate_limiter=rate_limiter,
                region_name=aws_region,
            )
        case "google":
            from langchain_google_genai import (
                ChatGoogleGenerativeAI,
            )

            return ChatGoogleGenerativeAI(
                model=settings.model_id,
                temperature=settings.temperature,
                rate_limiter=rate_limiter,
            )
