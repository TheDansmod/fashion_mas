"""Config for LLM models used in the application."""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, computed_field

from frag.utils.model_factory import get_llm_provider

class VLMAgent(BaseModel):
    """VLM agent for generating sample images or converting provided ones to descriptions."""

    model_config = ConfigDict(frozen=True, validate_default=True)

    name: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    temp: float = 0.6
    use_rate_limiter: bool = True


class RateLimiter(BaseModel):
    """For the langchain InMemoryRateLimiter.

    It uses a token (not LLM tokens) bucket model.
    """

    model_config = ConfigDict(frozen=True, validate_default=True)

    # this is the rate at which tokens get added to the bucket - it is kept far below the actual threshold to decrease risk of rate limiting even further and because there is also a 500k tokens per minute rate limit which is not captured. The limit for mistral-medium-latest is 375k tokens per min while for mistral-large-latest is 50k
    # for aws claude haiku 4.5 the request limit is 50 requests per second which translates to 0.833..
    requests_per_second: float = 0.8
    # this sets the frequency with which it checks if tokens are available to make a request
    check_every_n_seconds: int = 1
    # this is the size of the bucket - allows a burst of requests if the rate limiting is actually long term rather than strictly per second
    max_bucket_size: float = 1.0


class ModelsConfig(BaseModel):
    """Config for various LLM models used in the application."""

    model_config = ConfigDict(frozen=True, validate_default=True)

    vlm_agent: VLMAgent = VLMAgent()
    rate_limiter: RateLimiter = RateLimiter()

    # pydantic does not do any additional logic on this field
    # so, this does not do validation,
    # there is a cached_property + model_validator pattern that can do validation
    # but it (pattern) is not required for now
    @computed_field
    @property
    def llm_provider(self) -> Any:
        return get_llm_provider(self.vlm_agent.name)
