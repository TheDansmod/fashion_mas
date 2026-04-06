"""Config for LLM models used in the application."""
from pydantic import BaseModel, ConfigDict, computed_field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama

from src.utils.model_factory import get_llm_provider
from src.utils.mock_llm_agent import ChatMockLLM

class VLMAgent(BaseModel):
    """VLM agent for generating sample images or converting provided ones to descriptions."""
    model_config = ConfigDict(frozen=True)

    name: str = 'mistral-medium-latest'
    temp: float = 0.6
    use_rate_limiter: bool = True

class RateLimiter(BaseModel):
    """For the langchain InMemoryRateLimiter.

    It uses a token (not LLM tokens) bucket model.
    """
    model_config = ConfigDict(frozen=True)

    # this is the rate at which tokens get added to the bucket - it is kept far below the actual threshold to decrease risk of rate limiting even further and because there is also a 500k tokens per minute rate limit which is not captured. The limit for mistral-medium-latest is 375k tokens per min while for mistral-large-latest is 50k
    requests_per_second: float = 0.5
    # this sets the frequency with which it checks if tokens are available to make a request
    check_every_n_seconds: int = 1
    # this is the size of the bucket - allows a burst of requests if the rate limiting is actually long term rather than strictly per second
    max_bucket_size: float = 1.0

class ModelsConfig(BaseModel):
    """Config for various LLM models used in the application."""
    model_config = ConfigDict(frozen=True)
    
    vlm_agent: VLMAgent = VLMAgent()
    rate_limiter: RateLimiter = RateLimiter()

    # pydantic does not do any additional logic on this field
    # so, this does not do validation, 
    # there is a cached_property + model_validator pattern that can do validation
    # but it is not required for now
    @computed_field
    @property
    def llm_provider(self) -> ChatOllama | ChatGoogleGenerativeAI | ChatMistralAI | ChatMockLLM:
        return get_llm_provider(self.vlm_agent.name)
