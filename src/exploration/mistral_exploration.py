"""Exploring how to use Mistral's LLM model.

What I am looking to achieve:
    - [x] get it to do tool calling
    - [x] get it to do structured outputs
    - [x] get it to handle image inputs
    - [x] track token usage since Mistral has token limits
    - [x] enforce rate limits (1 request / second)
    - [ ] how many 256x256 images can the model handle together?
"""

import logging

import hydra
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import BaseModel, Field

from src.utils.common_utils import (
    encode_image,
    get_image_prompt_message,
    update_token_use,
    track_token_use,
    get_rate_limiter,
    get_multi_image_prompt_message,
)

log = logging.getLogger(__name__)


def check_token_usage(cfg):
    """Checks how to capture token usage and save it."""
    callback = UsageMetadataCallbackHandler()
    provider = hydra.utils.instantiate(cfg.models.vlm_agent)
    model = provider(
        model=cfg.models.vlm_agent.name, temperature=cfg.models.vlm_agent.temp
    )
    prompt = (
        "Give a short (5 - 10 sentences) explanation "
        "for what causes the sky to be blue?"
    )
    response = model.invoke(prompt, config={"callbacks": [callback]})
    log.info(response.content)
    update_token_use(cfg, callback.usage_metadata)


@tool
def population_by_country(country_name: str) -> int:
    """Returns an integer representing the number of people in the input country.

    Args:
        country_name (str): The name of the country whose population you wish to know.
            Some examples of country names are: China, India, United Kingdom.

    Returns:
        (int): A integer representing the number of people living in the country.
    """
    log.info(f"Population tool invoked with country: {country_name}.")
    name = country_name.strip().lower()
    if name == "china":
        return 7962
    elif name == "india":
        return 9116
    else:
        return 42


@tool
def calculate_tip_and_total(bill_amount: float) -> str:
    """Calculate the optimal tip amount and total amount for a given bill.

    Args:
        bill_amount (float): The tip-free cost of services in dollars.

    Returns:
        (str): A string contaning the tip amount and the total amount including the tip
            (all in dollars).
    """
    log.debug(f"Invoking the tip calculator with bill amount {bill_amount}")
    tip = bill_amount * (20 / 100)
    return f"Tip: ${tip:.2f}, Total: ${bill_amount + tip:.2f}"


class PopulationList(BaseModel):
    """List of Population Values for testing LLM Model."""

    pop_list: list[int] = Field(
        ...,
        description=(
            "List of integers where each integer represents the population "
            "of a country, in the order requested by the user"
        ),
    )


class BillBreakdown(BaseModel):
    """Breakdown of bill into base amount, tip, and total."""

    bill_amount: float = Field(
        ..., description="The base bill amount without the tip included."
    )
    tip_value: float = Field(..., description="The optimal tip amount on the bill.")
    total_amount: float = Field(
        ...,
        description=(
            "The total cost to be paid to the business including the base "
            "bill amount and the tip amount."
        ),
    )


class MultiBillBreakdown(BaseModel):
    """List of bill breakdowns."""

    breakdowns: list[BillBreakdown] = Field(
        ...,
        min_length=1,
        description="A list of bill breakdowns with one breakdown per requested bill.",
    )


def check_tools_and_structured(cfg):
    """Checking how model handles structured outputs and tool calling."""
    callback = UsageMetadataCallbackHandler()
    callback_config = {"callbacks": [callback]}
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
    provider = hydra.utils.instantiate(cfg.models.vlm_agent)
    model = provider(
        model=cfg.models.vlm_agent.name,
        temperature=cfg.models.vlm_agent.temp,
        rate_limiter=rate_limiter,
    )
    agent = create_agent(
        model=model,
        tools=[calculate_tip_and_total],
        response_format=MultiBillBreakdown,
        debug=True,
    )
    message = [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Please calculate the tip on a $30.71 bill, a $701.52 bill, "
                        "and a $106.11 bill; and return a list of bill breakdowns "
                        "where each breakdowns consists of the base bill amount, the "
                        "tip amount, and the total amount."
                    ),
                },
            ]
        )
    ]
    # when doing invocation the input must be a dictionary with messages key
    response = agent.invoke({"messages": message}, config=callback_config)
    # the response is a dictionary with keys messages, and if you requested structured
    # response - it also contains a structured_response key. The value of the structured
    # response is the class which you wanted. If you did not request a structured
    # response you should collect the content (.content) of the last element in the
    # messages list.
    log.debug(response["structured_response"])
    update_token_use(cfg, callback.usage_metadata)


def check_rate_limiting(cfg):
    """Check how to rate limit a model."""
    callback = UsageMetadataCallbackHandler()
    callback_config = {"callbacks": [callback]}
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.1, check_every_n_seconds=3, max_bucket_size=1.0
    )
    provider = hydra.utils.instantiate(cfg.models.vlm_agent)
    model = provider(
        model=cfg.models.vlm_agent.name,
        temperature=cfg.models.vlm_agent.temp,
        rate_limiter=rate_limiter,
    )
    prompt = (
        "Please give a code example of invoking a create_agent based agent in "
        "langchain. In the example I saw they were using a raw dictionary with the "
        "messages key to input the messages to the agent. Would it be better (the "
        "intended usage pattern or the recommended pattern) to use HumanMessage from "
        "langchain_core? Explain in detail.",
    )
    response = model.invoke(prompt, config=callback_config)
    log.debug(response.content)
    update_token_use(cfg, callback.usage_metadata)


def check_image_inputs(cfg):
    """Check how the model can handle image inputs."""
    callback = UsageMetadataCallbackHandler()
    callback_config = {"callbacks": [callback]}
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
    provider = hydra.utils.instantiate(cfg.models.vlm_agent)
    model = provider(
        model=cfg.models.vlm_agent.name,
        temperature=cfg.models.vlm_agent.temp,
        rate_limiter=rate_limiter,
    )
    prompt = "Please provide a detailed description of the clothing item in the image."
    msg = get_image_prompt_message(
        image_path=cfg.misc.input_image_path_01, text_prompt=prompt
    )
    response = model.invoke(msg, config=callback_config)
    log.debug(response.content)
    update_token_use(cfg, callback.usage_metadata)


def mistral_sdk(cfg):
    """Check how to use mistral's API since langchain said they don't support images."""
    from mistralai.client import Mistral

    # NOTE: I have removed the mistral api since I got langchain's integration to work
    encoded_image = encode_image(image_path=cfg.misc.input_image_path_01)
    log.debug(
        f"The base64 representation of the image has length: {len(encoded_image)}"
    )
    with Mistral(api_key=cfg.models.api_keys.mistral) as mistral:
        response = mistral.chat.complete(
            model=cfg.models.vlm_agent.name,
            temperature=cfg.models.vlm_agent.temp,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert fashion assistant.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Please provide a detailed plain string (no "
                                "markdown) description of the clothing item in the "
                                "uploaded image."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{encoded_image}",
                        },
                    ],
                },
            ],
        )
    log.debug(response)

def get_llm_model(cfg):
    provider = hydra.utils.instantiate(cfg.models.vlm_agent)
    model = provider(
        model=cfg.models.vlm_agent.name,
        temperature=cfg.models.vlm_agent.temp,
        rate_limiter=get_rate_limiter(cfg),
    )
    return model

@track_token_use
def check_multi_image_input(cfg, callback_config):
    """To see how many images the Mistral model can handle together."""
    model = get_llm_model(cfg)
    prompt = 'Please provide a description of each of the uploaded images'
    msg = get_multi_image_prompt_message(image_paths=cfg.misc.test_image_paths, text_prompt=prompt)
    response = model.invoke(msg, config=callback_config)
    log.debug(response.content)
