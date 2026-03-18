# this is taken from gemini
"""Generate Mock LLM to make each dev run cheaper when not testing the LLMs."""

import logging
from typing import Any, Mapping, Optional, Type

from langchain_core.messages import AIMessage
from pydantic import BaseModel

log = logging.getLogger(__name__)


def _flatten_prompt(payload: Any) -> str:
    """Convert arbitrary prompt/message payloads into a searchable string."""
    if payload is None:
        return ""

    if isinstance(payload, str):
        return payload

    if isinstance(payload, Mapping):
        return " ".join(_flatten_prompt(v) for v in payload.values())

    if isinstance(payload, (list, tuple, set)):
        return " ".join(_flatten_prompt(v) for v in payload)

    content = getattr(payload, "content", None)
    if content is not None:
        return _flatten_prompt(content)

    return str(payload)


def _count_images(payload: Any) -> int:
    """Best-effort count of image parts inside prompt payloads."""
    if payload is None:
        return 0

    if isinstance(payload, Mapping):
        count = 1 if payload.get("type") in {"image", "image_url"} else 0
        return count + sum(_count_images(v) for v in payload.values())

    if isinstance(payload, (list, tuple, set)):
        return sum(_count_images(v) for v in payload)

    content = getattr(payload, "content", None)
    if content is not None:
        return _count_images(content)

    return 0


class MockStructuredOutput:
    """Mock outputs for structured LLM calls."""

    def __init__(
        self,
        schema: Type[BaseModel],
        fixtures: Optional[dict[str, Any]] = None,
    ):
        self.schema = schema
        self.fixtures = fixtures or {}

    def _build_response(self, value: Any) -> BaseModel:
        if isinstance(value, self.schema):
            return value
        if isinstance(value, BaseModel):
            return self.schema(**value.model_dump())
        if isinstance(value, dict):
            return self.schema(**value)
        raise TypeError(
            f"Unsupported mock value for schema {self.schema.__name__}: {type(value)}"
        )

    def invoke(self, prompt: Any, *args, **kwargs) -> BaseModel:
        """Respond when the structured LLM is called."""
        schema_name = self.schema.__name__

        override = self.fixtures.get("structured", {}).get(schema_name)
        if override is not None:
            return self._build_response(override)

        default_responses = {
            "NumRecommendations": {
                "num_recommendations": 3,
            },
            "SingleImageDescription": {
                "image_description": (
                    "Mock description of a fashion item with visible color, fit, "
                    "material, and silhouette details."
                ),
            },
            "RequiredClothes": {
                "required_clothes_descriptions": [
                    "White cotton shirt",
                    "Dark straight-leg jeans",
                    "Black leather loafers",
                ],
            },
        }

        if schema_name not in default_responses:
            raise NotImplementedError(
                f"Mock configuration for schema {schema_name} is unresolved."
            )

        return self._build_response(default_responses[schema_name])

    async def ainvoke(self, prompt: Any, *args, **kwargs) -> BaseModel:
        return self.invoke(prompt, *args, **kwargs)


class MockAgent:
    """Mock object returned by create_agent(...)."""

    def __init__(
        self,
        response_format: Type[BaseModel],
        fixtures: Optional[dict[str, Any]] = None,
        tools: Optional[list[Any]] = None,
    ):
        self.response_format = response_format
        self.fixtures = fixtures or {}
        self.tools = tools or []

    def _build_response(self, value: Any) -> BaseModel:
        if isinstance(value, self.response_format):
            return value
        if isinstance(value, BaseModel):
            return self.response_format(**value.model_dump())
        if isinstance(value, dict):
            return self.response_format(**value)
        raise TypeError(
            "Unsupported mock value for agent response format "
            f"{self.response_format.__name__}: {type(value)}"
        )

    def invoke(self, inputs: Any, *args, **kwargs) -> dict[str, Any]:
        """Match the shape returned by agent.invoke(...) in rag_agent."""
        schema_name = self.response_format.__name__

        override = self.fixtures.get("agents", {}).get(schema_name)
        if override is not None:
            structured_response = self._build_response(override)
        elif schema_name == "UpdatedUserRequest":
            num_images = _count_images(inputs)
            structured_response = self.response_format(
                relevant_image_indexes=list(range(num_images)),
                updated_user_query=self.fixtures.get(
                    "default_updated_user_query",
                    "mocked updated user query",
                ),
            )
        elif schema_name == "MatchedImageId":
            structured_response = self.response_format(
                image_id=self.fixtures.get("default_matched_image_id", 0)
            )
        else:
            raise NotImplementedError(
                f"Mock configuration for agent response format {schema_name} is unresolved."
            )

        return {
            "structured_response": structured_response,
            "messages": [AIMessage(content=structured_response.model_dump_json())],
        }

    async def ainvoke(self, inputs: Any, *args, **kwargs) -> dict[str, Any]:
        return self.invoke(inputs, *args, **kwargs)


class ChatMockLLM:
    """The mocked LLM itself."""

    def __init__(self, fixtures: Optional[dict[str, Any]] = None, *args, **kwargs):
        self.fixtures = fixtures or {}
        self.bound_tools: list[Any] = []

    def invoke(self, prompt: Any, *args, **kwargs) -> AIMessage:
        """Respond to unstructured LLM invocations."""
        prompt_str = _flatten_prompt(prompt).lower()

        override = self.fixtures.get("unstructured")
        if isinstance(override, str):
            return AIMessage(content=override)

        intent_markers = [
            "focus of the vlm",
            "vlm should be",
            "analyse the user request",
            "analyze the user request",
            "request_focus_prompt",
        ]
        explanation_markers = [
            "explain how the recommended image",
            "recommended image description",
            "satisfies a concrete part",
            "successfully satisfies",
        ]

        if any(marker in prompt_str for marker in intent_markers):
            return AIMessage(
                content=(
                    "Focus on garment type, colors, materials, fit, and styling "
                    "constraints that matter for retrieval."
                )
            )

        if any(marker in prompt_str for marker in explanation_markers):
            return AIMessage(
                content=(
                    "This recommendation matches the requested clothing type, style, "
                    "and overall aesthetic."
                )
            )

        return AIMessage(content="Default mocked unstructured output.")

    async def ainvoke(self, prompt: Any, *args, **kwargs) -> AIMessage:
        return self.invoke(prompt, *args, **kwargs)

    def with_structured_output(
        self,
        schema: Type[BaseModel],
        **kwargs,
    ) -> MockStructuredOutput:
        """Respond to structured LLM invocations."""
        return MockStructuredOutput(schema=schema, fixtures=self.fixtures)

    def bind_tools(self, tools: list[Any]) -> "ChatMockLLM":
        """Keep compatibility with code paths that bind tools to the model."""
        self.bound_tools = tools
        return self


def mock_create_agent(
    model: Any = None,
    tools: Optional[list[Any]] = None,
    response_format: Optional[Type[BaseModel]] = None,
    **kwargs,
) -> MockAgent:
    """Drop-in stand-in for langchain.agents.create_agent."""
    if response_format is None:
        raise ValueError("mock_create_agent requires response_format to be provided.")

    fixtures = kwargs.pop("fixtures", None)
    if fixtures is None and model is not None:
        fixtures = getattr(model, "fixtures", None)

    return MockAgent(
        response_format=response_format,
        fixtures=fixtures,
        tools=tools,
    )


