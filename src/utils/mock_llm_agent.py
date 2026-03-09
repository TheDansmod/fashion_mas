"""Generate Mock LLM to make each dev run cheaper when not testing the LLMs."""

from typing import Any, Type

from langchain_core.messages import AIMessage
from pydantic import BaseModel


class MockStructuredOutput:
    """Mock Outputs for Structured LLM calls."""

    def __init__(self, schema: Type[BaseModel]):
        """Get the schema or class used to structure the call."""
        self.schema = schema

    def invoke(self, prompt: Any, *args, **kwargs) -> BaseModel:
        """Respond when the structured llm is called."""
        schema_name = self.schema.__name__

        if schema_name == "NumRecommendations":
            return self.schema(num_recommendations=3)
        elif schema_name == "SingleImageDescription":
            return self.schema(
                image_description="A rigorous representation of a blue cotton shirt."
            )
        elif schema_name == "RequiredClothes":
            return self.schema(
                required_clothes_descriptions=[
                    "Dark wash denim jeans",
                    "Black tailored trousers",
                    "Khaki chinos",
                ]
            )
        elif schema_name == "ValidCategories":
            return self.schema(categories=["JEANS", "PANTS"])
        else:
            raise NotImplementedError(
                f"Mock configuration for schema {schema_name} is unresolved."
            )


class ChatMockLLM:
    """The Mocked LLM itself."""

    def __init__(self, *args, **kwargs):
        """Setup the substrings that appear in unstructured input prompts."""
        self.intent_node_prompt_substring = (
            "Your goal is to analyse the user request"
            " (provided below) and determine what the focus of the VLM should be."
        )
        self.explanation_node_prompt_substring = (
            "to explain how the recommended image "
            "successfully satisfies a concrete part or all of the user's request."
        )

    def invoke(self, prompt: Any, *args, **kwargs) -> AIMessage:
        """Respond to unstructured llm invocations."""
        prompt_str = str(prompt).lower()

        if self.intent_node_prompt_substring.lower() in prompt_str:
            return AIMessage(
                content="Focus on the stylistic coherence of the lower-body garments."
            )
        elif self.explanation_node_prompt_substring.lower() in prompt_str:
            return AIMessage(
                content="The properties of this garment synergize with the input image."
            )
        return AIMessage(content="Default mocked unstructured output.")

    def with_structured_output(
        self, schema: Type[BaseModel], **kwargs
    ) -> MockStructuredOutput:
        """Respond to structured llm invocations."""
        return MockStructuredOutput(schema)
