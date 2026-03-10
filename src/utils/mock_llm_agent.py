"""Generate Mock LLM to make each dev run cheaper when not testing the LLMs."""

import logging
import random
from typing import Any, Type

from langchain_core.messages import AIMessage
from pydantic import BaseModel

log = logging.getLogger(__name__)


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
                    "Blue shorts",
                    "Black office shoe",
                ]
            )
        elif schema_name == "ValidCategories":
            if "jeans" in prompt:
                return self.schema(categories=["JEANS", "PANTS"])
            elif "shorts" in prompt:
                return self.schema(categories=["SHORTS"])
            elif "shoe" in prompt:
                return self.schema(categories=["BOOTS", "LOAFERS", "SNEAKERS"])
            else:
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
        self.filtration_node_prompt_substring = (
            "assign one or more of following categories to it:"
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
        elif self.filtration_node_prompt_substring.lower() in prompt_str:
            cats = random.choices(
                ["JEANS", "SCARVES", "SANDALS", "SKIRTS", "TOPS", "DRESSES"],
                k=random.randint(1, 3),
            )
            return AIMessage(content=f"The matching categories are {', '.join(cats)}")
        return AIMessage(content="Default mocked unstructured output.")

    def with_structured_output(
        self, schema: Type[BaseModel], **kwargs
    ) -> MockStructuredOutput:
        """Respond to structured llm invocations."""
        return MockStructuredOutput(schema)
