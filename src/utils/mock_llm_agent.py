from pydantic import BaseModel
from langchain_core.messages import AIMessage
import random

class MockAgent:
    def __init__(self, schema):
        self.schema = schema
        self.schema_name = schema.__name__

    def invoke(self, *args, **kwargs):
        if self.schema_name == "UpdatedUserRequest":
            # for human node
            return {'structured_response': self.schema(relevant_image_indexes=[0, 2], updated_user_query="Mock updated user query.")}
        else:
            raise NotImplementedError(f"invoking agent for this schema name {self.schema_name} is not implemented")

    async def ainvoke(self, *args, **kwargs):
        if self.schema_name == "MatchedImageId":
            # for recommender node
            return {'structured_response': self.schema(image_id=random.randint(0, 260400))}
        else:
            raise NotImplementedError(f"async invoking agent for this schema name {self.schema_name} is not implemented")

class MockStructuredOutput:
    def __init__(self, schema):
        self.schema = schema
        self.schema_name = schema.__name__

    def invoke(self, *args, **kwargs):
        if self.schema_name == "NumRecommendations":
            # for the quantifier node
            return self.schema(num_recommendations=3)
        elif self.schema_name == "SingleImageDescription":
            # for the vision node
            return self.schema(image_description="A blue cotton shirt with short sleeves.")
        elif self.schema_name == "RequiredClothes":
            # for modifier node
            return self.schema(required_clothes_descriptions=["Dard wash denim jeans", "blue shorts", "black office shoe"])
        elif self.schema_name == "CriticalEvaluation":
            # for critique node
            return self.schema(satisfactory="No", correction="Some correction")
        else:
            raise NotImplementedError(f"structured mock not implemented for {self.schema_name}")


class ChatMockLLM(BaseModel):
    def __init__(self, *args, **kwargs):
        self.intent_node_prompt_substring = "Ensure the instructions are concise (No more than 5 sentences) and only relate to what aspect of the image should be described / extracted."
        self.intent_node_return = AIMessage(content="Mock response: Focus on the stylistic coherence of the lower body garments.")
        self.explanation_node_prompt_substring = "Please explain how the recommended products successfully satisfy the user's request."
        self.explanation_node_return = AIMessage(content="The properties of the input image synergize with the input image")

    def invoke(self, prompt, *args, **kwargs):
        if isinstance(prompt, str):
            prompt_str = prompt.lower()
            if self.intent_node_prompt_substring.lower() in prompt_str:
                # for intent node
                return self.intent_node_return
            else:
                raise NotImplementedError(f"Invoking model for this string prompt: {prompt} is not implemented")
        elif isinstance(prompt, list):
            for content in prompt[0].content:
                # for explanation node
                if content['type'] == 'text' and self.explanation_node_prompt_substring.lower() in content['text'].lower():
                    return self.explanation_node_return
            raise NotImplementedError(f"Mock LLM in list prompt could not find the right text: {prompt}")
        else:
            raise NotImplementedError(f"Mock LLM got input that was neither list nor string: {prompt}")

    def with_structured_output(self, schema, **kwargs):
        return MockStructuredOutput(schema)

def mock_create_agent(model, response_format, tools=None, **kwargs):
    return MockAgent(response_format)
