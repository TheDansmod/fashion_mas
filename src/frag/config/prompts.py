"""Prompts for LLM models."""

from pydantic import BaseModel, ConfigDict


class HumanNodePrompts(BaseModel):
    """Prompts used in Human Node."""

    model_config = ConfigDict(frozen=True)

    # this is used to update the relevant images and to update the user prompt while taking into account the previous user prompt when the user gives some feedback or adjustment for the recommendations.
    input_update_prompt: str = """
    You are an expert fashion assistant. You have previously interacted with the user. Below I will be providing you with the following:
      - original user input image descriptions (given only if the user had uploaded any images)
      - original user text input (the original request of the user)
      - response image descriptions (descriptions of images that you had replied with previously)
      - new user input images (actual images, not their descriptions. Might not be present if the user does not upload any images on the new request)
      - new user text input (the user might modify the original request, give a new request, ask for a modification of the response, etc)
    Please synthesise the old interaction and the new user request and do the following:
      - Each of the old image descriptions has an index for the image. If the old image(s) - either input or response - are relevant to the new user request, then please include the index of the relevant old image(s) in the returned value. For example, if the user in their new text input says, "I like the responses, but please show them in red" then that would mean that only the response images are relevant. So, your returned value should include the indices for the response images only.
      - Combine the original user text input (if it is relevant) and the new user text input to create a fresh user query that comprehensively, but concisely states what the user wants. The goal here is that we should not have to refer to the original user request to successfully satisfy the users wants. For example, if the user says something like "Instead of pants, show me hats that go well with my uploaded shirt" then then new user query will be something along the lines of "Please suggest hats that go well with the uploaded shirt".
    The newly uploaded images are assumed to be relevant in any case. Thus, your return value should have a list of indices of the relevant images, and a comprehensive, but concise string description of what the user wants now.

    Original User Input Image Descriptions:
    {original_image_descriptions}

    Original User Text Input:
    {original_text_input}

    Response Image Descriptions:
    {response_image_descriptions}

    New User Text Input:
    {new_text_input}
    """


class QuantifierNodePrompts(BaseModel):
    """Prompts used in Quantifier Node."""

    model_config = ConfigDict(frozen=True)

    # this is to obtain the number of recommendations the user has asked for (if any)
    num_recommendations_prompt: str = """
    You are a expert fashion assistant. Please carefully go through the user request and determine if they have specified how many clothing item / accessory recommendations they want. If they have specified how many recommendations they want, please return a single integer denoting the number of required recommendations. If they have not specified how many recommendations they want, please return 0.

    User Request:
    {user_request}
    """


class IntentNodePrompts(BaseModel):
    """Prompts used in Intent Node."""

    model_config = ConfigDict(frozen=True)

    # this is essentially guidance for the vision node - to determine what aspect of the image it should focus on
    # TODO: figure out if you should feed this model the input images as well - come up with a scenario where that might be required
    request_focus_prompt: str = """
    You are an expert fashion assistant. I am providing you with a user request associated with some images. The input images are going to be fed into a vision language model (VLM) with some instructions. The goal of the vision language model (not your goal) is to generate a description of the image (not to fulfil the user request) based on the instructions you generate. Your goal is to analyse the user request (provided below) and determine what the focus of the VLM should be. For example, if the user asks for "clothing items that look like this", then the focus of the vision model should be to extract detailed descriptions of the clothes in the images. Another example: if the user asks for "clothes that have the same colour palette as the image", then the focus of the vision model should be to extract the colour palette from the image. Please provide detailed instructions, suitable for a vision language model, on what it should focus on when looking at the image. Ensure the instructions are concise (No more than 5 sentences) and only relate to what aspect of the image should be described / extracted. In the instructions you provide, do not explain why these are the instructions you are providing. Your output should only contain the actual instructions which can directly be fed to the vision language model.

    User Request:
    {user_request}
    """


class VisionNodePrompts(BaseModel):
    """Prompts used in Vision Node."""

    model_config = ConfigDict(frozen=True)

    user_prompt: str = """
    You are a proficient fashion assistant. Please analyse the uploaded images closely and extract the features mentioned in the instructions below. Do not use markdown for the response, just keep it to a plain string (can use newlines). Ensure the response is short and concise - no more than 5 sentences.

    Image Focus Instructions:
    {image_focus_instructions}
    """


class ModifierNodePrompts(BaseModel):
    """Prompts used in Modifier Node."""

    model_config = ConfigDict(frozen=True)

    # this prompt is for obtained clothing item descriptions when input images are present
    images_present_prompt: str = """
    You are a proficient fashion assistant. Using the provided descriptions of the clothing items (Reference Clothes Descriptions) and the user's request (User Request), create a detailed description of the clothing items the user wants. The number of recommendations you should generate is {num_recommendations}. The user could request multiple clothing items (like "suggest me one pant and one shirt") or there could be multiple clothing items that satisfy the user's request (like "what pants will go well with this dress" might have multiple types of pants that might go well with the dress). Ensure that the provided descriptions are only about the description of clothing items and not about why the recommendation is being made. Return a list of strings of length {num_recommendations}. Each string is a detailed description, one for each potential item, that together will satisfy the user's requests. Please ensure that each description is distinct from the others, there should be no repetition of descriptions.

    Reference Clothes Descriptions:
    {reference_descriptions}

    User Request:
    {user_request}

    {critique_node_correction}
    """

    # this prompt is for obtaining clothing item descriptions when input images are absent
    images_absent_prompt: str = """
    You are an expert fashion assistant. Using the provided user request, please generate detailed descriptions of the sort of clothing items / accessories the user wants. The number of recommendations you should generate is {num_recommendations}. The user could request multiple clothing items (like "suggest me one pant and one shirt") or there could be multiple clothing items that satisfy the user's request (like "what pants will go well with this dress" might have multiple types of pants that might go well with the dress). Ensure that the provided descriptions are only about the description of clothing items and not about why the recommendation is being made. Return a list of strings of length {num_recommendations}. Each string is a detailed description, one for each potential item, that together will satisfy the user's requests. Please ensure that each description is distinct from the others, there should be no repetition of descriptions.

    User Request:
    {user_request}

    {critique_node_correction}
    """


class RecommenderNodePrompts(BaseModel):
    """Prompts for Recommender Node"""

    model_config = ConfigDict(frozen=True)

    match_clothes_prompt: str = """
    You are an expert fashion assistant. Below, I will be providing you a textual description of an item of clothing. Please do the following:
      - Get the available categories in the product catalogue
      - Based on the available categories, assign one or more categories to the clothing item description
      - Fetch some clothes that match that description (semantic search) from the product catalogue
      - Evaluate which of the returned images best satisfy the clothing item description
      - Return the id of the image which best satisfies the clothing item description

    Clothing Item Description:
    {item_description}
    """


class CritiqueNodePrompts(BaseModel):
    """Prompts for Critique Node"""

    model_config = ConfigDict(frozen=True)

    critique_prompt: str = """
    You are an expert fashion assistant. You are provided with the user's uploaded images (if any) and the user request, along with descriptions of products intended to completely satisfy the user's request. Please carefully analyse the request and the recommendations and give a Yes or No answer as to whether the recommendations satisfy the request. If they don't, then give a concise 3 sentence or smaller fix of what was wrong with the recommendations and what not to do.
    """


class ExplanationNodePrompts(BaseModel):
    """Prompts for Explanation Node"""

    model_config = ConfigDict(frozen=True)

    explanation_prompt: str = """
    You are a proficient fashion assistant. You are provided with the user's uploaded images (if any) and the user request, along with images of the recommended products and their descriptions. Please explain how the recommended products successfully satisfy the user's request.
    """


class PromptsSetup(BaseModel):
    """Setup for LLM Prompts."""

    model_config = ConfigDict(frozen=True)

    human_node: HumanNodePrompts = HumanNodePrompts()
    quantifier_node: QuantifierNodePrompts = QuantifierNodePrompts()
    intent_node: IntentNodePrompts = IntentNodePrompts()
    vision_node: VisionNodePrompts = VisionNodePrompts()
    modifier_node: ModifierNodePrompts = ModifierNodePrompts()
    recommender_node: RecommenderNodePrompts = RecommenderNodePrompts()
    critique_node: CritiqueNodePrompts = CritiqueNodePrompts()
    explanation_node: ExplanationNodePrompts = ExplanationNodePrompts()

    evaluation_prompt: str = """
    You are an expert fashion_assistant. Your role is judge how well some recommendations meet the user requirements. You are provided with the user's uploaded images (if any) and the user request. You are also provided with the recommendation images and the recommendation image descriptions. You task is to provide an integer rating out of 10 for each of the following criteria along with a single sentence explanation / justification for each rating that you have given:
    1. visual grounding: do the recommendations demonstrably reflect what is present in the uploaded image (if any)?
      What to check:
        - for "items that go well with X": do recommended colours and silhouettes complement the uploaded item?
        - for "same item, different style": is the core garment category preserved while style markers (cut, embellishment, material) shift?
        - for "aesthetic from image": do recommended items have the same mood and follow the right colour pattern
      Scoring anchor: 1 = no visible connection to the uploaded image; 10 = every recommendation is clearly visually anchored to it.
    2. individual item suitability: Each recommended item should be evaluated on its own merits for the stated use case. This criterion is request-type-sensitive:
        - Complementary styling: Does this item actually pair well with the reference piece?
        - Body type fit: Does the cut, drape, or silhouette genuinely flatter the stated body type?
        - Style variation: Is the item a clear stylistic departure while staying in the same garment category?
        - Aesthetic match: Does this item independently embody the target aesthetic?
      Scoring anchor: Average across all recommended items; 1 = most items are unsuitable; 10 = every item is individually well-justified.
    3. completeness and coverage: check whether the response addresses the full scope of the request. If a user asks "suggest a complete outfit to match the aesthetic of this image," returning only tops scores low. If a user specifies multiple constraints ("I need something formal but also comfortable"), all constraints should be addressed.
      Scoring anchor: 1 = major aspects of the request unaddressed; 10 = every element of the request is covered.
    """
