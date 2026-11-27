"""The agent's LangGraph state, checkpointed per conversation thread."""

from pydantic import BaseModel


class AgentState(BaseModel):
    """State shared by the graph nodes.

    One graph run handles one user message. The caller supplies `new_input_text` and
    `new_input_images`; everything else is produced by the nodes and kept in the
    checkpoint, so the next message on the same thread can refer back to it.

    Attributes:
        new_input_text: Text of the message being handled.
        new_input_images: Paths of the images attached to that message.
        input_text: The request as the rest of the graph sees it. On a follow-up
            message it is rewritten to stand on its own, folding in the earlier
            request.
        input_images_path: Images relevant to the request: the uploaded ones plus,
            on follow-ups, earlier uploads or recommendations the user referred to.
        input_images_descriptions: One description per entry in `input_images_path`.
            An empty string means "not described yet" and the vision node fills it
            in; catalogue images keep their catalogue description.
        has_input_images: Whether `input_images_path` is non-empty.
        num_recommendations: How many items to recommend (asked for by the user, or
            the configured default, and at least one per requested item type).
        requested_item_types: The kinds of products the user named (e.g. "pants",
            "shoes"); each must be covered by at least one recommendation. Empty
            when the user named none.
        vlm_instructions: What the vision model should focus on in the images,
            written by the LLM for this request.
        required_clothes_descriptions: One description per item to search for.
        recommended_clothes_descriptions: Catalogue descriptions of the items found.
        recommended_clothes_image_paths: Local copies of the images of those items.
        recommended_clothes_explanation: Why the items fit the request.
        num_times_critiqued: How many times the critique node rejected the
            recommendations in this turn; capped by `agent.max_num_critiques`.
        critique_text: The critique node's correction for the modifier node, or
            empty when the recommendations were accepted.
    """

    new_input_text: str = ""
    new_input_images: list[str] = []
    input_text: str = ""
    input_images_path: list[str] = []
    input_images_descriptions: list[str] = []
    has_input_images: bool = False
    num_recommendations: int = 0
    requested_item_types: list[str] = []
    vlm_instructions: str = ""
    required_clothes_descriptions: list[str] = []
    recommended_clothes_descriptions: list[str] = []
    recommended_clothes_image_paths: list[str] = []
    recommended_clothes_explanation: str = ""
    num_times_critiqued: int = 0
    critique_text: str = ""
