"""How each graph node's state update is summarised as a step in the Chainlit UI."""

from collections.abc import Callable
from typing import Any

Update = dict[str, Any]


def _bullets(prefix: str, items: list[str]) -> str:
    return prefix + "\n".join(f"\t- {item}" for item in items)


def _critique(update: Update) -> str:
    if critique := update.get("critique_text", ""):
        return (
            "Existing recommendations unsatisfactory. Regenerating descriptions.\n"
            f"Critique: {critique}"
        )
    return "Existing recommendations found to be satisfactory. Continuing."


# node name -> (step label, summary of the node's update). Chainlit shows the labels
# with a "Used" prefix.
NODE_META: dict[str, tuple[str, Callable[[Update], str]]] = {
    "intake_node": (
        "Intake Node \N{RIGHTWARDS ARROW}",
        lambda _u: "Synthesizing inputs",
    ),
    "quantifier_node": (
        "Quantifier Node 🔢",
        lambda u: (
            f"Looking for **{u.get('num_recommendations', '?')}** item(s)"
            + (
                f": {', '.join(u['requested_item_types'])}."
                if u.get("requested_item_types")
                else "."
            )
        ),
    ),
    "intent_node": (
        "Intent Node 🎯",
        lambda u: f"Visual Focus Instructions: {u.get('vlm_instructions', '')}",
    ),
    "vision_node": (
        "Vision Node 👁️",
        lambda u: _bullets(
            "Input Image Descriptions:\n", u.get("input_images_descriptions", [])
        ),
    ),
    "modifier_node": (
        "Description Generator Node ✏️",
        lambda u: _bullets(
            "Searching for clothes with following descriptions:\n",
            u.get("required_clothes_descriptions", []),
        ),
    ),
    "recommender_node": (
        "Product Catalogue Search 🔍",
        lambda u: (
            f"Matched **{len(u.get('recommended_clothes_image_paths', []))}** "
            "item(s) from the Product Catalogue."
        ),
    ),
    "critique_node": ("Critique Node 🤔", _critique),
    "explanation_node": (
        "Explanation Node 💬",
        lambda _u: "Generated recommendation rationale.",
    ),
}
