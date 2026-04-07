"""The per-node updates to be presented in the ChainLit UI."""

def intent_node_update(update):
    instructions = update.get("vlm_instructions", "No instructions")
    return f"Visual Focus Instructions: {instructions}"

def vision_node_update(update):
    tabbed_descriptions = [f"\t- {descr}" for descr in update.get("input_images_descriptions", [])]
    prefix = "Input Image Descriptions:\n"
    descriptions = "\n".join(tabbed_descriptions)
    result = f"{prefix}{descriptions}"
    return result

def modifier_node_update(update):
    tabbed_descriptions = [f"\t- {descr}" for descr in update.get("required_clothes_descriptions", [])]
    prefix = "Searching for clothes with following descriptions:\n"
    descriptions = "\n".join(tabbed_descriptions)
    result = f"{prefix}{descriptions}"
    return result

def critique_node_update(update):
    if update.get("critique_text", ""):
        return f"Existing recommendations unsatisfactory. Regenerating descriptions.\nCritique: {update.get('critique_text', '?')}"
    else:
        return "Existing recommendations found to be satisfactory. Continuing."

# the names show up with the `Used` prefix on chainlit, so need to adapt accordingly
NODE_META = {
    "human_node": (
        "Start Node ❯❯❯❯",
        lambda u: f"Synthesizing Inputs"
    ),
    "quantifier_node": (
        "Quantifier Node 🔢",
        lambda u: f"Looking for **{u.get('num_recommendations', '?')}** item(s).",
    ),
    "intent_node": (
        "Intent Node 🎯",
        lambda u: intent_node_update(u),
    ),
    "vision_node": (
        "Vision Node 👁️",
        lambda u: vision_node_update(u),
    ),
    "modifier_node": (
        "Description Generator Node ✏️",
        lambda u: modifier_node_update(u),
    ),
    "recommender_node": (
        "Product Catalogue Search 🔍",
        lambda u: f"Matched **{len(u.get('recommended_clothes_image_paths', []))}** item(s) from the Product Catalogue.",
    ),
    "critique_node": (
        "Critique Node 🤔",
        lambda u: critique_node_update(u),
    ),
    "explanation_node": (
        "Explanation Node 💬",
        lambda u: "Generated recommendation rationale.",
    ),
}
