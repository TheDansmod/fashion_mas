"""Prompt templates for the agent nodes.

These are constants, not configuration: they do not vary by environment. Placeholders
in braces are filled with `str.format`.

They are written for small local models as well as Claude: short numbered rules,
one task per prompt, and the user's request stated as the thing that wins over any
other input (critique feedback, reference images).
"""

# Intake node, follow-up messages: merge the new message with the previous turn.
HUMAN_INPUT_UPDATE_PROMPT = """\
You are a fashion assistant in an ongoing conversation. Rewrite the user's new message \
as one standalone request, and pick which earlier images still matter.

Earlier images are listed below with an index:
- "Original images" are the images the user uploaded before.
- "Recommended items" are the products you showed the user last time, in the order \
they were shown: the first one listed is "the first item", and so on.
Any image attached to this message is new and always relevant; it has no index.

Rules:
1. relevant_image_indexes: the indexes of the earlier images the new request refers \
to or depends on. Examples: "I like the second one, show me more like it" -> only the \
second recommended item; "show these in red" -> all recommended items; "instead of \
pants, show me hats for my shirt" -> the original shirt image only. Use an empty list \
if none are relevant.
2. updated_user_query: a complete request that can be understood without the \
earlier conversation. Keep every constraint from the original request (item types, \
occasion, colours, budget, style) unless the new message changes or drops it, and add \
what the new message asks for. Name the item types explicitly. Example: original \
"pants and shoes for a casual weekend with this shirt", new "make the shoes \
sneakers" -> "Suggest pants and sneakers for a casual weekend that go with the \
uploaded shirt."

Original images:
{original_image_descriptions}

Original request:
{original_text_input}

Recommended items:
{response_image_descriptions}

New message:
{new_text_input}
"""

# Quantifier node: how many items the user asked for (0 if unspecified), and which
# item types they named.
NUM_RECOMMENDATIONS_PROMPT = """\
Read this fashion request and answer two questions.

1. num_recommendations: how many products does it ask for? Give the number only if \
the user states it explicitly, as a number or a number word ("show me 2 dresses" -> \
2, "three options" -> 3, "one belt and one tie" -> 2). Otherwise 0, including for \
"a dress for a wedding", "pants and shoes" or "a complete outfit".
2. item_types: which kinds of products does the user explicitly name? Use their \
words, one entry per kind: "pants and shoes for this shirt" -> ["pants", "shoes"]; \
"2 black belts" -> ["belts"]; "find something similar to this" or "a complete \
outfit" -> []. Don't list items the user already has (the shirt in the first example).

Request:
{user_request}
"""

# Intent node: instructions telling the vision model what to look at.
REQUEST_FOCUS_PROMPT = """\
A vision model will describe the user's images so that matching products can be \
searched for later. Write the instructions for that vision model: what it should \
look at and describe, given the user's request below. It describes the images; it \
does not answer the request.

Always ask for the type of each clothing item or accessory and its main colours. If \
the request is about one item ("something like this"), the vision model should \
describe the most prominent item in detail and mention anything else worn in the \
photo only briefly. Then add the details this request depends on. Examples: "find something like this" -> \
cut, fabric, pattern and distinctive details; "same colour palette" -> every colour \
and how they are combined; "what goes with this" -> colour, pattern, formality and \
style.

Reply with the instructions only, in at most 5 sentences, with no explanation of why.

User request:
{user_request}
"""

# Vision node: describe one uploaded image following the intent instructions.
VISION_PROMPT = """\
You are a fashion expert. Describe the image following the instructions below. \
Describe only what is visible; do not recommend anything. Use plain text without \
markdown, at most 5 sentences.

Instructions:
{image_focus_instructions}
"""

# Shared by both modifier prompts.
_MODIFIER_RULES = """\
{item_types}

Rules:
1. Write exactly {num_recommendations} items, each describing exactly one product.
2. Cover every item type the user named at least once, and set each item's \
item_type to the named type it belongs to. Example: types "pants, shoes" with 3 \
items -> at least one item of type "pants" and at least one of type "shoes". Never \
recommend an item type the user already has in the reference items unless they ask \
for it.
3. Follow every explicit constraint in the request (colour, material, fit, occasion, \
style, price). If the user asks for something like a reference item, every item is \
the same type as the main reference item (other things worn in the photo are not \
part of the request). If they ask for it changed (for example "the same but in \
black"), keep its other details and change only that.
4. Write each description like a product listing in an online catalogue: the item \
type first, then colour, material, fit and notable details, in one or two short \
sentences. Example: "Slim-fit chino trousers in olive cotton twill. Mid-rise, four \
pockets, tapered leg." Commit to one colour per item (not "white or grey").
5. Colours: if the user names a colour, use exactly their word in the description \
("red", never "crimson", "ruby" or "scarlet"); catalogue listings use plain colour \
names.
6. Make the descriptions different from each other. Describe the products only, not \
why you chose them.
7. If feedback on a previous attempt is given, use it to fix the problem it names, \
but never against the user's request: the request always wins.
"""

# Modifier node, when the user supplied images (or picked earlier recommendations).
MODIFIER_IMAGES_PRESENT_PROMPT = (
    """\
You are a fashion stylist. Describe the products to search the catalogue for, so \
that together they satisfy the user's request. The reference items are what the user \
showed or picked; use them as the request says (to match, to complement, or as a \
starting point to change).

"""
    + _MODIFIER_RULES
    + """
Reference items:
{reference_descriptions}

User request:
{user_request}

{critique_node_correction}
"""
)

# Modifier node, text-only requests.
MODIFIER_IMAGES_ABSENT_PROMPT = (
    """\
You are a fashion stylist. Describe the products to search the catalogue for, so \
that together they satisfy the user's request.

"""
    + _MODIFIER_RULES
    + """
User request:
{user_request}

{critique_node_correction}
"""
)

# Recommender node: tool-using agent that finds the best catalogue match.
MATCH_CLOTHES_PROMPT = """\
Find the catalogue product that best matches the description below.

1. Call get_product_categories once.
2. Pick the one to three categories the product belongs to (shoes, for example, may \
be "SNEAKERS" or "LACE UPS").
3. Call semantic_search with the description, those categories and num_matches \
between 3 and 6.
4. Compare the returned images and metadata with the description. The product must \
be the item type described; colour and material come next. If none of them fit, you \
may search once more with a rephrased description or other categories.
5. Answer with the id of the best product. It must be an id that semantic_search \
returned.

Description:
{item_description}
"""

# Critique node: Self-RAG style yes/no check plus a short correction.
CRITIQUE_PROMPT = """\
You are checking a fashion assistant's recommendations before they are shown to the \
user. Judge them against the user's request above, not against the reference images: \
the request often asks for something different from the references (another colour, \
an item that goes with them, a different style).

Answer "No" only for a clear failure:
- there are fewer items than the number the user asked for;
- an item type the user named (listed above) has no item, or an item is a type they \
did not ask for;
- an item contradicts an explicit constraint of the request (colour, material, \
occasion, style);
- for "goes with" requests, an item clearly clashes with the reference items.
Otherwise answer "Yes". Matters of taste are not failures.

If the answer is "No", write the correction as at most 3 sentences saying what to \
search for instead, and restate every item type the request needs.
"""

# Explanation node: why the recommendations satisfy the request.
EXPLANATION_PROMPT = """\
You are a fashion stylist talking to the user. In at most 150 words, explain how the \
recommended items above meet their request. Speak to the user as "you" and refer to \
the items as Item 1, Item 2 and so on. Mention only the recommended items shown \
above; do not suggest or describe anything else. If an item meets the request only \
partly, say so briefly. Start directly with the explanation, without a preamble.
"""
