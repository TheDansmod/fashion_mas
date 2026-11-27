"""Model-quality tests: the real agent on requests with catalogue ground truth.

Opt-in and slow: `uv run pytest -m ollama` (see tests/model/conftest.py for what must
be running). Every requested product exists in the catalogue in quantity (counted in
the metadata when the cases were written), so a failure points at the model or the
prompts rather than at an empty catalogue. Each conversation is graded with the rubric
in tests/model/rubric.py. FRAG_MODEL_TEST_REPEATS=N runs each case N times; a case
passes if at least half of its runs pass.
"""

import math

import pytest

from tests.model.harness import REPEATS, ModelHarness, record
from tests.model.rubric import (
    BROWN,
    LEATHER,
    PANTS,
    RED,
    SHOES,
    Check,
    all_in,
    at_least,
    count,
    covers,
    explanation_checks,
    grade,
    mention,
    same_product,
)

pytestmark = pytest.mark.ollama

# catalogue products used as uploaded images
GINGHAM_SHIRT = 9067  # "... button down gingham shirt in blue. Blue and white gingham"
RED_SNEAKERS = 93191  # "Low-top buffed leather sneakers in red. ..." (product shot)

Messages = list[tuple[str, list[str]]]


async def _check_case(
    harness: ModelHarness, case: str, messages: Messages, checks: list[list[Check]]
) -> None:
    passes, failures = 0, []
    for run in range(1, REPEATS + 1):
        turns = await harness.converse(messages)
        grades = [
            grade(turn, [*turn_checks, *explanation_checks()])
            for turn, turn_checks in zip(turns, checks, strict=True)
        ]
        record(case, run, turns, grades)
        if all(g.passed for g in grades):
            passes += 1
        else:
            failures.append(
                [
                    (g.score, [n for n, ok in g.results.items() if not ok], t.items)
                    for t, g in zip(turns, grades, strict=True)
                ]
            )
    assert passes >= math.ceil(REPEATS / 2), failures


async def test_text_request_with_count_and_attributes(harness):
    await _check_case(
        harness,
        "2 black leather belts",
        [("Show me 2 black leather belts.", [])],
        [
            [
                count(2),
                all_in({"BELTS & SUSPENDERS"}),
                mention(["black"], weight=2),
                mention(LEATHER),
            ]
        ],
    )


async def test_text_request_for_several_item_types(harness):
    blazers = {"SUITS & BLAZERS", "JACKETS & COATS"}
    loafers = {"LOAFERS", "BOAT SHOES & MOCCASINS"}
    shirts = {"SHIRTS", "TOPS"}  # the catalogue files some shirts under TOPS
    await _check_case(
        harness,
        "wedding outfit: navy blazer, white shirt, brown loafers",
        [
            (
                "I'm going to a summer wedding. Suggest a navy blazer, a white shirt "
                "and brown leather loafers.",
                [],
            )
        ],
        [
            [
                covers("blazer", blazers),
                covers("shirt", shirts),
                covers("loafers", loafers),
                covers("blazer", blazers, ["navy"], weight=2),
                covers("shirt", shirts, ["white"], weight=2),
                covers("loafers", loafers, BROWN, weight=2),
            ]
        ],
    )


async def test_image_request_for_complementary_items(harness):
    shirt = await harness.catalogue_image(GINGHAM_SHIRT)
    await _check_case(
        harness,
        "pants and shoes for an uploaded gingham shirt",
        [
            (
                "Suggest some pants and shoes that go well with this shirt for a "
                "casual weekend.",
                [shirt],
            )
        ],
        [
            [
                at_least(2),
                covers("pants", PANTS),
                covers("shoes", SHOES),
                all_in(PANTS | SHOES),
            ]
        ],
    )


async def test_image_request_for_similar_items(harness):
    sneakers = await harness.catalogue_image(RED_SNEAKERS)
    description = harness.metadata[RED_SNEAKERS]["description"]
    await _check_case(
        harness,
        "items similar to uploaded red leather sneakers",
        [("Find me something similar to this.", [sneakers])],
        [
            [
                all_in({"SNEAKERS"}),
                mention(RED, share=0.6, weight=2),
                mention(LEATHER, share=0.3, weight=0.5),
                same_product(description),
            ]
        ],
    )


async def test_follow_up_changes_the_colour(harness):
    await _check_case(
        harness,
        "white sneakers, then the same style in black",
        [
            ("Show me 2 white leather sneakers.", []),
            ("I like these. Show me the same style in black.", []),
        ],
        [
            [count(2), all_in({"SNEAKERS"}), mention(["white"], weight=2)],
            [all_in({"SNEAKERS"}), mention(["black"], share=0.6, weight=2)],
        ],
    )


async def test_follow_up_switches_the_item_type(harness):
    shirt = await harness.catalogue_image(GINGHAM_SHIRT)
    await _check_case(
        harness,
        "pants for an uploaded shirt, then hats instead",
        [
            ("Suggest pants that go well with this shirt.", [shirt]),
            ("Instead of pants, show me hats that would go with my shirt.", []),
        ],
        [[all_in(PANTS)], [all_in({"HATS"})]],
    )


async def test_text_request_with_count_occasion_and_colour(harness):
    await _check_case(
        harness,
        "3 red cocktail dresses",
        [("Give me 3 options for a red dress for a cocktail party.", [])],
        [[count(3), all_in({"DRESSES"}), mention(RED, weight=2)]],
    )
