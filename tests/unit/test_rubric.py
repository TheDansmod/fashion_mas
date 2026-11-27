"""The model-quality rubric grades by catalogue ground truth (no model involved)."""

from tests.model.rubric import (
    SHOES,
    Item,
    Turn,
    all_in,
    count,
    covers,
    explanation_checks,
    grade,
    invented_types,
    mention,
)


def _turn(items, explanation="Item 1 suits you.", message="pants and shoes"):
    return Turn(message, message, [Item(c, d) for c, d in items], explanation, "", 1.0)


def test_a_turn_meeting_every_check_passes():
    turn = _turn([("PANTS", "Chinos in olive."), ("SNEAKERS", "Sneakers in white.")])
    checks = [count(2), covers("shoes", SHOES), mention(["olive", "white"])]
    result = grade(turn, [*checks, *explanation_checks()])
    assert result.passed
    assert result.score == 1.0


def test_a_failed_critical_check_fails_the_turn_despite_the_score():
    turn = _turn([("PANTS", "Chinos in olive.")] * 9 + [("TOPS", "A top.")])
    checks = [all_in({"PANTS"}), *([mention(["olive"], share=0.5)] * 20)]
    result = grade(turn, checks)
    assert result.score > 0.9
    assert not result.passed


def test_invented_types_flags_items_neither_shown_nor_asked_for():
    shown = [("PANTS", "Chinos in olive.")]
    assert invented_types(_turn(shown, "Wear Item 1 with white sneakers.")) == [
        "sneaker"
    ]
    # asked for in the message, or present in a product description: not invented
    assert not invented_types(_turn(shown, "Add sneakers.", "chinos and sneakers"))
    assert not invented_types(_turn([("PANTS", "Boot-cut jeans.")], "Boot-cut jeans."))


def test_explanation_must_speak_to_the_user():
    turn = _turn([("PANTS", "Chinos.")], "The user wants chinos.")
    assert not grade(turn, explanation_checks()).results[
        "explanation speaks to the user"
    ]
