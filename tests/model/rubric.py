"""Rubric grading for the model-quality tests.

Model output varies from run to run, so a turn is not compared with one expected
answer. Instead each turn is graded against weighted checks that use catalogue ground
truth: the category and the description of every recommended product come from the
catalogue metadata, not from the model. A turn passes if every critical check passes
and the weighted score reaches PASS_SCORE.
"""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

PASS_SCORE = 0.8

PANTS = frozenset({"PANTS", "JEANS", "SHORTS"})
SHOES = frozenset(
    {
        "SNEAKERS",
        "LACE UPS",
        "LOAFERS",
        "BOOTS",
        "BOAT SHOES & MOCCASINS",
        "ESPADRILLES",
        "SANDALS",
        "MONKSTRAPS",
        "FLATS",
        "HEELS",
    }
)

# catalogue descriptions name materials and colours precisely, so a request word is
# matched by any of these
LEATHER = ("leather", "calfskin", "lambskin", "cowhide", "goatskin", "nappa", "suede")
BROWN = ("brown", "tan", "tobacco", "cognac", "chocolate", "camel", "chestnut")
RED = ("red", "scarlet", "crimson", "cherry", "ruby")


@dataclass(frozen=True)
class Item:
    """A recommended product, with its catalogue category."""

    category: str
    description: str


@dataclass
class Turn:
    """What one user message produced."""

    message: str
    request: str  # the request after the intake node (rewritten on follow-ups)
    items: list[Item]
    explanation: str
    reference_text: str  # descriptions of the reference images, if any
    seconds: float
    notes: list[str] = field(default_factory=list)  # agent log lines, for the report


@dataclass(frozen=True)
class Check:
    """One rubric line: a named yes/no test on a turn."""

    name: str
    test: Callable[[Turn], bool]
    weight: float = 1.0
    critical: bool = False


@dataclass
class Grade:
    """The result of grading one turn."""

    score: float
    passed: bool
    results: dict[str, bool] = field(default_factory=dict)


def grade(turn: Turn, checks: Iterable[Check]) -> Grade:
    """Score `turn` against `checks`."""
    checks = list(checks)
    results = {c.name: bool(c.test(turn)) for c in checks}
    total = sum(c.weight for c in checks)
    score = sum(c.weight for c in checks if results[c.name]) / total
    critical_ok = all(results[c.name] for c in checks if c.critical)
    return Grade(round(score, 3), critical_ok and score >= PASS_SCORE, results)


def _mentions(text: str, words: Iterable[str]) -> bool:
    return any(re.search(rf"\b{re.escape(w)}", text, re.IGNORECASE) for w in words)


# --- check factories ------------------------------------------------------------------


def count(n: int) -> Check:
    """Exactly `n` products (the user asked for a number)."""
    return Check(f"count == {n}", lambda t: len(t.items) == n, 2, critical=True)


def at_least(n: int) -> Check:
    """At least `n` products."""
    return Check(f"count >= {n}", lambda t: len(t.items) >= n, 2, critical=True)


def all_in(categories: Iterable[str]) -> Check:
    """Every product is in one of `categories`."""
    cats = frozenset(categories)
    return Check(
        f"all in {sorted(cats)}",
        lambda t: bool(t.items) and all(i.category in cats for i in t.items),
        2,
        critical=True,
    )


def covers(
    label: str,
    categories: Iterable[str],
    words: Iterable[str] = (),
    weight: float | None = None,
) -> Check:
    """At least one product in `categories` (that mentions one of `words`, if given)."""
    cats, words = frozenset(categories), tuple(words)

    def test(t: Turn) -> bool:
        return any(
            i.category in cats and (not words or _mentions(i.description, words))
            for i in t.items
        )

    name = f"has {label}" + (f" ({'/'.join(words)})" if words else "")
    # coverage of an item type is critical; its colour is weighted, not critical
    weight = weight if weight is not None else (2 if not words else 1)
    return Check(name, test, weight, critical=not words)


def mention(words: Iterable[str], share: float = 1.0, weight: float = 1.0) -> Check:
    """At least `share` of the products mention one of `words` in their description."""
    words = tuple(words)

    def test(t: Turn) -> bool:
        if not t.items:
            return False
        hits = sum(_mentions(i.description, words) for i in t.items)
        return hits / len(t.items) >= share

    return Check(f"{share:.0%} mention {'/'.join(words)}", test, weight)


def none_in(categories: Iterable[str]) -> Check:
    """No product is in `categories` (e.g. the type the user already has)."""
    cats = frozenset(categories)
    return Check(
        f"none in {sorted(cats)}",
        lambda t: not any(i.category in cats for i in t.items),
        1,
    )


def same_product(description: str) -> Check:
    """The exact catalogue product is among the results (a bonus, not required)."""
    return Check(
        "found the exact product",
        lambda t: any(i.description == description for i in t.items),
        0.5,
    )


# item-type words an explanation might use, and the categories they belong to; kept
# to unambiguous words ("tie", "cap", "heel" have other meanings in fashion text)
_TYPE_WORDS = {
    "sneaker": {"SNEAKERS"},
    "trainer": {"SNEAKERS"},
    "boot": {"BOOTS"},
    "loafer": {"LOAFERS", "BOAT SHOES & MOCCASINS"},
    "sandal": {"SANDALS", "ESPADRILLES"},
    "oxford": {"LACE UPS"},
    "belt": {"BELTS & SUSPENDERS"},
    "hat": {"HATS"},
    "beanie": {"HATS"},
    "scarf": {"SCARVES"},
    "backpack": {"BACKPACKS"},
    "sunglasses": {"EYEWEAR"},
    "blazer": {"SUITS & BLAZERS", "JACKETS & COATS"},
    "jeans": {"JEANS", "PANTS"},
    "chino": {"PANTS", "SHORTS"},
    "shorts": {"SHORTS"},
    "skirt": {"SKIRTS"},
    "sweater": {"SWEATERS"},
}


def invented_types(turn: Turn) -> list[str]:
    """Item types the explanation mentions that were neither shown nor asked for."""
    shown = {i.category for i in turn.items}
    context = " ".join(
        [turn.message, turn.request, turn.reference_text]
        + [i.description for i in turn.items]
    )
    invented = []
    for word, cats in _TYPE_WORDS.items():
        # "boot" must not match "boot-cut"
        pattern = rf"\b{word}(s|es)?\b(?!-)"
        if not re.search(pattern, turn.explanation, re.IGNORECASE):
            continue
        if cats & shown or re.search(pattern, context, re.IGNORECASE):
            continue
        invented.append(word)
    return invented


def explanation_grounded() -> Check:
    """The explanation only talks about what was recommended or asked for."""
    return Check("explanation invents no items", lambda t: not invented_types(t), 1)


def explanation_addresses_user() -> Check:
    """The explanation speaks to the user, not about "the user"."""
    return Check(
        "explanation speaks to the user",
        lambda t: (
            bool(t.explanation.strip()) and "the user" not in t.explanation.lower()
        ),
        0.5,
    )


def explanation_checks() -> list[Check]:
    """Checks that apply to every turn."""
    return [explanation_grounded(), explanation_addresses_user()]
