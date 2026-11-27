"""Agent nodes and routers, driven directly with fake LLM, agent factory and tools."""

import dataclasses
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from frag.agent.graph import (
    CRITIQUE_PREFIX,
    chosen_item_id,
    missing_item_types,
    offered_item_ids,
)
from frag.agent.schemas import RequiredClothes
from frag.agent.state import AgentState
from frag.tools.base import item_to_blocks
from tests.fakes import (
    FAKE_ITEMS,
    FIXTURE_IMAGE,
    UNKNOWN_ITEM_ID,
    ChatMockLLM,
    FakeToolsClient,
    fake_item,
    make_create_agent,
)


def _use_agent(monkeypatch, **kwargs):
    monkeypatch.setattr("frag.agent.graph.create_agent", make_create_agent(**kwargs))


# --- routers ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("has_images", "expected"), [(True, "intent_node"), (False, "modifier_node")]
)
def test_quantifier_router(make_agent, has_images, expected):
    state = AgentState(has_input_images=has_images)
    assert make_agent().quantifier_node_router(state) == expected


@pytest.mark.parametrize(
    ("critique", "expected"),
    [
        ("Too formal", "modifier_node"),
        ("", "explanation_node"),
        ("  ", "explanation_node"),
    ],
)
def test_critique_router(make_agent, critique, expected):
    state = AgentState(critique_text=critique)
    assert make_agent().critique_node_router(state) == expected


# --- intake, quantifier, modifier, critique -------------------------------------------


async def test_intake_first_turn_uses_the_message_and_resets_turn_fields(make_agent):
    state = AgentState(
        new_input_text="Pants for this shirt",
        new_input_images=["a.jpg"],
        num_times_critiqued=2,
        critique_text="stale",
    )
    update = await make_agent().intake_node(state, {})
    assert update == {
        "input_text": "Pants for this shirt",
        "input_images_path": ["a.jpg"],
        "input_images_descriptions": [""],
        "has_input_images": True,
        "num_times_critiqued": 0,
        "critique_text": "",
    }


async def test_quantifier_falls_back_to_default(make_agent):
    agent = make_agent(
        llm=ChatMockLLM(num_recommendations=0), default_num_recommendations=4
    )
    assert await agent.quantifier_node(AgentState(input_text="x"), {}) == {
        "num_recommendations": 4,
        "requested_item_types": [],
    }


async def test_quantifier_gives_each_named_item_type_a_recommendation(make_agent):
    llm = ChatMockLLM(
        num_recommendations=1, item_types=["Pants", " shoes", "pants", ""]
    )
    update = await make_agent(llm=llm).quantifier_node(AgentState(input_text="x"), {})
    assert update == {
        "num_recommendations": 2,
        "requested_item_types": ["pants", "shoes"],
    }


async def test_modifier_retries_until_every_item_type_is_covered(make_agent):
    llm = ChatMockLLM(
        modifier_answers=[
            [("pants", "Grey chinos."), ("pants", "Navy chinos.")],
            [("Pants", "Grey chinos."), ("shoe", "White sneakers.")],
        ]
    )
    state = AgentState(
        input_text="x", num_recommendations=2, requested_item_types=["pants", "shoes"]
    )
    update = await make_agent(llm=llm).modifier_node(state, {})
    assert len(llm.calls) == 2
    assert update["required_clothes_descriptions"] == [
        "Grey chinos.",
        "White sneakers.",
    ]
    assert "cover each one): pants, shoes" in llm.calls[0][1]


def test_missing_item_types_accepts_a_type_named_in_the_description():
    answer = RequiredClothes(
        items=[
            {"item_type": "shoes", "description": "Brown suede loafers."},
            {"item_type": "Blazer", "description": "Navy wool blazer."},
        ]
    )
    assert missing_item_types(answer, ["loafers", "blazers", "shirt"]) == ["shirt"]


async def test_modifier_keeps_the_answer_that_covers_the_types(make_agent):
    llm = ChatMockLLM(
        modifier_answers=[
            [("pants", "Grey chinos."), ("shoes", "White sneakers.")],  # covers, 2/3
            [("pants", "A."), ("pants", "B."), ("pants", "C.")],  # count only
        ]
    )
    state = AgentState(
        input_text="x", num_recommendations=3, requested_item_types=["pants", "shoes"]
    )
    agent = make_agent(llm=llm, num_recommendation_attempts=2)
    update = await agent.modifier_node(state, {})
    assert update["required_clothes_descriptions"] == [
        "Grey chinos.",
        "White sneakers.",
    ]


@pytest.mark.parametrize("critique", ["Too formal.", ""])
async def test_modifier_prefixes_critique(make_agent, critique):
    llm = ChatMockLLM()
    state = AgentState(input_text="x", num_recommendations=3, critique_text=critique)
    await make_agent(llm=llm).modifier_node(state, {})
    ((_, prompt),) = llm.calls
    assert (f"{CRITIQUE_PREFIX} Too formal." in prompt) is bool(critique)
    assert (CRITIQUE_PREFIX in prompt) is bool(critique)


async def test_modifier_retries_for_the_right_count(make_agent):
    llm = ChatMockLLM()
    state = AgentState(input_text="x", num_recommendations=5)  # mock always gives 3
    update = await make_agent(llm=llm, num_recommendation_attempts=2).modifier_node(
        state, {}
    )
    assert len(llm.calls) == 2
    assert len(update["required_clothes_descriptions"]) == 3  # the last answer


async def test_critique_node_stops_at_limit_without_calling_the_llm(make_agent):
    llm = ChatMockLLM()
    state = AgentState(num_times_critiqued=2, critique_text="old")
    update = await make_agent(llm=llm, max_num_critiques=2).critique_node(state, {})
    assert update == {"critique_text": ""}
    assert llm.calls == []


@pytest.mark.parametrize(
    ("satisfied", "expected"),
    [
        (True, {"critique_text": ""}),
        (False, {"num_times_critiqued": 1, "critique_text": "Some correction"}),
    ],
)
async def test_critique_node_counts_only_rejections(make_agent, satisfied, expected):
    agent = make_agent(llm=ChatMockLLM(critic_satisfied=satisfied))
    state = AgentState(input_text="x", recommended_clothes_descriptions=["a"])
    assert await agent.critique_node(state, {}) == expected


def _prompt_texts(llm):
    (_, messages) = llm.calls[-1]
    return " ".join(c["text"] for c in messages[0].content if c["type"] == "text")


@pytest.mark.parametrize("node", ["critique_node", "explanation_node"])
async def test_reference_images_are_labelled_only_when_present(make_agent, node):
    llm = ChatMockLLM(critic_satisfied=True)
    agent = make_agent(llm=llm)
    text_only = AgentState(input_text="a black belt")
    await getattr(agent, node)(text_only, {})
    assert "Above are the reference images" not in _prompt_texts(llm)
    assert "did not upload or pick any images" in _prompt_texts(llm)
    assert "User request: a black belt" in _prompt_texts(llm)

    with_image = AgentState(input_text="x", input_images_path=[str(FIXTURE_IMAGE)])
    await getattr(agent, node)(with_image, {})
    assert "Above are the reference images" in _prompt_texts(llm)


# --- recommender --------------------------------------------------------------------


def test_offered_item_ids_reads_only_semantic_search_results():
    blocks = [*item_to_blocks(fake_item(101)), {"type": "text", "text": "not json"}]
    messages = [
        ToolMessage(content=blocks, name="semantic_search", tool_call_id="1"),
        ToolMessage(
            content=item_to_blocks(fake_item(102)), name="other", tool_call_id="2"
        ),
        ToolMessage(
            content="Error: bad category", name="semantic_search", tool_call_id="3"
        ),
    ]
    assert offered_item_ids(messages) == {101}


async def _recommend(agent, *descriptions):
    state = AgentState(required_clothes_descriptions=list(descriptions))
    return await agent.recommender_node(state, {})


async def test_recommender_saves_item_image_and_description(make_agent, monkeypatch):
    _use_agent(monkeypatch, choices={"jeans": [101]})
    update = await _recommend(make_agent(), "jeans")
    (path,) = update["recommended_clothes_image_paths"]
    assert Path(path).read_bytes() == FIXTURE_IMAGE.read_bytes()
    assert update["recommended_clothes_descriptions"] == [FAKE_ITEMS[101][1]]


async def test_recommender_corrects_an_id_that_was_not_offered(make_agent, monkeypatch):
    # the first answer was never returned by semantic_search
    _use_agent(monkeypatch, choices={"jeans": [UNKNOWN_ITEM_ID, 101]})
    update = await _recommend(make_agent(), "jeans")
    assert update["recommended_clothes_descriptions"] == [FAKE_ITEMS[101][1]]


async def test_recommender_skips_after_a_second_invalid_id(make_agent, monkeypatch):
    _use_agent(
        monkeypatch, choices={"a": [UNKNOWN_ITEM_ID, UNKNOWN_ITEM_ID + 1], "b": [102]}
    )
    update = await _recommend(make_agent(), "a", "b")
    assert update["recommended_clothes_descriptions"] == [FAKE_ITEMS[102][1]]


async def test_recommender_rejects_answers_without_a_search(make_agent, monkeypatch):
    _use_agent(monkeypatch, choices={"a": [101]}, search=False)
    update = await _recommend(make_agent(), "a")
    assert update["recommended_clothes_image_paths"] == []


async def test_recommender_skips_failing_lookups_and_repeats(make_agent, monkeypatch):
    # regression for cleanup.md 3.12: a failing lookup is skipped, not fatal
    _use_agent(monkeypatch, choices={"a": [101], "b": [102], "c": [102], "d": [103]})
    agent = make_agent(tools=FakeToolsClient(failing_ids=frozenset({101})))
    update = await _recommend(agent, "a", "b", "c", "d")
    assert update["recommended_clothes_descriptions"] == [
        FAKE_ITEMS[102][1],  # once, though chosen twice
        FAKE_ITEMS[103][1],
    ]


async def test_recommender_shows_a_product_photographed_twice_once(
    make_agent, monkeypatch
):
    # the catalogue stores several photos of one product under different ids
    class SameProductTools(FakeToolsClient):
        async def get_item(self, item_id):
            item = await super().get_item(item_id)
            return dataclasses.replace(item, description="Black leather belt.")

    _use_agent(monkeypatch, choices={"a": [101], "b": [102]})
    update = await _recommend(make_agent(tools=SameProductTools()), "a", "b")
    assert update["recommended_clothes_descriptions"] == ["Black leather belt."]


async def test_recommender_accepts_a_prose_answer(make_agent, monkeypatch):
    # small local models often answer in text instead of the structured response
    _use_agent(monkeypatch, choices={"jeans": [103]}, prose=True)
    update = await _recommend(make_agent(), "jeans")
    assert update["recommended_clothes_descriptions"] == [FAKE_ITEMS[103][1]]


def test_chosen_item_id_from_prose_only_takes_offered_ids():
    def response(text):
        return {"messages": [AIMessage(content=text)]}

    # "10" and "2026" are not offered; 102 is
    assert (
        chosen_item_id(response("Of 10 results in 2026, 102 fits."), {101, 102}) == 102
    )
    assert chosen_item_id(response("Nothing fits."), {101}) is None
    assert chosen_item_id({"messages": []}, {101}) is None
