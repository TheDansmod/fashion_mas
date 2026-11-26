"""Routers and the recommender node's handling of tool output."""

from pathlib import Path

import pytest

from frag.rag_pipeline.llm_schemas import AgentState
from tests.fakes import (
    FAKE_ITEMS,
    FIXTURE_IMAGE,
    UNKNOWN_ITEM_ID,
    FakeToolsClient,
    make_create_agent,
)


@pytest.mark.parametrize(
    ("has_images", "expected"), [(True, "intent_node"), (False, "modifier_node")]
)
def test_quantifier_router(make_agent, has_images, expected):
    state = AgentState(is_chat_start=False, has_input_images=has_images)
    assert make_agent().quantifier_node_router(state) == expected


@pytest.mark.parametrize(
    ("critique", "expected"),
    [
        ("Too formal", "modifier_node"),
        ("", "explanation_node"),
        ("   ", "explanation_node"),
    ],
)
def test_critique_router(make_agent, critique, expected):
    state = AgentState(is_chat_start=False, critique_text=critique)
    assert make_agent().critique_node_router(state) == expected


def test_critique_node_stops_at_limit(make_agent):
    agent = make_agent(max_num_critiques=2)
    state = AgentState(is_chat_start=False, num_times_critiqued=2, critique_text="old")
    assert agent.critique_node(state) == {"num_times_critiqued": 2, "critique_text": ""}


async def test_recommender_saves_item_image_and_description(make_agent, monkeypatch):
    monkeypatch.setattr(
        "frag.rag_pipeline.rag_agent.create_agent", make_create_agent(chosen_ids=[101])
    )
    agent = make_agent()
    state = AgentState(is_chat_start=False, required_clothes_descriptions=["jeans"])
    update = await agent.recommender_node(state)
    (path,) = update["recommended_clothes_image_paths"]
    assert Path(path).read_bytes() == FIXTURE_IMAGE.read_bytes()
    assert update["recommended_clothes_descriptions"] == [FAKE_ITEMS[101][1]]


async def test_recommender_skips_unknown_and_failing_items(make_agent, monkeypatch):
    # regression for cleanup.md 3.12: a hallucinated id is skipped, not fatal; so is
    # a lookup that raises (S3 or MCP error)
    monkeypatch.setattr(
        "frag.rag_pipeline.rag_agent.create_agent",
        make_create_agent(chosen_ids=[UNKNOWN_ITEM_ID, 101, 102]),
    )
    agent = make_agent(tools=FakeToolsClient(failing_ids=frozenset({101})))
    state = AgentState(
        is_chat_start=False, required_clothes_descriptions=["a", "b", "c"]
    )
    update = await agent.recommender_node(state)
    assert update["recommended_clothes_descriptions"] == [FAKE_ITEMS[102][1]]
