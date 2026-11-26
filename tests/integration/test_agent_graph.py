"""End-to-end run of the compiled agent graph with fake LLM and tools (offline)."""

from pathlib import Path

import pytest
from langgraph.types import Command

from tests.fakes import FAKE_ITEMS, FIXTURE_IMAGE, make_create_agent


async def _turn(agent, config, text, images):
    return await agent.ainvoke(
        Command(resume={"input_text": text, "input_images_path": images}), config
    )


async def test_image_turn_then_follow_up(make_agent, monkeypatch):
    # follow-up turn: the LLM marks images 0 and 2 relevant, plus an out-of-range index
    monkeypatch.setattr(
        "frag.rag_pipeline.rag_agent.create_agent",
        make_create_agent(relevant_image_indexes=(0, 2, 99)),
    )
    agent = make_agent(max_num_critiques=2)
    config = {"configurable": {"thread_id": "test-thread"}}
    await agent.ainvoke({"is_chat_start": True}, config)

    # turn 1: one uploaded image
    first = await _turn(
        agent, config, "Pants that match this shirt", [str(FIXTURE_IMAGE)]
    )
    assert first["num_recommendations"] == 3
    assert first["has_input_images"] is True
    assert first["vlm_instructions"]
    assert first["input_images_descriptions"] == [
        "A blue cotton shirt with short sleeves."
    ]
    paths = first["recommended_clothes_image_paths"]
    assert len(paths) == 3
    assert all(Path(p).is_file() for p in paths)
    assert set(first["recommended_clothes_descriptions"]) == {
        descr for _, descr in FAKE_ITEMS.values()
    }
    # the mock critic is never satisfied, so the loop runs exactly to its limit
    assert first["num_times_critiqued"] == 2
    assert first["critique_text"] == ""
    assert first["recommended_clothes_explanation"]

    # turn 2: text-only follow-up resolved against the previous turn
    second = await _turn(agent, config, "Show these in red", [])
    assert second["input_text"] == "Show me similar items in red."
    # index 99 is ignored; index 0 is the uploaded shirt, index 2 a recommendation
    assert second["input_images_path"] == [str(FIXTURE_IMAGE), paths[1]]
    descriptions = second["input_images_descriptions"]
    assert descriptions[0] == "A blue cotton shirt with short sleeves."
    # the catalogue description of the reused recommendation is kept, not re-generated
    assert descriptions[1] == first["recommended_clothes_descriptions"][1]
    assert len(second["recommended_clothes_image_paths"]) == 3
    assert second["recommended_clothes_explanation"]


@pytest.mark.usefixtures("mocked_create_agent")
async def test_text_only_turn_skips_vision(make_agent):
    agent = make_agent()
    config = {"configurable": {"thread_id": "text-only"}}
    await agent.ainvoke({"is_chat_start": True}, config)
    result = await _turn(agent, config, "Suggest three summer items", [])
    assert result["has_input_images"] is False
    # intent/vision never ran, so they never wrote their channels
    assert "vlm_instructions" not in result
    assert (
        "input_images_descriptions" not in result
        or result["input_images_descriptions"] == []
    )
    assert len(result["recommended_clothes_image_paths"]) == 3
