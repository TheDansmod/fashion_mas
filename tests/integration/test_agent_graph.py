"""End-to-end runs of the compiled agent graph with fake LLM and tools (offline)."""

from pathlib import Path

import pytest
from langchain_core.callbacks import BaseCallbackHandler

from tests.fakes import FAKE_ITEMS, FIXTURE_IMAGE, ChatMockLLM, make_create_agent


async def test_image_turn_then_follow_up(make_agent, monkeypatch):
    # follow-up turn: the LLM marks images 0 and 2 relevant, plus an out-of-range index
    monkeypatch.setattr("frag.agent.graph.create_agent", make_create_agent())
    llm = ChatMockLLM(relevant_image_indexes=(0, 2, 99))
    agent = make_agent(llm=llm, max_num_critiques=2)

    # turn 1: one uploaded image; no priming call is needed
    first = await agent.run_turn(
        "Pants that match this shirt", [str(FIXTURE_IMAGE)], "test-thread"
    )
    assert first["input_text"] == "Pants that match this shirt"
    assert first["num_recommendations"] == 3
    assert first["has_input_images"] is True
    assert first["vlm_instructions"]
    assert first["input_images_descriptions"] == [
        "A blue cotton shirt with short sleeves."
    ]
    paths = first["recommended_clothes_image_paths"]
    assert len(paths) == 3
    assert all(Path(p).is_file() for p in paths)
    assert first["recommended_clothes_descriptions"] == [
        descr for _, descr in FAKE_ITEMS.values()
    ]
    # the mock critic is never satisfied, so the loop runs exactly to its limit
    assert first["num_times_critiqued"] == 2
    assert first["critique_text"] == ""
    assert first["recommended_clothes_explanation"]

    # turn 2: text-only follow-up resolved against the previous turn
    calls_before = len(llm.calls)
    second = await agent.run_turn("Show these in red", [], "test-thread")
    assert second["input_text"] == "Show me similar items in red."
    # index 99 is ignored; index 0 is the uploaded shirt, index 2 a recommendation
    assert second["input_images_path"] == [str(FIXTURE_IMAGE), paths[1]]
    descriptions = second["input_images_descriptions"]
    assert descriptions[0] == "A blue cotton shirt with short sleeves."
    # the catalogue description of the reused recommendation is kept, not re-generated
    assert descriptions[1] == first["recommended_clothes_descriptions"][1]
    assert len(second["recommended_clothes_image_paths"]) == 3
    assert second["recommended_clothes_explanation"]
    # the critique count was reset for the new turn, so the critic ran again
    turn_two = [name for name, _ in llm.calls[calls_before:]]
    assert turn_two.count("CriticalEvaluation") == 2


@pytest.mark.usefixtures("mocked_create_agent")
async def test_text_only_turn_skips_vision(make_agent):
    llm = ChatMockLLM(critic_satisfied=True)
    result = await make_agent(llm=llm).run_turn(
        "Suggest three summer items", [], "text-only"
    )
    assert result["has_input_images"] is False
    # intent/vision never ran, so they never wrote their channels
    assert "vlm_instructions" not in result
    assert result["input_images_descriptions"] == []
    assert len(result["recommended_clothes_image_paths"]) == 3
    assert [name for name, _ in llm.calls].count("SingleImageDescription") == 0


class _NodeCounter(BaseCallbackHandler):
    def __init__(self):
        self.names = []

    def on_chain_start(self, serialized, inputs, *, name=None, **kwargs):  # noqa: ARG002
        self.names.append(name)


@pytest.mark.usefixtures("mocked_create_agent")
async def test_stream_turn_runs_to_the_end_and_passes_callbacks(make_agent):
    agent = make_agent(llm=ChatMockLLM(critic_satisfied=True))
    counter = _NodeCounter()
    chunks = [
        chunk
        async for chunk in agent.stream_turn(
            "Two summer items", [], "streamed", callbacks=[counter]
        )
    ]
    nodes = [name for chunk in chunks for name in chunk]
    assert nodes == [
        "intake_node",
        "quantifier_node",
        "modifier_node",
        "recommender_node",
        "critique_node",
        "explanation_node",
    ]
    assert "explanation_node" in counter.names  # the callbacks reached the run
    # no interrupt: the thread is finished and waiting for the next message
    snapshot = await agent._graph.aget_state(  # noqa: SLF001
        {"configurable": {"thread_id": "streamed"}}
    )
    assert snapshot.next == ()
