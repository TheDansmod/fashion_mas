"""The evaluation pipeline end to end with fake LLM and tools (offline)."""

import dataclasses
import json
import shutil
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from frag.agent.graph import FashionAgent
from frag.evaluation import run as evaluation
from frag.runtime import Resources
from frag.settings import LLMSettings
from tests.fakes import FIXTURE_IMAGE, ChatMockLLM, FakeToolsClient, make_create_agent


@pytest.fixture
def resources(tmp_path, clean_env, monkeypatch):
    """Two queries (text only, then one with an image) over fake LLM and tools."""
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps([["Black belts"], ["Pants for this shirt"]]))
    images = tmp_path / "eval_images"
    images.mkdir()
    shutil.copyfile(FIXTURE_IMAGE, images / "02_01.jpg")
    settings = clean_env(
        token_usage_csv=None,
        agent={"temp_dir": str(tmp_path / "temp_images"), "max_num_critiques": 1},
        evaluation={
            "queries_path": str(queries),
            "images_dir": str(images),
            "runs_dir": str(tmp_path / "runs"),
        },
    )
    monkeypatch.setattr("frag.agent.graph.create_agent", make_create_agent())

    async def collection_size(_settings):
        return 3

    monkeypatch.setattr(evaluation, "_collection_size", collection_size)
    monkeypatch.setattr(evaluation, "_git_commit", lambda: "abc123")
    llm = ChatMockLLM(critic_satisfied=True)
    return Resources(
        settings=settings,
        llm=llm,
        judge_llm=llm,
        checkpointer=InMemorySaver(),
        tools=FakeToolsClient(),
    )


def _read(run_dir: Path, name: str):
    return evaluation.read_jsonl(run_dir / name)


async def test_a_run_is_self_contained_and_judged(resources):
    run_dir = await evaluation.evaluate(resources)

    config = json.loads((run_dir / evaluation.CONFIG).read_text())
    assert config["agent_model"] == config["judge_model"] == "ollama:qwen3.5:4b"
    assert (config["collection"], config["collection_points"]) == ("fashion_gen", 3)
    assert config["git_commit"] == "abc123"

    results = _read(run_dir, evaluation.RESULTS)
    assert [(r["key"], r["status"]) for r in results] == [
        ("01_01", "ok"),
        ("02_01", "ok"),
    ]
    assert results[0]["input_images"] == []
    assert results[1]["input_images"] == ["images/02_01_input_1.jpg"]
    for record in results:
        assert record["agent_model"] == "ollama:qwen3.5:4b"
        assert record["recommendations"]
        for rec in record["recommendations"]:
            # paths are relative to the run directory, and the files are copies
            assert not Path(rec["image"]).is_absolute()
            assert (run_dir / rec["image"]).is_file()

    judgements = _read(run_dir, evaluation.JUDGEMENT)
    assert [j["status"] for j in judgements] == ["ok", "ok"]
    # visual grounding counts only for the query with an image
    assert [j["final_score"] for j in judgements] == [8.25, 8.2]

    summary = json.loads((run_dir / evaluation.SUMMARY).read_text())
    assert (summary["completed"], summary["total"], summary["judged"]) == (2, 2, 2)
    assert summary["score"]["n"] == 2
    assert set(summary["groups"]) == {"1", "2"}


async def test_failures_are_counted_not_dropped(resources, monkeypatch):
    run_turn = FashionAgent.run_turn

    async def failing_run_turn(self, text, *args, **kwargs):
        if text == "Black belts":
            msg = "model unavailable"
            raise RuntimeError(msg)
        return await run_turn(self, text, *args, **kwargs)

    monkeypatch.setattr(FashionAgent, "run_turn", failing_run_turn)
    run_dir = await evaluation.evaluate(resources)

    first = _read(run_dir, evaluation.RESULTS)[0]
    assert (first["status"], first["error"]) == (
        "error",
        "RuntimeError: model unavailable",
    )
    judgements = _read(run_dir, evaluation.JUDGEMENT)
    assert [j["status"] for j in judgements] == ["skipped", "ok"]
    summary = evaluation.summarise(run_dir)
    assert (summary["completed"], summary["total"]) == (1, 2)
    assert summary["run_status"] == {"error": 1, "ok": 1}
    assert summary["groups"]["1"]["completed"] == 0


async def test_judge_errors_are_recorded(resources, monkeypatch):
    async def broken_judge(*_args, **_kwargs):
        msg = "bad JSON"
        raise ValueError(msg)

    run_dir = await evaluation.evaluate(resources, judge=False, limit=1)
    assert not (run_dir / evaluation.JUDGEMENT).exists()

    class BrokenModel:
        ainvoke = staticmethod(broken_judge)

    monkeypatch.setattr(
        resources.judge_llm, "with_structured_output", lambda _schema: BrokenModel()
    )
    await evaluation.evaluate(resources, run=False)  # judges the latest run
    (judgement,) = _read(run_dir, evaluation.JUDGEMENT)
    assert (judgement["status"], judgement["error"]) == (
        "error",
        "ValueError: bad JSON",
    )
    assert evaluation.summarise(run_dir)["judge_errors"] == 1


async def test_a_stored_run_is_not_overwritten(resources):
    run_dir = await evaluation.evaluate(resources, judge=False, limit=1)
    with pytest.raises(FileExistsError):
        await evaluation.evaluate(resources, run_dir=str(run_dir))


async def test_the_judge_can_be_a_different_llm(resources):
    judge = ChatMockLLM()
    settings = resources.settings.model_copy(
        update={
            "evaluation": resources.settings.evaluation.model_copy(
                update={"judge_llm": LLMSettings(provider="bedrock")}
            )
        }
    )
    resources = dataclasses.replace(resources, settings=settings, judge_llm=judge)
    run_dir = await evaluation.evaluate(resources, limit=1)

    config = json.loads((run_dir / evaluation.CONFIG).read_text())
    assert config["agent_model"] == "ollama:qwen3.5:4b"
    assert config["judge_model"].startswith("bedrock:us.anthropic.claude-haiku")
    (judgement,) = _read(run_dir, evaluation.JUDGEMENT)
    assert judgement["judge_model"] == config["judge_model"]
    # only the judge model was asked to judge
    assert [name for name, _ in judge.calls] == ["RecommendationEvaluation"]
    assert "RecommendationEvaluation" not in [n for n, _ in resources.llm.calls]
