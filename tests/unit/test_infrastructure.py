"""Checkpointers, LLM factory, token tracking, messages and evaluation helpers."""

import base64
import csv
import json
import os

import psycopg
import pytest
from langgraph.checkpoint.base import empty_checkpoint
from PIL import Image

from frag.checkpointer import open_checkpointer
from frag.evaluation.run import get_evaluation_inputs, get_final_score
from frag.evaluation.schemas import RecommendationEvaluation
from frag.llm import build_llm
from frag.messages import build_message
from frag.settings import CheckpointerSettings, EvalWeights, LLMSettings
from frag.tokens import CSV_HEADER, append_usage_csv, track_tokens
from tests.fakes import FIXTURE_IMAGE

# --- checkpointers -------------------------------------------------------------------


async def _roundtrip(settings: CheckpointerSettings) -> None:
    config = {"configurable": {"thread_id": "pytest-thread", "checkpoint_ns": ""}}
    async with open_checkpointer(settings, "us-east-1") as saver:
        checkpoint = empty_checkpoint()
        saved = await saver.aput(
            config, checkpoint, {"source": "input", "step": -1}, {}
        )
        assert (await saver.aget_tuple(saved)).checkpoint["id"] == checkpoint["id"]
        await saver.adelete_thread("pytest-thread")
        assert await saver.aget_tuple(config) is None


async def test_memory_checkpointer():
    await _roundtrip(CheckpointerSettings(backend="memory"))


async def test_sqlite_checkpointer_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "checkpoints.db"
    await _roundtrip(CheckpointerSettings(backend="sqlite", sqlite_path=str(path)))
    assert path.is_file()


def _postgres_available(dsn: str) -> bool:
    try:
        psycopg.connect(dsn, connect_timeout=1).close()
    except psycopg.OperationalError:
        return False
    return True


async def test_postgres_checkpointer():
    settings = CheckpointerSettings(backend="postgres")
    if not _postgres_available(settings.postgres_dsn):
        pytest.skip("postgres not running (podman compose --profile postgres up -d)")
    await _roundtrip(settings)


@pytest.mark.aws
async def test_dynamodb_checkpointer():
    """Opt-in (`pytest -m aws`): a few requests against the real checkpointer table."""
    from dotenv import load_dotenv  # noqa: PLC0415  only needed for this opt-in test

    load_dotenv()
    table = os.environ["CHECKPOINTER__DYNAMODB_TABLE"]
    await _roundtrip(CheckpointerSettings(backend="dynamodb", dynamodb_table=table))


# --- LLM factory ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "class_name"),
    [("bedrock", "ChatBedrockConverse"), ("ollama", "ChatOllama")],
)
def test_build_llm_selects_provider(provider, class_name):
    llm = build_llm(
        LLMSettings(provider=provider, model_id="some-model", rate_limit_rps=1.0),
        "us-east-1",
    )
    assert type(llm).__name__ == class_name
    assert llm.rate_limiter is not None


def test_build_llm_without_rate_limit():
    llm = build_llm(
        LLMSettings(provider="ollama", model_id="m", rate_limit_rps=None), "x"
    )
    assert llm.rate_limiter is None


# --- token tracking ------------------------------------------------------------------


def _usage(n):
    return {"model-a": {"input_tokens": n, "output_tokens": 1, "total_tokens": n + 1}}


def test_append_usage_csv(tmp_path):
    path = tmp_path / "sub" / "usage.csv"
    append_usage_csv(str(path), _usage(5), "agent", "thread-1")
    append_usage_csv(str(path), {})  # nothing to write
    append_usage_csv(str(path), _usage(7))
    header, *rows = list(csv.reader(path.open()))
    assert header == list(CSV_HEADER)
    assert [row[1:] for row in rows] == [
        ["model-a", "5", "1", "6", "agent", "thread-1"],
        ["model-a", "7", "1", "8", "", ""],
    ]


def test_track_tokens_records_even_when_the_body_raises(tmp_path):
    path = tmp_path / "usage.csv"
    with (  # noqa: PT012  the raise must happen inside track_tokens
        pytest.raises(RuntimeError),
        track_tokens(str(path), "eval", "t-1") as handler,
    ):
        handler.usage_metadata.update(_usage(3))
        raise RuntimeError
    (_, row) = list(csv.reader(path.open()))
    assert row[1:] == ["model-a", "3", "1", "4", "eval", "t-1"]


# --- message building ----------------------------------------------------------------


def _image_urls(messages):
    return [
        c["image_url"]["url"] for c in messages[0].content if c["type"] == "image_url"
    ]


def test_message_images_get_mime_types(tmp_path):
    # regression for cleanup.md 3.1 (a follow-up with a new image raised NameError)
    png = tmp_path / "x.png"
    Image.new("RGB", (4, 4)).save(png)
    unknown = tmp_path / "x.unknown"
    unknown.write_bytes(png.read_bytes())
    urls = _image_urls(build_message([FIXTURE_IMAGE, png, unknown, "hi"]))
    assert urls[0].startswith("data:image/jpeg;base64,")
    assert urls[1].startswith("data:image/png;base64,")
    assert urls[2].startswith("data:image/png;base64,")  # from the content
    assert base64.b64decode(urls[0].split(",", 1)[1]) == FIXTURE_IMAGE.read_bytes()


@pytest.mark.parametrize(
    ("fmt", "mime"),
    [("WEBP", "image/webp"), ("PNG", "image/png"), ("GIF", "image/gif")],
)
def test_misnamed_images_get_the_type_of_their_content(tmp_path, fmt, mime):
    # a WebP saved as .jpg made Bedrock reject a follow-up (declared type != bytes)
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (4, 4)).save(path, format=fmt)
    assert _image_urls(build_message([path]))[0].startswith(f"data:{mime};base64,")


def test_unrecognised_content_falls_back_to_the_name_then_the_default(tmp_path):
    named = tmp_path / "x.jpg"
    named.write_bytes(b"not an image signature")
    nameless = tmp_path / "x"
    nameless.write_bytes(b"not an image signature")
    urls = _image_urls(build_message([named, nameless]))
    assert urls[0].startswith("data:image/jpeg;base64,")
    assert urls[1].startswith("data:image/png;base64,")


def test_message_parts_keep_their_order_and_strings_stay_text():
    message = build_message(["a", FIXTURE_IMAGE, str(FIXTURE_IMAGE)])
    assert [c["type"] for c in message[0].content] == ["text", "image_url", "text"]
    # a string is never read as a file, even if it is a path
    assert message[0].content[2]["text"] == str(FIXTURE_IMAGE)


# --- evaluation helpers --------------------------------------------------------------


def _evaluation(vis, suit, comp):
    return RecommendationEvaluation(
        visual_grounding_score=vis,
        visual_grounding_explanation="",
        item_suitability_score=suit,
        item_suitability_explanation="",
        completeness_coverage_score=comp,
        completeness_coverage_explanation="",
    )


def test_final_score_is_a_weighted_mean():
    # regression for cleanup.md 3.2 (operator precedence)
    weights = EvalWeights(
        visual_grounding=0.2, item_suitability=0.3, completeness_coverage=0.5
    )
    assert get_final_score(
        _evaluation(10, 10, 10), has_input_image=True, weights=weights
    ) == pytest.approx(10)
    assert get_final_score(
        _evaluation(0, 10, 10), has_input_image=False, weights=weights
    ) == pytest.approx(10)
    assert get_final_score(
        _evaluation(2, 4, 8), has_input_image=True, weights=weights
    ) == pytest.approx(5.6)
    assert get_final_score(
        _evaluation(0, 4, 8), has_input_image=False, weights=weights
    ) == pytest.approx(6.5)


def test_evaluation_inputs_attach_matching_images(tmp_path):
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps([["q1", "q2"], ["q3"]]))
    images = tmp_path / "images"
    images.mkdir()
    (images / "01_02.jpg").write_bytes(b"x")
    inputs = get_evaluation_inputs(str(queries), str(images))
    assert [(i["group_num"], i["query_num_in_group"]) for i in inputs] == [
        (1, 1),
        (1, 2),
        (2, 1),
    ]
    assert [len(i["input_images_path"]) for i in inputs] == [0, 1, 0]
    assert inputs[1]["input_images_path"] == [str(images / "01_02.jpg")]


@pytest.mark.parametrize("reasoning", [None, False, True])
def test_build_llm_passes_ollama_reasoning(reasoning):
    llm = build_llm(
        LLMSettings(provider="ollama", model_id="m", reasoning=reasoning), "x"
    )
    assert llm.reasoning is reasoning
