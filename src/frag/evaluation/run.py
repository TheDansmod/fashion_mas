"""LLM-as-a-judge evaluation of the agent over the curated query set (`frag eval`).

Two passes keep LLM use bounded: run the agent on every query and store the outputs,
then judge the stored outputs. Each evaluation is a self-contained directory
`<runs_dir>/<UTC time>/`:

- `config.json`: the agent and judge models, the embedding backend, the collection and
  its size, and the git commit; results from different setups aren't comparable.
- `results.jsonl`: one line per query, appended as it finishes, with a `status`
  (`ok`, `no_recommendations` or `error`) so failures are counted, not dropped.
- `judgement.jsonl`: one line per query, with a `status` (`ok`, `skipped` or `error`).
- `images/`: copies of the input and recommended images; all paths in the files are
  relative to the run directory.
- `summary.json`: completed/total, mean ± std of the final score over the judged
  queries, and a per-group breakdown.
"""

import asyncio
import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from loguru import logger as log
from qdrant_client import AsyncQdrantClient

from frag.agent.graph import FashionAgent
from frag.evaluation.prompts import EVALUATION_PROMPT
from frag.evaluation.schemas import RecommendationEvaluation
from frag.runtime import Resources
from frag.settings import EvalWeights, Settings
from frag.tokens import track_tokens
from frag.utils.common_utils import get_multi_image_multi_prompt_message

CONFIG = "config.json"
RESULTS = "results.jsonl"
JUDGEMENT = "judgement.jsonl"
SUMMARY = "summary.json"
IMAGES = "images"


# --- inputs ---------------------------------------------------------------------------


def get_evaluation_inputs(queries_path: str, images_dir: str) -> list[dict[str, Any]]:
    """One input per query; query `q` of group `g` has image `gg_qq.jpg` if it exists.

    The queries file is a list of groups, each a list of query strings.
    """
    groups = json.loads(Path(queries_path).read_text())
    images = Path(images_dir)
    inputs = []
    for group_num, queries in enumerate(groups, start=1):
        for query_num, text in enumerate(queries, start=1):
            image = images / f"{group_num:02}_{query_num:02}.jpg"
            inputs.append(
                {
                    "group_num": group_num,
                    "query_num_in_group": query_num,
                    "input_text": text,
                    "input_images_path": [str(image)] if image.is_file() else [],
                }
            )
    return inputs


def _key(item: dict[str, Any]) -> str:
    return f"{item['group_num']:02}_{item['query_num_in_group']:02}"


# --- run directory and configuration --------------------------------------------------


def _model_name(settings: Settings) -> str:
    return f"{settings.llm.provider}:{settings.llm.model_id}"


def _git_commit() -> str | None:
    """The checked-out commit, with "-dirty" if there are uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607  git from PATH, read-only
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607  git from PATH, read-only
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if dirty else commit


async def _collection_size(settings: Settings) -> int | None:
    """Points in the configured Qdrant collection (None if it can't be read)."""
    client = AsyncQdrantClient(
        url=settings.vector_db.url, prefer_grpc=settings.vector_db.prefer_grpc
    )
    try:
        return (await client.count(settings.collection, exact=True)).count
    except Exception:  # noqa: BLE001  the size is informational only
        log.opt(exception=True).warning("Could not count the Qdrant collection.")
        return None
    finally:
        await client.close()


async def build_config(settings: Settings, limit: int | None) -> dict[str, Any]:
    """What produced a run's results."""
    embedding = settings.embedding
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "agent_model": _model_name(settings),
        # the judge uses the same LLM settings; updated if a later pass judges
        "judge_model": _model_name(settings),
        "temperature": settings.llm.temperature,
        "reasoning": settings.llm.reasoning,
        "embedding_backend": embedding.backend,
        "embedding_model": (
            embedding.siglip_model
            if embedding.backend == "siglip"
            else embedding.nova_model_id
        ),
        "collection": settings.collection,
        "collection_points": await _collection_size(settings),
        "tools_mode": settings.tools.mode,
        "limit": limit,
        "weights": settings.evaluation.weights.model_dump(),
    }


def new_run_dir(runs_dir: str) -> Path:
    """Create `<runs_dir>/<UTC time>/` with its images folder."""
    path = Path(runs_dir) / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (path / IMAGES).mkdir(parents=True)
    return path


def latest_run_dir(runs_dir: str) -> Path:
    """The most recent run directory (for judging a stored run)."""
    runs = sorted(p for p in Path(runs_dir).glob("*") if (p / RESULTS).is_file())
    if not runs:
        msg = f"No evaluation runs with {RESULTS} under {runs_dir}."
        raise FileNotFoundError(msg)
    return runs[-1]


def _append(path: Path, record: dict[str, Any]) -> None:
    # one line per record, appended: no read-modify-write of the whole file
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """The records of a JSON-lines file."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- pass 1: run the agent ------------------------------------------------------------


def _copy_image(source: str, run_dir: Path, name: str) -> str:
    """Copy an image into the run's images folder; return its run-relative path."""
    relative = Path(IMAGES) / f"{name}{Path(source).suffix or '.png'}"
    shutil.copyfile(source, run_dir / relative)
    return str(relative)


async def run_query(
    resources: Resources, eval_input: dict[str, Any], run_dir: Path
) -> dict[str, Any]:
    """Run one query through a fresh thread; return its results record."""
    key = _key(eval_input)
    record: dict[str, Any] = {
        "key": key,
        "group_num": eval_input["group_num"],
        "query_num_in_group": eval_input["query_num_in_group"],
        "input_text": eval_input["input_text"],
        "input_images": [
            _copy_image(path, run_dir, f"{key}_input_{i + 1}")
            for i, path in enumerate(eval_input["input_images_path"])
        ],
        "agent_model": _model_name(resources.settings),
        "status": "ok",
        "error": None,
        "recommendations": [],
        "explanation": "",
    }
    agent = FashionAgent(
        llm=resources.llm,
        tools=resources.tools,
        checkpointer=resources.checkpointer,
        settings=resources.settings.agent,
    )
    thread_id = str(uuid4())
    start = time.monotonic()
    with track_tokens(resources.settings.token_usage_csv, "eval", thread_id) as usage:
        try:
            state = await agent.run_turn(
                eval_input["input_text"],
                eval_input["input_images_path"],
                thread_id,
                callbacks=[usage],
            )
        except Exception as exc:  # noqa: BLE001  recorded; the other queries go on
            log.opt(exception=True).warning("Query {} failed.", key)
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
        else:
            pairs = zip(
                state["recommended_clothes_image_paths"],
                state["recommended_clothes_descriptions"],
                strict=True,
            )
            record["recommendations"] = [
                {
                    "image": _copy_image(path, run_dir, f"{key}_rec_{i + 1}"),
                    "description": description,
                }
                for i, (path, description) in enumerate(pairs)
            ]
            record["explanation"] = state["recommended_clothes_explanation"]
            if not record["recommendations"]:
                record["status"] = "no_recommendations"
    record["seconds"] = round(time.monotonic() - start, 1)
    record["tokens"] = usage.usage_metadata
    return record


async def run_evaluation_set(
    resources: Resources,
    run_dir: Path,
    limit: int | None = None,
    sleep_s: float = 0.0,
) -> None:
    """Run the agent over the queries (the first `limit`); append to results.jsonl."""
    settings = resources.settings.evaluation
    inputs = get_evaluation_inputs(settings.queries_path, settings.images_dir)[:limit]
    for n, eval_input in enumerate(inputs, start=1):
        record = await run_query(resources, eval_input, run_dir)
        _append(run_dir / RESULTS, record)
        log.info(
            "Query {} ({}/{}): {}, {} items, {}s.",
            record["key"],
            n,
            len(inputs),
            record["status"],
            len(record["recommendations"]),
            record["seconds"],
        )
        # the LLM's rate limiter throttles requests; this is only for providers
        # whose limits it doesn't capture
        if sleep_s and n < len(inputs):
            await asyncio.sleep(sleep_s)


# --- pass 2: judge --------------------------------------------------------------------


def get_final_score(
    reco_eval: RecommendationEvaluation,
    has_input_image: bool,  # noqa: FBT001  positional by long-standing call sites
    weights: EvalWeights,
) -> float:
    """Weighted mean of the criteria; visual grounding only counts with an image."""
    vis_score = reco_eval.visual_grounding_score
    suit_score = reco_eval.item_suitability_score
    comp_score = reco_eval.completeness_coverage_score
    if has_input_image:
        wt_sum = (
            weights.visual_grounding
            + weights.item_suitability
            + weights.completeness_coverage
        )
        weighted = (
            weights.visual_grounding * vis_score
            + weights.item_suitability * suit_score
            + weights.completeness_coverage * comp_score
        )
        return weighted / wt_sum
    wt_sum = weights.item_suitability + weights.completeness_coverage
    weighted = (
        weights.item_suitability * suit_score
        + weights.completeness_coverage * comp_score
    )
    return weighted / wt_sum


def judge_message(record: dict[str, Any], run_dir: Path) -> list:
    """The judge's input: the user's images and request, then each recommendation."""
    parts = [("image", str(run_dir / path)) for path in record["input_images"]]
    if parts:
        parts.append(("text", "Above are the images the user uploaded.\n\n"))
    else:
        parts.append(("text", "The user did not upload any images.\n\n"))
    parts.append(("text", f"User request: {record['input_text']}\n\n"))
    for i, rec in enumerate(record["recommendations"], start=1):
        parts.append(("image", str(run_dir / rec["image"])))
        parts.append(
            ("text", f"Above is Recommended Item {i}: {rec['description']}\n\n")
        )
    parts.append(("text", EVALUATION_PROMPT))
    return get_multi_image_multi_prompt_message(parts)


async def judge_record(
    resources: Resources,
    record: dict[str, Any],
    run_dir: Path,
    usage: Any,  # noqa: ANN401  a LangChain callback handler
) -> dict[str, Any]:
    """Judge one results record; failures are recorded, not raised."""
    judgement: dict[str, Any] = {
        "key": record["key"],
        "group_num": record["group_num"],
        "query_num_in_group": record["query_num_in_group"],
        "agent_model": record["agent_model"],
        "judge_model": _model_name(resources.settings),
        "status": "ok",
        "error": None,
        "final_score": None,
    }
    if record["status"] != "ok":
        judgement.update(status="skipped", error=f"run status: {record['status']}")
        return judgement
    model = resources.llm.with_structured_output(RecommendationEvaluation)
    try:
        response: RecommendationEvaluation = await model.ainvoke(
            judge_message(record, run_dir), config={"callbacks": [usage]}
        )
    except Exception as exc:  # noqa: BLE001  recorded; the other judgements go on
        log.opt(exception=True).warning("Judging {} failed.", record["key"])
        judgement.update(status="error", error=f"{type(exc).__name__}: {exc}")
        return judgement
    score = get_final_score(
        response, bool(record["input_images"]), resources.settings.evaluation.weights
    )
    judgement.update(**response.model_dump(), final_score=round(score, 2))
    return judgement


async def judge_evaluation_outputs(resources: Resources, run_dir: Path) -> None:
    """Judge every record in results.jsonl, writing judgement.jsonl afresh."""
    results = read_jsonl(run_dir / RESULTS)
    # a stored run may be judged later by another model: record which one did
    config = json.loads((run_dir / CONFIG).read_text())
    config["judge_model"] = _model_name(resources.settings)
    (run_dir / CONFIG).write_text(json.dumps(config, indent=2))
    out = run_dir / JUDGEMENT
    out.unlink(missing_ok=True)
    with track_tokens(resources.settings.token_usage_csv, "eval-judge") as usage:
        for record in results:
            judgement = await judge_record(resources, record, run_dir, usage)
            _append(out, judgement)
            log.info(
                "Judged {}: {} {}",
                judgement["key"],
                judgement["status"],
                judgement["final_score"],
            )


# --- summary --------------------------------------------------------------------------


def _stats(scores: list[float]) -> dict[str, Any]:
    if not scores:
        return {"n": 0, "mean": None, "std": None}
    return {
        "n": len(scores),
        "mean": round(float(np.mean(scores)), 2),
        "std": round(float(np.std(scores)), 2),
    }


def summarise(run_dir: Path) -> dict[str, Any]:
    """Counts and scores of a run; failed and skipped queries are counted too.

    Also written to summary.json.
    """
    results = read_jsonl(run_dir / RESULTS)
    judgements = (
        read_jsonl(run_dir / JUDGEMENT) if (run_dir / JUDGEMENT).is_file() else []
    )
    config = json.loads((run_dir / CONFIG).read_text())
    run_status: dict[str, int] = {}
    for r in results:
        run_status[r["status"]] = run_status.get(r["status"], 0) + 1
    scored = [j for j in judgements if j["status"] == "ok"]
    groups = sorted({r["group_num"] for r in results})
    summary = {
        "agent_model": config["agent_model"],
        "judge_model": config["judge_model"],
        "total": len(results),
        "completed": run_status.get("ok", 0),
        "run_status": run_status,
        "judged": len(scored),
        "judge_errors": sum(j["status"] == "error" for j in judgements),
        "score": _stats([j["final_score"] for j in scored]),
        "groups": {
            str(g): {
                "total": sum(r["group_num"] == g for r in results),
                "completed": sum(
                    r["group_num"] == g and r["status"] == "ok" for r in results
                ),
                "score": _stats(
                    [j["final_score"] for j in scored if j["group_num"] == g]
                ),
            }
            for g in groups
        },
    }
    (run_dir / SUMMARY).write_text(json.dumps(summary, indent=2))
    return summary


def log_summary(summary: dict[str, Any], run_dir: Path) -> None:
    """Log the summary in a few readable lines."""
    score = summary["score"]
    log.info(
        "Evaluation {}: agent {}, judge {}.",
        run_dir,
        summary["agent_model"],
        summary["judge_model"],
    )
    log.info(
        "Completed {}/{} queries {}; judged {} ({} judge errors); "
        "score {} ± {} over the judged queries.",
        summary["completed"],
        summary["total"],
        summary["run_status"],
        summary["judged"],
        summary["judge_errors"],
        score["mean"],
        score["std"],
    )
    for group, g in summary["groups"].items():
        log.info(
            "  group {}: completed {}/{}, score {} ± {} (n={})",
            group,
            g["completed"],
            g["total"],
            g["score"]["mean"],
            g["score"]["std"],
            g["score"]["n"],
        )


# --- the whole pipeline ---------------------------------------------------------------


async def evaluate(  # noqa: PLR0913  keyword-only options, one per CLI flag
    resources: Resources,
    *,
    run: bool = True,
    judge: bool = True,
    limit: int | None = None,
    sleep_s: float = 0.0,
    run_dir: str | None = None,
) -> Path:
    """Run and/or judge an evaluation; return its directory.

    With `run`, a new run directory is created (unless `run_dir` is given, which must
    not hold results yet). Without it, `run_dir` (default: the latest run) is judged.
    """
    runs_dir = resources.settings.evaluation.runs_dir
    if run:
        path = Path(run_dir) if run_dir else new_run_dir(runs_dir)
        (path / IMAGES).mkdir(parents=True, exist_ok=True)
        if (path / RESULTS).exists():
            msg = f"{path / RESULTS} already exists; use a new run directory."
            raise FileExistsError(msg)
        config = await build_config(resources.settings, limit)
        (path / CONFIG).write_text(json.dumps(config, indent=2))
        log.info("Evaluation run directory: {}", path)
        await run_evaluation_set(resources, path, limit, sleep_s)
    else:
        path = Path(run_dir) if run_dir else latest_run_dir(runs_dir)
    if judge:
        await judge_evaluation_outputs(resources, path)
    log_summary(summarise(path), path)
    return path
