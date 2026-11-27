"""LLM-as-a-judge evaluation of the agent over the curated query set.

Two passes keep LLM use bounded: run the agent on every query and store the outputs,
then judge the stored outputs. Phase 9 of cleanup.md restructures the output files and
records failures; this module currently mirrors the original behaviour.
"""

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from loguru import logger as log

from frag.agent.graph import FashionAgent
from frag.evaluation.prompts import EVALUATION_PROMPT
from frag.evaluation.schemas import RecommendationEvaluation
from frag.runtime import Resources
from frag.settings import EvalSettings, EvalWeights
from frag.tokens import track_tokens
from frag.utils.common_utils import get_multi_image_multi_prompt_message


def get_evaluation_inputs(settings: EvalSettings) -> list[dict]:
    """Build one input per query; query `i` of group `g` may have image `gg_ii.jpg`."""
    queries = json.loads(Path(settings.queries_path).read_text())
    images_dir = Path(settings.images_dir)
    available_images = {p.name for p in images_dir.iterdir()}
    eval_inputs = []
    for group_idx, group_queries in enumerate(queries):
        for query_idx, input_text in enumerate(group_queries):
            image_name = f"{group_idx + 1:0>2}_{query_idx + 1:0>2}.jpg"
            has_image = image_name in available_images
            images = [str(images_dir / image_name)] if has_image else []
            eval_inputs.append(
                {
                    "group_num": group_idx + 1,
                    "query_num_in_group": query_idx + 1,
                    "input_text": input_text,
                    "input_images_path": images,
                }
            )
    return eval_inputs


async def run_fashion_agent_single_input(
    resources: Resources, eval_input: dict
) -> tuple[list[str], list[str]]:
    """Run one query through a fresh agent; return (image paths, descriptions)."""
    log.debug("Running single evaluation")
    recommended_clothes_image_paths = []
    recommended_clothes_descriptions = []
    thread_id = str(uuid4())
    with track_tokens(resources.settings.token_usage_csv, "eval", thread_id) as usage:
        try:
            agent = FashionAgent(
                llm=resources.llm,
                tools=resources.tools,
                checkpointer=resources.checkpointer,
                settings=resources.settings.agent,
            )
            result = await agent.run_turn(
                eval_input["input_text"],
                eval_input["input_images_path"],
                thread_id,
                callbacks=[usage],
            )
            log.debug("got result {}", result)
            if (
                "recommended_clothes_image_paths" in result
                and "recommended_clothes_descriptions" in result
            ):
                recommended_clothes_image_paths = result[
                    "recommended_clothes_image_paths"
                ]
                recommended_clothes_descriptions = result[
                    "recommended_clothes_descriptions"
                ]
            else:
                log.error("recommended_clothes_image_paths not in result, but expected")
        except Exception:  # noqa: BLE001  one failed query must not abort the whole evaluation run
            # Phase 9 records these failures in the results instead of only logging
            log.exception("Some exception while running the fashion agent.")
    return recommended_clothes_image_paths, recommended_clothes_descriptions


def write_result(result: dict, file_path: str) -> None:
    """Append `result` to the JSON list in `file_path` (created if missing)."""
    path = Path(file_path)
    if not path.is_file():
        path.write_text("[]")
    data = json.loads(path.read_text())
    data.append(result)
    path.write_text(json.dumps(data, indent=2))


async def run_evaluation_set(resources: Resources, limit: int | None = None) -> None:
    """Run the agent over the evaluation queries (the first `limit` if given)."""
    log.debug("Running evaluation set")
    settings = resources.settings.evaluation
    for eval_input in get_evaluation_inputs(settings)[:limit]:
        log.debug("eval input: {}", eval_input)
        paths, descr = await run_fashion_agent_single_input(resources, eval_input)
        result = {
            "recommended_clothes_image_paths": paths,
            "recommended_clothes_descriptions": descr,
            **eval_input,
        }
        write_result(result, settings.eval_response_file_path)
        await asyncio.sleep(5)


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


def judge_evaluation_outputs(resources: Resources) -> None:
    """Judge every stored run that has outputs; runs without outputs are skipped."""
    settings = resources.settings.evaluation
    structured_model = resources.llm.with_structured_output(RecommendationEvaluation)
    data = json.loads(Path(settings.eval_response_file_path).read_text())
    with track_tokens(resources.settings.token_usage_csv, "eval-judge") as usage:
        callback_config = {"callbacks": [usage]}
        for eval_run in data:
            if (
                len(eval_run["recommended_clothes_image_paths"]) == 0
                or len(eval_run["recommended_clothes_descriptions"]) == 0
            ):
                continue
            has_input_image = len(eval_run["input_images_path"]) > 0
            prompts = [("image", path) for path in eval_run["input_images_path"]]
            prompt_prefix = (
                "All the above are user input images.\n\n" if has_input_image else ""
            )
            prompts.append(
                (
                    "text",
                    f"{prompt_prefix}The user request is: {eval_run['input_text']}\n\n",
                )
            )
            reco_paths = eval_run["recommended_clothes_image_paths"]
            reco_descrs = eval_run["recommended_clothes_descriptions"]
            pairs = zip(reco_paths, reco_descrs, strict=False)
            for idx, (img_path, img_descr) in enumerate(pairs):
                prompts.append(("image", img_path))
                prompts.append(
                    (
                        "text",
                        f"Above is Recommended Image {idx + 1}.\n"
                        f"Description of Item {idx + 1}: {img_descr}\n\n",
                    )
                )
            prompts.append(("text", EVALUATION_PROMPT))
            msg = get_multi_image_multi_prompt_message(prompts)
            try:
                log.debug("Invoking model")
                response = structured_model.invoke(msg, config=callback_config)
            except Exception:  # noqa: BLE001  one failed judgement must not abort the rest
                log.exception(
                    "Judge model failed for group {}, query {}.",
                    eval_run["group_num"],
                    eval_run["query_num_in_group"],
                )
                continue
            log.debug(
                "For group {}, query {} got judgement:\n{}",
                eval_run["group_num"],
                eval_run["query_num_in_group"],
                response,
            )
            final_score = get_final_score(response, has_input_image, settings.weights)
            result = {
                **eval_run,
                **response.model_dump(),
                "final_score": f"{final_score:.2f}",
            }
            write_result(result, settings.judgement_file_path)


def print_average_score(judgement_file_path: str) -> None:
    """Log the mean and standard deviation of all final scores."""
    data = json.loads(Path(judgement_file_path).read_text())
    scores = [float(judgement["final_score"]) for judgement in data]
    log.info("Mean = {}; Std = {}", np.mean(scores), np.std(scores))


async def run_full_evaluation_pipeline(
    resources: Resources,
    *,
    run: bool = True,
    judge: bool = True,
    limit: int | None = None,
) -> None:
    """Run the agent over the query set, judge the outputs and print the score.

    `run=False` judges the outputs already stored; `judge=False` only stores them.
    `limit` runs only the first N queries.
    """
    if run:
        await run_evaluation_set(resources, limit)
    if judge:
        judge_evaluation_outputs(resources)
        print_average_score(resources.settings.evaluation.judgement_file_path)
