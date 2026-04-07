import os
import json
from pathlib import Path
from uuid import uuid4

import asyncio
import numpy as np
from langgraph.types import interrupt, Command
from loguru import logger as log
from dependency_injector.wiring import inject, Provide as PV

from frag.rag_pipeline.rag_agent import FashionAgent
from frag.utils.common_utils import track_token_use, get_multi_image_multi_prompt_message
from frag.rag_pipeline.llm_schemas import RecommendationEvaluation
from frag.config.container import Container

cfg = Container.config.provided


@inject
def get_evaluation_inputs(
    eval_queries_path: str = PV[cfg.evaluation.json_queries_path],
    query_images_folder: str = PV[cfg.evaluation.query_images_folder_path],
    query_images_path: str = PV[cfg.evaluation.query_images_path],
):
    eval_inputs = []
    with open(eval_queries_path, 'r') as f:
        queries = json.load(f)
    input_images = os.listdir(query_images_folder)
    for group_idx, group_queries in enumerate(queries):
        for query_idx, input_text in enumerate(group_queries):
            image_name = f'{group_idx + 1:0>2}_{query_idx + 1:0>2}.jpg'
            if image_name in input_images:
                input_image_path = query_images_path.format(image_filename=image_name)
                query_input = {'input_text': input_text, 'input_images_path': [input_image_path], 'group_num': group_idx + 1, 'query_num_in_group': query_idx + 1}
            else:
                query_input = {'group_num': group_idx + 1, 'query_num_in_group': query_idx + 1, 'input_text': input_text, 'input_images_path': []}
            eval_inputs.append(query_input)
    return eval_inputs

@track_token_use
async def run_fashion_agent_single_input(eval_input, callback_config):
    log.debug("Running single evaluation")
    agent = FashionAgent(callback_config)
    recommended_clothes_image_paths = []
    recommended_clothes_descriptions = []
    try:
        await agent.compile_graph()
        config = {"configurable": {"thread_id": uuid4()}}
        result = await agent.ainvoke({"is_chat_start": True}, config)
        # single run
        resume_payload = {"input_text": eval_input['input_text'], "input_images_path": eval_input['input_images_path']}
        result = await agent.ainvoke(Command(resume=resume_payload), config=config)
        log.debug(f"got result {result}")
        if 'recommended_clothes_image_paths' in result and 'recommended_clothes_descriptions' in result:
            recommended_clothes_image_paths = result['recommended_clothes_image_paths']
            recommended_clothes_descriptions = result['recommended_clothes_descriptions']
        else:
            log.error("ERROR: recommended_clothes_image_paths not in result, but expected")
    except Exception as ex:
        log.exception("Some exception while running the fashion agent.")
    finally:
        return recommended_clothes_image_paths, recommended_clothes_descriptions

@inject
def write_result(
    result,
    solution=True,
    judgement=False,
    judgement_file_path = PV[cfg.evaluation.judgement_file_path],
    eval_response_file_path = PV[cfg.evaluation.eval_response_file_path],
):
    # if solution = True (default), we assume that we are just doing evaluation
    # - not judgement writes - so we write to the solution file
    # if judgement is true - we write to the judegement file
    if judgement and not solution:
        file_path = judgement_file_path
    else:
        file_path = eval_response_file_path
    # we want to write the results after each run so as not to lose any information
    # json file can't simply be appended to, so we first load the json, modify it and write it back
    # the paths in the above files are relative, need to make the full
    absolute_image_paths = []
    for image_path in result['recommended_clothes_image_paths']:
        absolute_image_paths.append(image_path)
    result['recommended_clothes_image_paths'] = absolute_image_paths
    path = Path(file_path)
    if not (path.exists() and path.is_file()):
        path.write_text('[]')
    with open(file_path, 'r') as f:
        data = json.load(f)
    data.append(result)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

async def run_evaluation_set():
    log.debug("Running evaluation set")
    for eval_input in get_evaluation_inputs():
        log.debug(f"eval input: {eval_input}")
        paths, descr = await run_fashion_agent_single_input(eval_input)
        result = {'recommended_clothes_image_paths': paths, 'recommended_clothes_descriptions': descr, **eval_input}
        write_result(result)
        await asyncio.sleep(5)

@inject
def get_final_score(
    reco_eval: RecommendationEvaluation,
    has_input_image: bool,
    vis_wt = PV[cfg.evaluation.weights.visual_grounding],
    suit_wt = PV[cfg.evaluation.weights.item_suitability],
    comp_wt = PV[cfg.evaluation.weights.completeness_coverage],
):
    vis_score = reco_eval.visual_grounding_score
    suit_score = reco_eval.item_suitability_score
    comp_score = reco_eval.completeness_coverage_score
    if has_input_image:
        wt_sum = vis_wt + suit_wt + comp_wt
        return vis_wt * vis_score + suit_wt * suit_score + comp_wt * comp_score / wt_sum
    else:
        wt_sum = suit_wt + comp_wt
        return suit_wt * suit_score + comp_wt * comp_score / wt_sum

# this judges the outputs that are present - ignores those that have no outputs
@track_token_use
def judge_evaluation_outputs(
    model = PV[Container.llm_model.provided],
    eval_response_file_path = PV[cfg.evaluation.eval_response_file_path],
    eval_prompt = PV[cfg.prompts.evaluation_prompt],
    callback_config = None,
):
    # setup llm
    structured_model = model.with_structured_output(RecommendationEvaluation)
    # get the evaluation outputs
    with open(eval_response_file_path, 'r') as f:
        data = json.load(f)
    # iterate through the evaluation outputs
    for eval_run in data:
        # ignore runs with no outputs
        if len(eval_run['recommended_clothes_image_paths']) == 0 or len(eval_run['recommended_clothes_descriptions']) == 0:
            continue
        # generate the prompt
        has_input_image = len(eval_run['input_images_path']) > 0
        prompts = []
        for img_path in eval_run['input_images_path']:  # we want error if the key is not in the dict
            prompts.append(("image", img_path))
        prompt_prefix = "All the above are user input images.\n\n" if has_input_image else ""
        prompts.append(("text", f"{prompt_prefix}The user request is: {eval_run['input_text']}\n\n"))
        for idx, img_descr in enumerate(eval_run['recommended_clothes_descriptions']):
            prompts.append(("image", eval_run['recommended_clothes_image_paths'][idx]))
            prompts.append(("text", f"Above is Recommended Image {idx + 1}.\nDescription of Item {idx + 1}: {img_descr}\n\n"))
        prompts.append(("text", eval_prompt))
        msg = get_multi_image_multi_prompt_message(prompts)
        # invoke model, get response
        try:
            log.debug("Invoking model")
            response = structured_model.invoke(msg, config=callback_config)
        except Exception as e:
            log.exception("For group: {eval_run['group_num']}, query: {eval_run['query_num_in_group']} There was an exception while invoking the structured model for judgement.")
            continue
        log.debug(f"For group: {eval_run['group_num']}, query: {eval_run['query_num_in_group']} Got response from model:\n{response}")
        # compute final score
        final_score = get_final_score(response, has_input_image)
        # write final score
        result = {**eval_run, **response.model_dump(), 'final_score': f'{final_score:.2f}'}
        write_result(result, solution=False, judgement=True)

@inject
def print_average_score(judgement_file_path = PV[cfg.evaluation.judgement_file_path]):
    scores = []
    with open(judgement_file_path, 'r') as f:
        data = json.load(f)
    for judgement in data:
        scores.append(float(judgement['final_score']))
    mean, std = np.mean(scores), np.std(scores)
    log.info(f"Mean = {mean}; Std = {std}")

async def run_full_evaluation_pipeline():
    await run_evaluation_set()
    judge_evaluation_outputs()
    print_average_score()
