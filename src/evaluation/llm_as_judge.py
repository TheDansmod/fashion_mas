import os
import json
from pathlib import Path
import logging
from uuid import uuid4

import asyncio
import hydra

from src.rag_pipeline.rag_agent import FashionAgent
from src.utils.common_utils import track_token_use
from langgraph.types import interrupt, Command

log = logging.getLogger(__name__)

def get_evaluation_inputs(cfg):
    eval_inputs = []
    with open(cfg.eval.json_queries_path, 'r') as f:
        queries = json.load(f)
    input_images = os.listdir(cfg.eval.query_images_folder_path)
    for group_idx, group_queries in enumerate(queries):
        for query_idx, input_text in enumerate(group_queries):
            image_name = f'{group_idx + 1:0>2}_{query_idx + 1:0>2}.jpg'
            if image_name in input_images:
                input_image_path = cfg.eval.query_images_path.format(image_filename=image_name)
                query_input = {'input_text': input_text, 'input_images_path': [input_image_path], 'group_num': group_idx + 1, 'query_num_in_group': query_idx + 1}
            else:
                query_input = {'group_num': group_idx + 1, 'query_num_in_group': query_idx + 1, 'input_text': input_text, 'input_images_path': []}
            eval_inputs.append(query_input)
    return eval_inputs

@track_token_use
async def run_fashion_agent_single_input(cfg, eval_input, callback_config):
    log.debug("Running single evaluation")
    agent = FashionAgent(cfg, callback_config)
    recommended_clothes_image_paths = []
    recommended_clothes_descriptions = []
    try:
        await agent.compile_graph(cfg.rag_pipeline.persistence.db_path)
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
        await agent.close_connection()
        return recommended_clothes_image_paths, recommended_clothes_descriptions

def write_result(cfg, result):
    # we want to write the results after each run so as not to lose any information
    # json file can't simply be appended to, so we first load the json, modify it and write it back
    # the paths in the above files are relative, need to make the full
    absolute_image_paths = []
    for image_path in result['recommended_clothes_image_paths']:
        absolute_path = hydra.utils.to_absolute_path(image_path)
        absolute_image_paths.append(absolute_path)
    result['recommended_clothes_image_paths'] = absolute_image_paths
    path = Path(cfg.eval.eval_response_file_path)
    if not (path.exists() and path.is_file()):
        path.write_text('[]')
    with open(cfg.eval.eval_response_file_path, 'r') as f:
        data = json.load(f)
    data.append(result)
    with open(cfg.eval.eval_response_file_path, 'w') as f:
        json.dump(data, f, indent=2)

async def run_evaluation_set(cfg):
    log.debug("Running evaluation set")
    for eval_input in get_evaluation_inputs(cfg):
        log.debug(f"eval input: {eval_input}")
        paths, descr = await run_fashion_agent_single_input(cfg, eval_input)
        result = {'recommended_clothes_image_paths': paths, 'recommended_clothes_descriptions': descr, **eval_input}
        write_result(cfg, result)
        await asyncio.sleep(5)

# TODO: def can remove this function later
def check_evaluation_output(cfg):
    # the goal of this function is to check how many evaluations actually produced valid outputs
    no_paths, no_descr, no_both = 0, 0, 0
    with open(cfg.eval.eval_response_file_path, 'r') as f:
        data = json.load(f)
    total = len(data)
    for eval_run in data:
        if len(eval_run['recommended_clothes_image_paths']) == 0:
            no_paths += 1
        if len(eval_run['recommended_clothes_descriptions']) == 0:
            no_descr += 1
        if len(eval_run['recommended_clothes_image_paths']) == 0 and len(eval_run['recommended_clothes_descriptions']) == 0:
            no_both += 1
    log.info(f"Out of {total} runs, {no_paths} produced no paths.")
    log.info(f"Out of {total} runs, {no_descr} produced no descriptions.")
    log.info(f"Out of {total} runs, {no_both} produced neither descriptions nor paths.")

