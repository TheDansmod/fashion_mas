import os
import json
import logging
from uuid import uuid4

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
                query_input = {'input_text': input_text, 'input_images_path': [input_image_path]}
            else:
                query_input = {'input_text': input_text, 'input_images_path': []}
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


async def run_evaluation_set(cfg):
    log.debug("Running evaluation set")
    results = []
    for eval_input in get_evaluation_inputs(cfg)[:2]:
        log.debug(f"eval input: {eval_input}")
        paths, descr = await run_fashion_agent_single_input(cfg, eval_input)
        results.append({'recommended_clothes_image_paths': paths, 'recommended_clothes_descriptions': descr, **eval_input})
    with open(cfg.eval.eval_response_file_path, 'w') as f:
        json.dump(results, f, indent=2)
