import logging
import sqlite3
from typing import Optional

import hydra
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.utils.common_utils import get_image_prompt_message, draw_langraph_topology
from src.data_manager.vector_db_writer import FashionSigLIPEmbedding, get_fashion_gen_data
from src.data_manager.vector_db_reader import VectorDbReader

log = logging.getLogger(__name__)

def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}

def langgraph_hello_world(cfg):
    graph = StateGraph(MessagesState)
    graph.add_node(mock_llm)
    graph.add_edge(START, "mock_llm")
    graph.add_edge("mock_llm", END)
    graph = graph.compile()
    
    response = graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
    log.info(response)

class AgentState(BaseModel):
    input_images_path: list[str]
    input_text: str
    input_images_descriptions: Optional[list[str]] = None
    required_clothes_descriptions: Optional[list[str]] = None
    recommended_clothes_images: Optional[list[int]] = None
    recommended_clothes_explanation: Optional[list[str]] = None

class RequiredClothes(BaseModel):
    required_clothes_descriptions: list[str] = Field(min_length=1, description="List of descriptions of clothing items that satisfy the user's requests. One description per index.")

class FashionAgent:
    def __init__(self, cfg):
        provider = hydra.utils.instantiate(cfg.models.vlm_agent)
        self._model = provider(model=cfg.models.vlm_agent.name, temperature=cfg.models.vlm_agent.temp)
        self._embedder = FashionSigLIPEmbedding(cfg)
        self._reader = VectorDbReader(cfg)
        self._cfg = cfg

    def vision_node(self, state: AgentState) -> AgentState:
        log.debug("Entered vision node.")
        descr = []
        for image_path in state.input_images_path:
            msg = get_image_prompt_message(image_path=image_path, text_prompt=self._cfg.prompts.vision_node.user_prompt)
            log.debug("Invoking model.")
            response = self._model.invoke(msg)
            log.debug("Received response from model.")
            descr.append(response.content)
        log.debug(f"Descriptions obtained from vision node:\n{descr}")
        return {"input_images_descriptions": descr}

    def modifier_node(self, state: AgentState) -> AgentState:
        log.debug("Entered modifier node.")
        descr = []
        ref_descr = "\n".join([f"{idx+1}. {img_descr}" for idx, img_descr in enumerate(state.input_images_descriptions)])
        prompt = self._cfg.prompts.modifier_node.user_prompt.format(reference_descriptions=ref_descr, user_request=state.input_text)
        structured_model = self._model.with_structured_output(RequiredClothes)
        log.debug("Invoking model.")
        response = structured_model.invoke(prompt)
        log.debug("Received response from model.")
        log.debug(f"Required clothes descriptions obtained from modifier node:\n{response.required_clothes_descriptions}")
        return {"required_clothes_descriptions": response.required_clothes_descriptions}

    def recommender_node(self, state: AgentState) -> AgentState:
        log.debug("Entered recommender node.")
        # get embeddings for each of the required clothes descriptions
        # get the top image match for each of the embeddings - need client for this
        # populate the recommended_clothes_images with the indexes of the returned matches
        match_ids = []
        embeddings = self._embedder.get_text_embedding_batch(state.required_clothes_descriptions)
        for embedding in embeddings:
            match_ids.extend(self._reader.get_image_matches(embedding, num_matches=1))
        log.debug(f"Image IDs obtained from recommender_node:\n{match_ids}")
        return {"recommended_clothes_images": set(list(match_ids))}

    def explanation_node(self, state: AgentState) -> AgentState:
        log.debug("Entered explanation node.")
        expl = []
        ref_descr = "\n".join([f"{idx+1}. {img_descr}" for idx, img_descr in enumerate(state.input_images_descriptions)])
        descr_key = self._cfg.data.fashion_gen.descriptions_key
        img_key = self._cfg.data.fashion_gen.images_key
        for img_id in state.recommended_clothes_images:
            data, done = get_fashion_gen_data(self._cfg, from_idx=img_id, to_idx=img_id+1)
            img_descr = data[descr_key][0]
            text_prompt = self._cfg.prompts.explanation_node.user_prompt.format(reference_descriptions=ref_descr, recommended_image_description=img_descr, user_request=state.input_text)
            msg = get_image_prompt_message(text_prompt=text_prompt, numpy_image=data[img_key][0])
            log.debug("Invoking model.")
            response = self._model.invoke(msg)
            log.debug("Received response from model.")
            expl.append(response.content)
        log.debug(f"Explanations from explanation_node:\n{expl}")
        return {"recommended_clothes_explanation": expl}

    def invoke(self, initial_state, config, conn_string):
        with SqliteSaver.from_conn_string(conn_string) as checkpointer:
            builder = StateGraph(AgentState)
            # nodes
            builder.add_node("vision_node", self.vision_node)
            builder.add_node("modifier_node", self.modifier_node)
            builder.add_node("recommender_node", self.recommender_node)
            builder.add_node("explanation_node", self.explanation_node)
            # edges
            builder.add_edge(START, "vision_node")
            builder.add_edge("vision_node", "modifier_node")
            builder.add_edge("modifier_node", "recommender_node")
            builder.add_edge("recommender_node", "explanation_node")
            builder.add_edge("explanation_node", END)
            # compile and run
            app = builder.compile(checkpointer=checkpointer)
            draw_langraph_topology(app, self._cfg.misc.node_diagram_path)
            invocation_state = None if self._cfg.rag_pipeline.persistence.resume_from_checkpoint else initial_state
            result = app.invoke(invocation_state, config)
            return result

def run_fashion_agent(cfg):
    # check_same_thread=False since SqliteSaver manages its own locking
    # conn = sqlite3.connect(cfg.rag_pipeline.persistence.db_path, check_same_thread=False)
    agent = FashionAgent(cfg)
    initial_state = {"input_images_path": [cfg.misc.input_image_path_01], "input_text": "Please provide jeans pants that will go will with the uploaded shirt."}
    config = {"configurable": {"thread_id": cfg.rag_pipeline.persistence.thread_id}}
    result = agent.invoke(initial_state, config, cfg.rag_pipeline.persistence.db_path)
    log.debug(f"Result: {result}")
