"""The Agentic RAG Pipeline / Graph setup."""

import json
import logging
from typing import Literal
from uuid import uuid4

import aiosqlite
from langchain.agents import create_agent
# from src.utils.mock_llm_agent import mock_create_agent as create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from langgraph.types import interrupt, Command

from src.rag_pipeline.llm_schemas import (
    MatchedImageId,
    NumRecommendations,
    RequiredClothes,
    SingleImageDescription,
    UpdatedUserRequest,
    CriticalEvaluation,
    AgentState,
)
from src.utils.common_utils import (
    draw_langraph_topology,
    get_image_prompt_message,
    get_llm_model,
    get_tool_with_name,
    get_multi_image_prompt_message,
    save_image_url_to_folder,
    make_mistral_compatible,
    track_token_use,
    get_multi_image_multi_prompt_message,
)

log = logging.getLogger(__name__)


class FashionAgent:
    """Constructs and orchestrates the stateful DAG for multimodal retrieval.

    This primary interface initializes the vector database connection, establishes the
    underlying embedding functions, and connects the disparate computational nodes
    (quantification, intent synthesis, visual extraction, modification, and retrieval)
    via LangGraph topologies.

    Args:
        cfg (DictConfig): The hierarchical Hydra configuration object containing
            deterministic parameters for model instantiation, vector database routing,
            and prompt templates.
    """

    def __init__(self, cfg, callback_config):
        """Inits the agentic state representations and external client integrations.

        Instantiates the underlying generative models, the specialized SigLIP embedding
        architecture for multimodal vectorization, and the SQLite-backed vector database
        reader required for semantic similarity operations.

        Args:
            cfg (DictConfig): The Hydra configuration mapping containing systemic
                hyperparameters.
            callback_config (dict): A config object to be used with each llm call in
                order to track the number of input, output, and total tokens used. It
                is useful for Mistral API calls since they are rate limited to 500k
                tokens per minute.
        """
        self._model = get_llm_model(cfg)
        # setup embeddings
        self._cfg = cfg
        self._callback_config = callback_config
        self._connection = None
        self._graph = None
        self._client = MultiServerMCPClient(
            {
                "product_catalogue_server": {
                    "transport": "streamable_http",
                    "url": "http://localhost:9000/mcp",
                },
            }
        )

    async def human_node(self, state: AgentState) -> AgentState:
        """Node for human interaction.

        This is where we will be taking input from the user including the images and
        the input text.

        If this is not the start of the chat, then we need to send the original input
        images and the input text and the recommended images and the new images and the
        new input text to the LLM. Then, we need to ask the LLM to tell us which input
        images are still relevant, and to combine the old input and new input to create
        a input prompt that presents the user request as if this was the first input.
        For now, instead of providing the images themselves, which might be expensive,
        I am going to switch to providing just the descriptions of the images.
        """
        user_input = interrupt("Waiting for user input...")
        log.debug("Entered human node")
        if not state.is_chat_start:
            # extract all info needed for building new user request
            log.debug("Not chat start")
            org_img_descr = []
            for idx, descr in enumerate(state.input_images_descriptions):
                org_img_descr.append(f"Index {idx}: {descr}")
            org_img_descr = "\n".join(org_img_descr)
            resp_img_descr = []
            base_len = len(state.input_images_descriptions)
            for idx, descr in enumerate(state.recommended_clothes_descriptions):
                resp_img_descr.append(f"Index {base_len + idx}: {descr}")
            resp_img_descr = "\n".join(resp_img_descr)
            # construct the input message for LLM agent
            template = self._cfg.prompts.human_node.input_update_prompt
            prompt = template.format(original_image_descriptions=org_img_descr, original_text_input=state.input_text, response_image_descriptions=resp_img_descr, new_text_input=user_input.get("input_text", ""))
            msg = get_multi_image_prompt_message(user_input.get("input_images_path", []), text_prompt=prompt)
            # create and invoke agent
            log.debug("Invoking agent")
            agent = create_agent(model=self._model, response_format=UpdatedUserRequest, debug=False)
            response = await agent.ainvoke({"messages": msg}, config=self._callback_config)
            log.debug("Got agent response")
            # update the user input so that the normal state update (which assumes first chat, can also serve for non-first chat situations
            updated_user_request = response["structured_response"]
            user_input["input_text"] = updated_user_request.updated_user_query
            num_new_input_images = len(user_input.get("input_images_path", []))
            input_images_path = user_input.get("input_images_path", [])
            # we want to already populate the input image descriptions since although we would be able to obtain descriptions later on from the llm, the descriptions in the fashion dataset often include things that are not apparent visually - like the material
            input_images_descriptions = [""] * num_new_input_images # the multiplication is because append the other values later
            # save the input_images_path. some of the paths are for previously recommended images, so we take those from the state
            log.debug("Getting relevant images")
            for idx in updated_user_request.relevant_image_indexes:
                if idx < base_len:
                    input_images_path.append(state.input_images_path[idx])
                    input_images_descriptions.append("")  # we append empty strings so that we can figure out in later nodes, which images need a description - we don't re-use the description of the image that we have already - the llm generated one - since this loop, we might need a description of an aspect of the image that is different from the aspect described in the previous loop
                else:
                    input_images_path.append(state.recommended_clothes_image_paths[idx - base_len])
                    input_images_descriptions.append(state.recommended_clothes_descriptions[idx - base_len])
            log.debug("Got relevant images")
            # we don't expect the user input to have input image descriptions
            user_input['input_images_path'] = input_images_path
            user_input['input_images_descriptions'] = input_images_descriptions
        num_input_images = len(user_input.get("input_images_path", []))
        state_update = {
                "is_chat_start": False,
                # we don't actually trust or require the user provided presence of input images
                "has_input_images": num_input_images > 0,
                "num_times_critiqued": 0,
                "input_images_path": user_input.get("input_images_path", []),
                "input_text": user_input.get("input_text", ""),
                "input_images_descriptions": user_input.get("input_images_descriptions", [""] * num_input_images)
        }
        log.debug(f"State update: {json.dumps(state_update, indent=2)}")
        return state_update

    def quantifier_node(self, state: AgentState) -> AgentState:
        """Figures out how many recommendations need to be made to the user.

        Utilizes the underlying language model to parse the raw textual imperative and
        isolate the explicit numeric quantity requested by the user. If the user omits
        this parameter, the system defaults to a predefined systemic scalar.

        Args:
            state (AgentState): The current multidimensional state vector.

        Returns:
            AgentState: The updated state dictionary containing the
                `num_recommendations` scalar.
        """
        log.debug("Entered quantifier node.")
        prompt = self._cfg.prompts.quantifier_node.num_recommendations_prompt.format(
            user_request=state.input_text
        )
        structured_model = self._model.with_structured_output(NumRecommendations)
        log.debug("Invoking model.")
        response = structured_model.invoke(
            prompt, config=self._callback_config
        )  # text prompt - structured
        log.debug("Received response from model.")
        num_recommendations = (
            self._cfg.rag_pipeline.default_num_recommendations
            if response.num_recommendations == 0
            else response.num_recommendations
        )
        log.debug(
            f"Number of recommendations required by the user:\n{response.num_recommendations}"
        )
        return {"num_recommendations": num_recommendations}

    def intent_node(self, state: AgentState) -> AgentState:
        """Formulates specialized instructions for the downstream visual parsing phase.

        Synthesizes a precise operational directive instructing the Vision Lang Model
        on the specific topological or stylistic features it must extract from the
        multimodal visual matrices.

        Args:
            state (AgentState): The current multidimensional state vector.

        Returns:
            AgentState: The updated state dictionary populated with `vlm_instructions`.
        """
        log.debug("Entered intent node.")
        prompt = self._cfg.prompts.intent_node.request_focus_prompt.format(
            user_request=state.input_text
        )
        log.debug("Invoking model.")
        response = self._model.invoke(
            prompt, config=self._callback_config
        )  # text prompt - unstructured
        log.debug("Received response from model.")
        log.debug(f"Instructions for the VLM:\n{response.content}")
        return {"vlm_instructions": response.content}

    def vision_node(self, state: AgentState) -> AgentState:
        """Executes targeted visual feature extraction across the user-supplied image.

        Iterates sequentially over the provided array of visual inputs, leveraging the
        dynamically synthesized `vlm_instructions` to generate rigorous textual
        representations of the specified features within the images.

        Args:
            state (AgentState): The current multidimensional state vector.

        Returns:
            AgentState: The updated state dictionary appended with
                `input_images_descriptions`.
        """
        log.debug("Entered vision node.")
        structured_model = self._model.with_structured_output(SingleImageDescription)
        descr = state.input_images_descriptions
        for idx, image_path in enumerate(state.input_images_path):
            # if the image path already has a description - which might happen if this is not the first loop, then we skip generating the description - this only happens for images from the fashion database - not for user images since the intent capture for user images might be different, but for the fashion database image descriptions - they often capture things that are not apparent visually
            if state.input_images_descriptions[idx].strip():
                continue
            msg = get_image_prompt_message(
                image_path=image_path,
                text_prompt=self._cfg.prompts.vision_node.user_prompt.format(
                    image_focus_instructions=state.vlm_instructions
                ),
            )
            log.debug("Invoking model.")
            response = structured_model.invoke(
                msg, config=self._callback_config
            )  # vision prompt - structured
            log.debug("Received response from model.")
            descr[idx] = response.image_description
        log.debug(f"Descriptions obtained from vision node:\n{descr}")
        return {"input_images_descriptions": descr}

    def modifier_node(self, state: AgentState) -> AgentState:
        """Produces text descriptions of clothing items requested by user.

        Integrates both the original textual imperative and the newly extracted visual
        feature descriptions to generate an array of target apparel descriptions.
        Implements an iterative retry mechanism to strictly enforce the constraint
        dictated by `num_recommendations`.

        Args:
            state (AgentState): The current multidimensional state vector.

        Returns:
            AgentState: The updated state dictionary containing
                `required_clothes_descriptions`.
        """
        log.debug("Entered modifier node.")
        ref_descr = "\n".join(
            [
                f"{idx + 1}. {img_descr}"
                for idx, img_descr in enumerate(state.input_images_descriptions)
            ]
        )
        if state.has_input_images:
            prompt = self._cfg.prompts.modifier_node.images_present_prompt.format(
                reference_descriptions=ref_descr,
                user_request=state.input_text,
                num_recommendations=state.num_recommendations,
                critique_node_correction=state.critique_text,
            )
        else:
            prompt = self._cfg.prompts.modifier_node.images_absent_prompt.format(
                user_request=state.input_text,
                num_recommendations=state.num_recommendations,
                critique_node_correction=state.critique_text,
            )
        structured_model = self._model.with_structured_output(RequiredClothes)
        for attempt_num in range(self._cfg.rag_pipeline.num_recommendation_attempts):
            log.debug("Invoking model.")
            response = structured_model.invoke(
                prompt, config=self._callback_config
            )  # text prompt - structured
            log.debug("Received response from model.")
            if len(response.required_clothes_descriptions) == state.num_recommendations:
                break
            else:
                log.debug(
                    "Failed to get right number of recommendations from LLM after"
                    f"attempt number {attempt_num + 1}."
                )
        log.debug(
            "Required clothes descriptions obtained from modifier node:"
            f"\n{response.required_clothes_descriptions}"
        )
        return {"required_clothes_descriptions": response.required_clothes_descriptions}

    async def recommender_node(self, state: AgentState) -> AgentState:
        # TODO: adjust description - we no longer use recommended_clothes_images
        """Executes semantic similarity search within the vector embedding space.

        Transforms the target apparel descriptions into dense vector embeddings
        utilizing the SigLIP architecture. Subsequently executes a nearest-neighbor
        search operation across the vector database to extract the highest-scoring
        candidate indices via cosine similarity calculations.

        Args:
            state (AgentState): The current multidimensional state vector.

        Returns:
            AgentState: The updated state dictionary containing a deduplicated set of
                `recommended_clothes_images` indices.
        """
        # clothing items
        recommended_clothes_images = []
        log.debug("Entered recommender node.")
        tools = await self._client.get_tools()
        tools = [make_mistral_compatible(tool) for tool in tools if tool.name in self._cfg.rag_pipeline.mcp.llm_tool_names]
        agent = create_agent(
            model=self._model,
            tools=tools,
            response_format=MatchedImageId,
            debug=False,
        )
        template = self._cfg.prompts.recommender_node.match_clothes_prompt
        for descr in state.required_clothes_descriptions:
            prompt = template.format(item_description=descr)
            log.debug("Invoking agent.")
            response = await agent.ainvoke(
                {"messages": prompt}, config=self._callback_config
            )
            recommended_clothes_images.append(response['structured_response'].image_id)
            log.debug("Received response from agent.")
        # saving the images to path and storing the path - this is a bit of a repeat from human node, but for now does not matter
        tools = await self._client.get_tools()
        tool = get_tool_with_name(tools, self._cfg.rag_pipeline.mcp.db_tool_name)
        recommended_clothes_image_paths = []
        recommended_clothes_descriptions = []
        for img_index in recommended_clothes_images:
            response = await tool.ainvoke({"index": img_index})
            image_url = None
            for block in response:
                if 'base64' in block:
                    image_url = block['base64']
                elif 'text' in block:
                    image_descr = json.loads(block['text'])['description']
            if (image_url is None) or (image_descr is None):
                raise ValueError("Could not find either image or description")
            path = save_image_url_to_folder(self._cfg.rag_pipeline.temporary_images_folder, image_url)
            recommended_clothes_image_paths.append(path)
            recommended_clothes_descriptions.append(image_descr)
        log.debug(f"recommended_clothes_image_paths: {recommended_clothes_image_paths}\n recommended_clothes_descriptions: {recommended_clothes_descriptions}")
        return {"recommended_clothes_image_paths": recommended_clothes_image_paths, "recommended_clothes_descriptions": recommended_clothes_descriptions}

    def critique_node(self, state: AgentState) -> AgentState:
        """Critiques the recommendations and proposes suggestions.

        Checks if the recommendations are a good response to the user's queries. If
        yes, then we simply move on. If not, it specifically mentions how the
        modifier node should adjust its descriptions of recommended clothes so that
        the next attempt has a better chance of satisfying the users requests. There
        is a limited number of retries after which we continue anyways.

        Args:
            state (AgentState): The current multidimensional state vector.

        Returns:
            AgentState: The updated state dictionary containing the
                `num_times_critiqued` and `critique_text`.
        """
        # check if we are done with critiquing
        log.debug("Entered critique node.")
        if state.num_times_critiqued >= self._cfg.rag_pipeline.max_num_critiques:
            return {"num_times_critiqued": state.num_times_critiqued, "critique_text": ""}
        # generate the prompt
        prompts = []
        for img_path in state.input_images_path:
            prompts.append(("image", img_path))
        prompts.append(("text", f"All the above are user input images.\n\nThe user request is: {state.input_text}\n\n"))
        for idx, img_descr in enumerate(state.recommended_clothes_descriptions):
            prompts.append(("text", f"Description of Item {idx + 1}: {img_descr}\n\n"))
        prompts.append(("text", self._cfg.prompts.critique_node.critique_prompt))
        msg = get_multi_image_multi_prompt_message(prompts)
        # make and invoke the model
        structured_model = self._model.with_structured_output(CriticalEvaluation)
        log.debug("Invoking model")
        response = structured_model.invoke(msg)
        log.debug(f"Got model response: {response}")
        if response.satisfactory == "Yes":
            # don't change the number of times critique happened - we are only counting wrong recommendations
            return {"critique_text": ""}
        else:
            return {"num_times_critiqued": state.num_times_critiqued + 1, "critique_text": response.correction}

    def explanation_node(self, state: AgentState) -> AgentState:
        """Formulates analytical justifications elucidating the retrieval congruence.

        Correlates the database-retrieved image data with the initial multimodal query,
        tasking the model with producing an explicit, text-based logical defense of how
        the retrieved k elements satisfy the multidimensional input constraints.

        Args:
            state (AgentState): The current multidimensional state vector.

        Returns:
            AgentState: The updated state dictionary containing the
                `recommended_clothes_explanation` array.
        """
        log.debug("Entered explanation node.")
        # create prompt
        prompts = []
        for img_path in state.input_images_path:
            prompts.append(("image", img_path))
        prompts.append(("text", f"All the above are user input images.\n\nThe user request is: {state.input_text}\n\n"))
        for idx, img_descr in enumerate(state.recommended_clothes_descriptions):
            prompts.append(("image", state.recommended_clothes_image_paths[idx]))
            prompts.append(("text", f"Above is Recommended Image {idx + 1}.\nDescription of Item {idx + 1}: {img_descr}\n\n"))
        prompts.append(("text", self._cfg.prompts.explanation_node.explanation_prompt))
        msg = get_multi_image_multi_prompt_message(prompts)
        # invoke model
        log.debug("Invoking model")
        response = self._model.invoke(msg)
        log.debug(f"Got response: {response.content}")
        return {"recommended_clothes_explanation": response.content}

    def quantifier_node_router(
        self, state: AgentState
    ) -> Literal["intent_node", "modifier_node"]:
        """Evaluates conditional branching logic within the LangGraph DAG topology.

        Determines the execution path by inspecting the presence of visual input.
        If images exist, execution is routed to `intent_node`; otherwise, the graph
        bypasses visual processing and directly enters the `modifier_node`.

        Args:
            state (AgentState): The current multidimensional state vector.

        Returns:
            Literal["intent_node", "modifier_node"]: The deterministic string identifier
                of the subsequent vertex in the execution graph.
        """
        return "intent_node" if state.has_input_images else "modifier_node"

    def critique_node_router(self, state: AgentState) -> Literal["explanation_node", "modifier_node"]:
        """If the recommendation was good, we move on to explanation node, else modifier."""
        if (not state.critique_text.strip()) or state.num_times_critiqued > self._cfg.rag_pipeline.max_num_critiques:
            return "explanation_node"
        else:
            return "modifier_node"

    async def compile_graph(self, conn_string):
        self._connection = await aiosqlite.connect(conn_string)
        checkpointer = AsyncSqliteSaver(self._connection)
        builder = StateGraph(AgentState)
        # nodes
        builder.add_node("human_node", self.human_node)
        builder.add_node("quantifier_node", self.quantifier_node)
        builder.add_node("intent_node", self.intent_node)
        builder.add_node("vision_node", self.vision_node)
        builder.add_node("modifier_node", self.modifier_node)
        builder.add_node("recommender_node", self.recommender_node)
        builder.add_node("explanation_node", self.explanation_node)
        builder.add_node("critique_node", self.critique_node)
        # edges
        builder.add_edge(START, "human_node")
        builder.add_edge("human_node", "quantifier_node")
        builder.add_conditional_edges("quantifier_node", self.quantifier_node_router)
        builder.add_edge("intent_node", "vision_node")
        builder.add_edge("vision_node", "modifier_node")
        builder.add_edge("modifier_node", "recommender_node")
        builder.add_edge("recommender_node", "critique_node")
        builder.add_conditional_edges("critique_node", self.critique_node_router)
        builder.add_edge("explanation_node", "human_node")
        # compile and run
        self._graph = builder.compile(checkpointer=checkpointer)
        draw_langraph_topology(self._graph, self._cfg.rag_pipeline.node_diagram_path)

    async def ainvoke(self, initial_state, config):
        invocation_state = (
            None
            if self._cfg.rag_pipeline.persistence.resume_from_checkpoint
            else initial_state
        )
        return await self._graph.ainvoke(invocation_state, config)
    
    async def astream(self, input_state, config):
        """Streams node-level state updates from the compiled graph."""
        invocation_state = (
            None
            if self._cfg.rag_pipeline.persistence.resume_from_checkpoint
            else input_state
        )
        async for chunk in self._graph.astream(invocation_state, config, stream_mode="updates"):
            yield chunk

    async def close_connection(self):
        if self._connection:
            await self._connection.close()

# this is used for running the code without chainlit
@track_token_use
async def run_fashion_agent(cfg, callback_config):
    log.debug("running fashion agent")
    agent = FashionAgent(cfg, callback_config)
    try:
        await agent.compile_graph(cfg.rag_pipeline.persistence.db_path)
        config = {"configurable": {"thread_id": cfg.rag_pipeline.persistence.thread_id}}
        result = await agent.ainvoke({"is_chat_start": True}, config)
        while True:
            input_text = input("Enter `quit` to exit gracefully. Please provide input text: ")
            if input_text.strip().lower() == "quit":
                log.info("Exiting gracefully.")
                break
            num_input_images = int(input("Enter number of input images: "))
            input_images_path = []
            for img_num in range(num_input_images):
                path = input(f"Please provide the path to the input image {img_num + 1}: ")
                input_images_path.append(path)
            resume_payload = {"input_text": input_text, "input_images_path": input_images_path}
            result = await agent.ainvoke(Command(resume=resume_payload), config=config)
            if 'recommended_clothes_image_paths' in result:
                for path in result['recommended_clothes_image_paths']:
                    log.info(f"Recommended Image Path: {path}")
                log.info("Explanation: {result['recommended_clothes_explanation']}")
            else:
                log.error("ERROR: recommended_clothes_image_paths not in result, but expected")
    finally:
        await agent.close_connection()
