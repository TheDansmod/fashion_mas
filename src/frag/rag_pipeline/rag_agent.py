"""The Agentic RAG Pipeline / Graph setup."""

import json
from typing import Literal
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from loguru import logger as log

from frag.prompts import (
    CRITIQUE_PROMPT,
    EXPLANATION_PROMPT,
    HUMAN_INPUT_UPDATE_PROMPT,
    MATCH_CLOTHES_PROMPT,
    MODIFIER_IMAGES_ABSENT_PROMPT,
    MODIFIER_IMAGES_PRESENT_PROMPT,
    NUM_RECOMMENDATIONS_PROMPT,
    REQUEST_FOCUS_PROMPT,
    VISION_PROMPT,
)
from frag.rag_pipeline.llm_schemas import (
    AgentState,
    CriticalEvaluation,
    MatchedImageId,
    NumRecommendations,
    RequiredClothes,
    SingleImageDescription,
    UpdatedUserRequest,
)
from frag.runtime import Resources
from frag.settings import AgentSettings
from frag.tokens import track_tokens
from frag.tools.base import ToolsClient
from frag.utils.common_utils import (
    draw_langraph_topology,
    get_image_prompt_message,
    get_multi_image_multi_prompt_message,
    get_multi_image_prompt_message,
    save_image_b64,
)


class FashionAgent:
    """Constructs and orchestrates the stateful DAG for multimodal retrieval.

    This primary interface initializes the vector database connection, establishes the
    underlying embedding functions, and connects the disparate computational nodes
    (quantification, intent synthesis, visual extraction, modification, and retrieval)
    via LangGraph topologies.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: ToolsClient,
        checkpointer: BaseCheckpointSaver,
        settings: AgentSettings,
        callbacks: list | None = None,
    ) -> None:
        """Store the dependencies and compile the graph.

        Args:
            llm: Chat model used by every node.
            tools: Catalogue tools (local or MCP-backed).
            checkpointer: LangGraph checkpointer; conversation state is keyed by the
                `thread_id` in the invocation config.
            settings: Agent behaviour settings.
            callbacks: LangChain callbacks passed to every LLM call, e.g. a
                `UsageMetadataCallbackHandler` for token tracking.
        """
        self._model = llm
        self._tools = tools
        self._settings = settings
        self._callback_config = {"callbacks": callbacks or []}
        self._graph = self._build_graph(checkpointer)

    async def human_node(
        self,
        state: AgentState,
    ) -> AgentState:
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
            prompt = HUMAN_INPUT_UPDATE_PROMPT.format(
                original_image_descriptions=org_img_descr,
                original_text_input=state.input_text,
                response_image_descriptions=resp_img_descr,
                new_text_input=user_input.get("input_text", ""),
            )
            msg = get_multi_image_prompt_message(
                user_input.get("input_images_path", []), text_prompt=prompt
            )
            # create and invoke agent
            log.debug("Invoking agent")
            agent = create_agent(
                model=self._model, response_format=UpdatedUserRequest, debug=False
            )
            response = await agent.ainvoke(
                {"messages": msg}, config=self._callback_config
            )
            log.debug("Got agent response")
            # update the user input so that the normal state update (which assumes first chat, can also serve for non-first chat situations
            updated_user_request = response["structured_response"]
            user_input["input_text"] = updated_user_request.updated_user_query
            num_new_input_images = len(user_input.get("input_images_path", []))
            input_images_path = user_input.get("input_images_path", [])
            # we want to already populate the input image descriptions since although we would be able to obtain descriptions later on from the llm, the descriptions in the fashion dataset often include things that are not apparent visually - like the material
            input_images_descriptions = (
                [""] * num_new_input_images
            )  # the multiplication is because append the other values later
            # save the input_images_path. some of the paths are for previously recommended images, so we take those from the state
            log.debug("Getting relevant images")
            # the indexes come from the LLM, so they are not trusted to be in range
            num_previous_images = base_len + len(state.recommended_clothes_image_paths)
            for idx in updated_user_request.relevant_image_indexes:
                if not 0 <= idx < num_previous_images:
                    log.warning(
                        "Ignoring out-of-range image index {} (have {} images).",
                        idx,
                        num_previous_images,
                    )
                    continue
                if idx < base_len:
                    input_images_path.append(state.input_images_path[idx])
                    input_images_descriptions.append(
                        ""
                    )  # we append empty strings so that we can figure out in later nodes, which images need a description - we don't re-use the description of the image that we have already - the llm generated one - since this loop, we might need a description of an aspect of the image that is different from the aspect described in the previous loop
                else:
                    input_images_path.append(
                        state.recommended_clothes_image_paths[idx - base_len]
                    )
                    input_images_descriptions.append(
                        state.recommended_clothes_descriptions[idx - base_len]
                    )
            log.debug("Got relevant images")
            # we don't expect the user input to have input image descriptions
            user_input["input_images_path"] = input_images_path
            user_input["input_images_descriptions"] = input_images_descriptions
        num_input_images = len(user_input.get("input_images_path", []))
        state_update = {
            "is_chat_start": False,
            # we don't actually trust or require the user provided presence of input images
            "has_input_images": num_input_images > 0,
            "num_times_critiqued": 0,
            "input_images_path": user_input.get("input_images_path", []),
            "input_text": user_input.get("input_text", ""),
            "input_images_descriptions": user_input.get(
                "input_images_descriptions", [""] * num_input_images
            ),
        }
        # was using string substitution earlier, but since I switched to loguru, I am using their f-string string format equivalent
        log.debug("State update: {}", json.dumps(state_update, indent=2))
        return state_update

    def quantifier_node(
        self,
        state: AgentState,
    ) -> AgentState:
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
        prompt = NUM_RECOMMENDATIONS_PROMPT.format(user_request=state.input_text)
        structured_model = self._model.with_structured_output(NumRecommendations)
        log.debug("Invoking model.")
        response = structured_model.invoke(
            prompt, config=self._callback_config
        )  # text prompt - structured
        log.debug("Received response from model.")
        log.debug(
            "default num recommendations: {}",
            self._settings.default_num_recommendations,
        )
        log.debug("response.num_recommendations: {}", response.num_recommendations)
        num_recommendations = (
            self._settings.default_num_recommendations
            if response.num_recommendations == 0
            else response.num_recommendations
        )
        log.debug(
            f"Number of recommendations required by the user:\n{num_recommendations}"
        )
        return {"num_recommendations": num_recommendations}

    def intent_node(
        self,
        state: AgentState,
    ) -> AgentState:
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
        prompt = REQUEST_FOCUS_PROMPT.format(user_request=state.input_text)
        log.debug("Invoking model.")
        response = self._model.invoke(
            prompt, config=self._callback_config
        )  # text prompt - unstructured
        log.debug("Received response from model.")
        log.debug(f"Instructions for the VLM:\n{response.content}")
        return {"vlm_instructions": response.content}

    def vision_node(
        self,
        state: AgentState,
    ) -> AgentState:
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
        descr = state.input_images_descriptions[:]  # create copy rather then reference
        for idx, image_path in enumerate(state.input_images_path):
            # if the image path already has a description - which might happen if this is not the first loop, then we skip generating the description - this only happens for images from the fashion database - not for user images since the intent capture for user images might be different, but for the fashion database image descriptions - they often capture things that are not apparent visually
            if state.input_images_descriptions[idx].strip():
                continue
            msg = get_image_prompt_message(
                image_path=image_path,
                text_prompt=VISION_PROMPT.format(
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

    def modifier_node(
        self,
        state: AgentState,
    ) -> AgentState:
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
            prompt = MODIFIER_IMAGES_PRESENT_PROMPT.format(
                reference_descriptions=ref_descr,
                user_request=state.input_text,
                num_recommendations=state.num_recommendations,
                critique_node_correction=state.critique_text,
            )
        else:
            prompt = MODIFIER_IMAGES_ABSENT_PROMPT.format(
                user_request=state.input_text,
                num_recommendations=state.num_recommendations,
                critique_node_correction=state.critique_text,
            )
        structured_model = self._model.with_structured_output(RequiredClothes)
        for attempt_num in range(self._settings.num_recommendation_attempts):
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

    async def recommender_node(
        self,
        state: AgentState,
    ) -> AgentState:
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
        # TODO: adjust description - we no longer use recommended_clothes_images
        # clothing items
        recommended_clothes_images = []
        log.debug("Entered recommender node.")
        tools = await self._tools.get_llm_tools()
        agent = create_agent(
            model=self._model,
            tools=tools,
            response_format=MatchedImageId,
            debug=False,
        )
        for descr in state.required_clothes_descriptions:
            prompt = MATCH_CLOTHES_PROMPT.format(item_description=descr)
            log.debug("Invoking agent.")
            response = await agent.ainvoke(
                {"messages": prompt}, config=self._callback_config
            )
            recommended_clothes_images.append(response["structured_response"].image_id)
            log.debug("Received response from agent.")
        recommended_clothes_image_paths = []
        recommended_clothes_descriptions = []
        for item_id in recommended_clothes_images:
            # the id is chosen by the LLM and may not exist in the catalogue
            # (hallucinated, or outside the loaded subset): skip such items rather than
            # failing the whole turn
            try:
                item = await self._tools.get_item(item_id)
            # failures differ per tool backend (S3 errors, MCP ToolException); any of
            # them means skip this item
            except Exception:  # noqa: BLE001
                log.opt(exception=True).warning(
                    "Skipping recommended item {}: catalogue lookup failed.", item_id
                )
                continue
            if item is None:
                log.warning(
                    "Skipping recommended item {}: not in the catalogue.", item_id
                )
                continue
            path = save_image_b64(self._settings.temp_dir, item.image_b64)
            recommended_clothes_image_paths.append(path)
            recommended_clothes_descriptions.append(item.description)
        log.debug(
            f"recommended_clothes_image_paths: {recommended_clothes_image_paths}\n recommended_clothes_descriptions: {recommended_clothes_descriptions}"
        )
        return {
            "recommended_clothes_image_paths": recommended_clothes_image_paths,
            "recommended_clothes_descriptions": recommended_clothes_descriptions,
        }

    def critique_node(
        self,
        state: AgentState,
    ) -> AgentState:
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
        if state.num_times_critiqued >= self._settings.max_num_critiques:
            return {
                "num_times_critiqued": state.num_times_critiqued,
                "critique_text": "",
            }
        # generate the prompt
        prompts = []
        for img_path in state.input_images_path:
            prompts.append(("image", img_path))
        prompts.append(
            (
                "text",
                f"All the above are user input images.\n\nThe user request is: {state.input_text}\n\n",
            )
        )
        for idx, img_descr in enumerate(state.recommended_clothes_descriptions):
            prompts.append(("text", f"Description of Item {idx + 1}: {img_descr}\n\n"))
        prompts.append(("text", CRITIQUE_PROMPT))
        msg = get_multi_image_multi_prompt_message(prompts)
        # make and invoke the model
        structured_model = self._model.with_structured_output(CriticalEvaluation)
        log.debug("Invoking model")
        response = structured_model.invoke(msg, config=self._callback_config)
        log.debug(f"Got model response: {response}")
        if response.satisfactory == "Yes":
            # don't change the number of times critique happened - we are only counting wrong recommendations
            return {"critique_text": ""}
        else:
            return {
                "num_times_critiqued": state.num_times_critiqued + 1,
                "critique_text": response.correction,
            }

    def explanation_node(
        self,
        state: AgentState,
    ) -> AgentState:
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
        prompts.append(
            (
                "text",
                f"All the above are user input images.\n\nThe user request is: {state.input_text}\n\n",
            )
        )
        for idx, img_descr in enumerate(state.recommended_clothes_descriptions):
            prompts.append(("image", state.recommended_clothes_image_paths[idx]))
            prompts.append(
                (
                    "text",
                    f"Above is Recommended Image {idx + 1}.\nDescription of Item {idx + 1}: {img_descr}\n\n",
                )
            )
        prompts.append(("text", EXPLANATION_PROMPT))
        msg = get_multi_image_multi_prompt_message(prompts)
        # invoke model
        log.debug("Invoking model")
        response = self._model.invoke(msg, config=self._callback_config)
        log.debug(f"Got response: {response.content}")
        return {"recommended_clothes_explanation": response.content}

    def quantifier_node_router(
        self,
        state: AgentState,
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

    def critique_node_router(
        self,
        state: AgentState,
    ) -> Literal["explanation_node", "modifier_node"]:
        """Route back to the modifier node if there is a critique, else explain.

        The critique node itself enforces the retry limit: once `max_num_critiques` is
        reached it returns an empty `critique_text`, which routes to explanation.
        """
        if state.critique_text.strip():
            return "modifier_node"
        return "explanation_node"

    def _build_graph(self, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
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
        graph = builder.compile(checkpointer=checkpointer)
        if self._settings.draw_graph_path:
            draw_langraph_topology(graph, self._settings.draw_graph_path)
        return graph

    async def ainvoke(self, initial_state, config):
        return await self._graph.ainvoke(initial_state, config)

    async def astream(self, input_state, config):
        """Streams node-level state updates from the compiled graph."""
        async for chunk in self._graph.astream(
            input_state, config, stream_mode="updates"
        ):
            yield chunk


# this is used for running the code without chainlit
# there is no resume from checkpoint for this - you always start fresh
# chainlit handles the resume when required
# here we use uuid to have fresh start everytime
async def run_fashion_agent(resources: Resources) -> None:
    """Terminal chat loop (moves to frag.cli in cleanup.md Phase 8)."""
    with track_tokens(resources.settings.token_usage_csv) as usage:
        await _chat_loop(resources, usage)


async def _chat_loop(resources: Resources, usage) -> None:  # noqa: ANN001  callback handler
    log.debug("running fashion agent")
    fashion_agent = FashionAgent(
        llm=resources.llm,
        tools=resources.tools,
        checkpointer=resources.checkpointer,
        settings=resources.settings.agent,
        callbacks=[usage],
    )
    config = {"configurable": {"thread_id": str(uuid4())}}
    result = await fashion_agent.ainvoke({"is_chat_start": True}, config)
    while True:
        input_text = input(
            "Enter `quit` to exit gracefully. Please provide input text: "
        )
        if input_text.strip().lower() == "quit":
            log.info("Exiting gracefully.")
            break
        num_input_images = int(input("Enter number of input images: "))
        input_images_path = []
        for img_num in range(num_input_images):
            path = input(f"Please provide the path to the input image {img_num + 1}: ")
            input_images_path.append(path)
        resume_payload = {
            "input_text": input_text,
            "input_images_path": input_images_path,
        }
        result = await fashion_agent.ainvoke(
            Command(resume=resume_payload), config=config
        )
        if "recommended_clothes_image_paths" in result:
            for path in result["recommended_clothes_image_paths"]:
                log.info(f"Recommended Image Path: {path}")
            log.info(f"Explanation: {result['recommended_clothes_explanation']}")
        else:
            log.error(
                "ERROR: recommended_clothes_image_paths not in result, but expected"
            )
