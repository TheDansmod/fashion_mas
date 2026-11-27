"""The fashion-recommendation agent: one LangGraph run per user message.

START -> intake -> quantifier -> [intent -> vision] -> modifier -> recommender
      -> critique -> (modifier again | explanation) -> END

State is checkpointed per `thread_id`, so a later message on the same thread is
treated as a follow-up to the earlier ones.
"""

import asyncio
import base64
import json
import re
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger as log

from frag.agent.schemas import (
    CriticalEvaluation,
    MatchedImageId,
    NumRecommendations,
    RequiredClothes,
    SingleImageDescription,
    UpdatedUserRequest,
)
from frag.agent.state import AgentState
from frag.messages import build_message
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
from frag.settings import AgentSettings
from frag.tools.base import SEMANTIC_SEARCH, ToolsClient

Update = dict[str, Any]

CRITIQUE_PREFIX = "Feedback on the previous attempt:"


def _critique_section(critique_text: str) -> str:
    """The critique as inserted into the modifier prompt (empty if there is none)."""
    critique_text = critique_text.strip()
    return f"{CRITIQUE_PREFIX} {critique_text}" if critique_text else ""


def _normalise_type(item_type: str) -> str:
    # "Shoes" / "shoe" / " shoes " count as the same type
    return item_type.strip().lower().removesuffix("s")


def missing_item_types(response: RequiredClothes, requested: list[str]) -> list[str]:
    """Requested item types that no item of the modifier's answer covers.

    An item covers a type if it is labelled with it, or if its description names it
    (a model may label loafers "shoes" while the user asked for "loafers").
    """
    labels = {_normalise_type(item.item_type) for item in response.items}
    text = " ".join(item.description for item in response.items).lower()
    return [
        t
        for t in requested
        if _normalise_type(t) not in labels
        and not re.search(rf"\b{re.escape(_normalise_type(t))}", text)
    ]


def _item_types_line(item_types: list[str]) -> str:
    if not item_types:
        return "Item types the user named: none."
    return f"Item types the user named (cover each one): {', '.join(item_types)}."


def offered_item_ids(messages: list[BaseMessage]) -> set[int]:
    """Ids of the items that `semantic_search` returned in an agent's messages."""
    ids = set()
    for message in messages:
        if not isinstance(message, ToolMessage) or message.name != SEMANTIC_SEARCH:
            continue
        blocks = message.content if isinstance(message.content, list) else []
        for block in blocks:
            if not (isinstance(block, dict) and block.get("type") == "text"):
                continue
            try:
                metadata = json.loads(block["text"])
            except json.JSONDecodeError:
                continue
            if isinstance(metadata, dict) and "id" in metadata:
                ids.add(int(metadata["id"]))
    return ids


def chosen_item_id(response: dict[str, Any], offered: set[int]) -> int | None:
    """The item id the tool-using agent settled on, or None if it gave none.

    Small local models often answer in prose instead of calling the structured
    response tool. In that case the first offered id mentioned in the agent's final
    message is taken; it is still checked against `offered` by the caller.
    """
    structured = response.get("structured_response")
    if structured is not None:
        return structured.image_id
    final = response["messages"][-1] if response["messages"] else None
    if not isinstance(final, AIMessage):
        return None
    for match in re.finditer(r"\d+", final.text):
        if int(match.group()) in offered:
            log.debug("Took id {} from the agent's prose answer.", match.group())
            return int(match.group())
    return None


def _invalid_id_message(item_id: int | None, offered: set[int]) -> HumanMessage:
    if not offered:
        text = (
            "You did not run semantic_search. Search the catalogue first, then answer "
            "with one of the returned ids."
        )
    else:
        choices = ", ".join(str(i) for i in sorted(offered))
        problem = (
            "You did not answer with an item id."
            if item_id is None
            else f"The id {item_id} was not returned by semantic_search."
        )
        text = f"{problem} Answer with one of the ids it returned: {choices}."
    return HumanMessage(content=text)


def _request_parts(state: AgentState) -> list[str | Path]:
    """Reference images (if any) and the request, as message parts.

    Used by the critique and explanation nodes. The images are labelled as references
    because after a follow-up they include earlier recommendations the user picked,
    not only uploads.
    """
    parts: list[str | Path] = [Path(path) for path in state.input_images_path]
    if parts:
        parts.append(
            "Above are the reference images: images the user uploaded, or products "
            "they picked from earlier recommendations.\n\n"
        )
    else:
        # otherwise small models take the first recommended image for an upload
        parts.append("The user did not upload or pick any images.\n\n")
    parts.append(f"User request: {state.input_text}\n\n")
    return parts


def save_image_b64(folder: str, image_b64: str) -> str:
    """Write a base64-encoded PNG to a uniquely named file in `folder`."""
    directory = Path(folder)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid.uuid4()}.png"
    path.write_bytes(base64.b64decode(image_b64))
    return str(path)


def draw_graph(graph: CompiledStateGraph, path: str) -> None:
    """Draw the compiled graph as a PNG (calls the mermaid.ink web API)."""
    png = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    Path(path).write_bytes(png)


class FashionAgent:
    """Builds the graph over an LLM, the catalogue tools and a checkpointer."""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: ToolsClient,
        checkpointer: BaseCheckpointSaver,
        settings: AgentSettings,
    ) -> None:
        """Store the dependencies and compile the graph.

        Args:
            llm: Chat model used by every node.
            tools: Catalogue tools (local or MCP-backed).
            checkpointer: LangGraph checkpointer; conversation state is keyed by the
                thread id passed to `run_turn` / `stream_turn`.
            settings: Agent behaviour settings.
        """
        self._model = llm
        self._tools = tools
        self._settings = settings
        self._graph = self._build_graph(checkpointer)

    # --- nodes ------------------------------------------------------------------

    async def intake_node(self, state: AgentState, config: RunnableConfig) -> Update:
        """Turn the new message into the request the rest of the graph works on.

        On the first message of a thread the message is the request. On later
        messages the LLM rewrites it into a standalone request using the previous
        request, and picks which earlier images (uploads or recommendations) are
        still relevant. Per-turn fields are reset either way.
        """
        log.debug("Entered intake node.")
        new_text = state.new_input_text
        new_images = list(state.new_input_images)
        if state.input_text:
            text, images, descriptions = await self._follow_up(
                state, new_text, new_images, config
            )
        else:
            text, images = new_text, new_images
            descriptions = [""] * len(images)
        update = {
            "input_text": text,
            "input_images_path": images,
            # empty descriptions are filled in by the vision node
            "input_images_descriptions": descriptions,
            "has_input_images": bool(images),
            "num_times_critiqued": 0,
            "critique_text": "",
        }
        log.debug("Intake update: {}", json.dumps(update, indent=2))
        return update

    async def _follow_up(
        self,
        state: AgentState,
        new_text: str,
        new_images: list[str],
        config: RunnableConfig,
    ) -> tuple[str, list[str], list[str]]:
        """Merge a follow-up message with the previous turn."""
        num_inputs = len(state.input_images_descriptions)
        # the LLM sees the earlier images by description (cheaper than the images);
        # recommendations are numbered after the earlier inputs
        original = "\n".join(
            f"Index {i}: {d}" for i, d in enumerate(state.input_images_descriptions)
        )
        recommended = "\n".join(
            f"Index {num_inputs + i}: {d}"
            for i, d in enumerate(state.recommended_clothes_descriptions)
        )
        prompt = HUMAN_INPUT_UPDATE_PROMPT.format(
            original_image_descriptions=original,
            original_text_input=state.input_text,
            response_image_descriptions=recommended,
            new_text_input=new_text,
        )
        # a plain structured call: no tools are involved, and small local models
        # often skip the structured-response tool of a tool-calling agent
        model = self._model.with_structured_output(UpdatedUserRequest)
        request: UpdatedUserRequest = await model.ainvoke(
            build_message([*(Path(p) for p in new_images), prompt]), config=config
        )

        images = list(new_images)
        descriptions = [""] * len(new_images)
        num_previous = num_inputs + len(state.recommended_clothes_image_paths)
        for idx in request.relevant_image_indexes:
            # the indexes come from the LLM, so they may be out of range
            if not 0 <= idx < num_previous:
                log.warning(
                    "Ignoring out-of-range image index {} (have {} images).",
                    idx,
                    num_previous,
                )
                continue
            if idx < num_inputs:
                images.append(state.input_images_path[idx])
                # described again: this request may need a different aspect of it
                descriptions.append("")
            else:
                rec = idx - num_inputs
                images.append(state.recommended_clothes_image_paths[rec])
                # catalogue descriptions include details that aren't visible
                # (e.g. material), so they are kept
                descriptions.append(state.recommended_clothes_descriptions[rec])
        return request.updated_user_query, images, descriptions

    async def quantifier_node(
        self, state: AgentState, config: RunnableConfig
    ) -> Update:
        """Find how many items the user wants, and which item types they named.

        Without a stated number the configured default is used. Either way there is
        at least one item per named type.
        """
        log.debug("Entered quantifier node.")
        model = self._model.with_structured_output(NumRecommendations)
        response = await model.ainvoke(
            NUM_RECOMMENDATIONS_PROMPT.format(user_request=state.input_text),
            config=config,
        )
        # de-duplicated, keeping the order; blank entries dropped
        item_types = list(
            dict.fromkeys(t.strip().lower() for t in response.item_types if t.strip())
        )
        num = response.num_recommendations or self._settings.default_num_recommendations
        num = max(num, len(item_types))
        log.debug("Number of recommendations: {}; item types: {}", num, item_types)
        return {"num_recommendations": num, "requested_item_types": item_types}

    async def intent_node(self, state: AgentState, config: RunnableConfig) -> Update:
        """Write instructions for what the vision model should look at."""
        log.debug("Entered intent node.")
        response = await self._model.ainvoke(
            REQUEST_FOCUS_PROMPT.format(user_request=state.input_text), config=config
        )
        log.debug("Instructions for the vision model: {}", response.content)
        return {"vlm_instructions": response.content}

    async def vision_node(self, state: AgentState, config: RunnableConfig) -> Update:
        """Describe each input image that has no description yet."""
        log.debug("Entered vision node.")
        model = self._model.with_structured_output(SingleImageDescription)
        descriptions = list(state.input_images_descriptions)
        prompt = VISION_PROMPT.format(image_focus_instructions=state.vlm_instructions)
        for idx, image_path in enumerate(state.input_images_path):
            if descriptions[idx].strip():
                continue
            response = await model.ainvoke(
                build_message([Path(image_path), prompt]), config=config
            )
            descriptions[idx] = response.image_description
        log.debug("Image descriptions: {}", descriptions)
        return {"input_images_descriptions": descriptions}

    async def modifier_node(self, state: AgentState, config: RunnableConfig) -> Update:
        """Describe the items to search for, one description per recommendation.

        Retries up to `num_recommendation_attempts` times until the answer has the
        right number of descriptions and covers every requested item type, then uses
        the best answer. A critique from the previous attempt is included in the
        prompt.
        """
        log.debug("Entered modifier node.")
        common = {
            "user_request": state.input_text,
            "num_recommendations": state.num_recommendations,
            "item_types": _item_types_line(state.requested_item_types),
            "critique_node_correction": _critique_section(state.critique_text),
        }
        if state.has_input_images:
            references = "\n".join(
                f"{i + 1}. {d}" for i, d in enumerate(state.input_images_descriptions)
            )
            prompt = MODIFIER_IMAGES_PRESENT_PROMPT.format(
                reference_descriptions=references, **common
            )
        else:
            prompt = MODIFIER_IMAGES_ABSENT_PROMPT.format(**common)
        model = self._model.with_structured_output(RequiredClothes)
        best, best_rank = None, None
        for attempt in range(1, self._settings.num_recommendation_attempts + 1):
            response: RequiredClothes = await model.ainvoke(prompt, config=config)
            missing = missing_item_types(response, state.requested_item_types)
            count_ok = len(response.items) == state.num_recommendations
            # prefer covering the item types, then the right count
            rank = (not missing, count_ok)
            if best_rank is None or rank > best_rank:
                best, best_rank = response, rank
            if rank == (True, True):
                break
            log.debug(
                "Attempt {}: {} descriptions (wanted {}), missing item types {}.",
                attempt,
                len(response.items),
                state.num_recommendations,
                missing,
            )
        descriptions = [item.description for item in best.items]
        log.debug("Item descriptions: {}", descriptions)
        return {"required_clothes_descriptions": descriptions}

    async def _match_item(
        self,
        agent: CompiledStateGraph,
        description: str,
        config: RunnableConfig,
    ) -> int | None:
        """Ask the tool-using agent for the catalogue item that best fits `description`.

        The chosen id (structured, or mentioned in a prose answer) must be one that
        `semantic_search` returned in this conversation; otherwise the agent is
        corrected once, and the item is skipped if the second answer is invalid too.
        """
        messages: list[BaseMessage] | str = MATCH_CLOTHES_PROMPT.format(
            item_description=description
        )
        for attempt in (1, 2):
            response = await agent.ainvoke({"messages": messages}, config=config)
            offered = offered_item_ids(response["messages"])
            item_id = chosen_item_id(response, offered)
            if item_id is not None and item_id in offered:
                return item_id
            log.warning(
                "Attempt {}: the agent's answer {} is not an id semantic_search "
                "returned for {!r}.",
                attempt,
                item_id,
                description,
            )
            messages = [*response["messages"], _invalid_id_message(item_id, offered)]
        return None

    async def recommender_node(
        self, state: AgentState, config: RunnableConfig
    ) -> Update:
        """Find a catalogue item for each description and save its image locally.

        One tool-using agent call per description, run concurrently (the shared rate
        limiter throttles them). Items that can't be matched or fetched are skipped,
        as are repeats of a product already chosen (same id, or another photo of it
        with the same description).
        """
        log.debug("Entered recommender node.")
        agent = create_agent(
            model=self._model,
            tools=await self._tools.get_llm_tools(),
            response_format=MatchedImageId,
            # don't inherit the graph's checkpointer: this inner agent's messages hold
            # the base64 images of every search, which are only needed within this
            # node and exceed DynamoDB's 400 KB item limit
            checkpointer=False,
        )
        matched = await asyncio.gather(
            *(
                self._match_item(agent, description, config)
                for description in state.required_clothes_descriptions
            )
        )
        # dict.fromkeys keeps the first occurrence and the order
        item_ids = [i for i in dict.fromkeys(matched) if i is not None]
        paths, descriptions = [], []
        for item_id in item_ids:
            try:
                item = await self._tools.get_item(item_id)
            # failures differ per tool backend (S3 errors, MCP ToolException); any of
            # them means skip this item
            except Exception:  # noqa: BLE001  see the comment above
                log.opt(exception=True).warning(
                    "Skipping recommended item {}: catalogue lookup failed.", item_id
                )
                continue
            if item is None:
                log.warning(
                    "Skipping recommended item {}: not in the catalogue.", item_id
                )
                continue
            # the catalogue has several photos of one product under different ids,
            # with the same description: show the product once
            if item.description in descriptions:
                log.debug("Skipping item {}: same product as an earlier one.", item_id)
                continue
            paths.append(save_image_b64(self._settings.temp_dir, item.image_b64))
            descriptions.append(item.description)
        log.debug("Recommended items: {}", descriptions)
        return {
            "recommended_clothes_image_paths": paths,
            "recommended_clothes_descriptions": descriptions,
        }

    async def critique_node(self, state: AgentState, config: RunnableConfig) -> Update:
        """Check the recommendations against the request; give a correction if not.

        After `max_num_critiques` rejections in a turn the recommendations are
        accepted as they are.
        """
        log.debug("Entered critique node.")
        if state.num_times_critiqued >= self._settings.max_num_critiques:
            return {"critique_text": ""}
        parts = _request_parts(state)
        parts.append(_item_types_line(state.requested_item_types) + "\n\n")
        parts.extend(
            f"Recommended Item {i + 1}: {d}\n\n"
            for i, d in enumerate(state.recommended_clothes_descriptions)
        )
        parts.append(CRITIQUE_PROMPT)
        model = self._model.with_structured_output(CriticalEvaluation)
        response = await model.ainvoke(build_message(parts), config=config)
        log.debug("Critique: {}", response)
        if response.satisfactory == "Yes":
            # only rejections count towards the limit
            return {"critique_text": ""}
        return {
            "num_times_critiqued": state.num_times_critiqued + 1,
            "critique_text": response.correction,
        }

    async def explanation_node(
        self, state: AgentState, config: RunnableConfig
    ) -> Update:
        """Explain how the recommended items fit the request."""
        log.debug("Entered explanation node.")
        parts = _request_parts(state)
        for i, (path, description) in enumerate(
            zip(
                state.recommended_clothes_image_paths,
                state.recommended_clothes_descriptions,
                strict=True,
            )
        ):
            parts.append(Path(path))
            parts.append(f"Above is Recommended Item {i + 1}: {description}\n\n")
        parts.append(EXPLANATION_PROMPT)
        response = await self._model.ainvoke(build_message(parts), config=config)
        return {"recommended_clothes_explanation": response.content}

    # --- routing ----------------------------------------------------------------

    def quantifier_node_router(
        self, state: AgentState
    ) -> Literal["intent_node", "modifier_node"]:
        """Describe the images first if there are any."""
        return "intent_node" if state.has_input_images else "modifier_node"

    def critique_node_router(
        self, state: AgentState
    ) -> Literal["explanation_node", "modifier_node"]:
        """Go back to the modifier if there is a critique, else explain.

        The critique node enforces the retry limit by returning an empty critique.
        """
        if state.critique_text.strip():
            return "modifier_node"
        return "explanation_node"

    def _build_graph(self, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
        builder = StateGraph(AgentState)
        builder.add_node("intake_node", self.intake_node)
        builder.add_node("quantifier_node", self.quantifier_node)
        builder.add_node("intent_node", self.intent_node)
        builder.add_node("vision_node", self.vision_node)
        builder.add_node("modifier_node", self.modifier_node)
        builder.add_node("recommender_node", self.recommender_node)
        builder.add_node("critique_node", self.critique_node)
        builder.add_node("explanation_node", self.explanation_node)
        builder.add_edge(START, "intake_node")
        builder.add_edge("intake_node", "quantifier_node")
        builder.add_conditional_edges("quantifier_node", self.quantifier_node_router)
        builder.add_edge("intent_node", "vision_node")
        builder.add_edge("vision_node", "modifier_node")
        builder.add_edge("modifier_node", "recommender_node")
        builder.add_edge("recommender_node", "critique_node")
        builder.add_conditional_edges("critique_node", self.critique_node_router)
        builder.add_edge("explanation_node", END)
        graph = builder.compile(checkpointer=checkpointer)
        if self._settings.draw_graph_path:
            # uses the mermaid.ink web API, so it is opt-in
            draw_graph(graph, self._settings.draw_graph_path)
        return graph

    # --- running ----------------------------------------------------------------

    @staticmethod
    def _turn_args(
        text: str,
        image_paths: list[str],
        thread_id: str,
        callbacks: list[BaseCallbackHandler] | None,
    ) -> tuple[dict[str, Any], RunnableConfig]:
        turn_input = {"new_input_text": text, "new_input_images": list(image_paths)}
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id},
            "callbacks": callbacks or [],
        }
        return turn_input, config

    async def run_turn(
        self,
        text: str,
        image_paths: list[str],
        thread_id: str,
        callbacks: list[BaseCallbackHandler] | None = None,
    ) -> dict[str, Any]:
        """Handle one user message; return the final state of the thread.

        `callbacks` (e.g. a token-usage handler) reach every LLM call of the turn.
        """
        turn_input, config = self._turn_args(text, image_paths, thread_id, callbacks)
        return await self._graph.ainvoke(turn_input, config)

    async def stream_turn(
        self,
        text: str,
        image_paths: list[str],
        thread_id: str,
        callbacks: list[BaseCallbackHandler] | None = None,
    ) -> AsyncIterator[dict[str, Update]]:
        """Like `run_turn`, but yield `{node name: state update}` as each node ends."""
        turn_input, config = self._turn_args(text, image_paths, thread_id, callbacks)
        async for chunk in self._graph.astream(
            turn_input, config, stream_mode="updates"
        ):
            yield chunk
