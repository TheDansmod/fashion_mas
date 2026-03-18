# this was taken from gemini
import chainlit as cl
from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# 1. Define the state for the graph
class State(TypedDict):
    lastinput: str
    image_path: Optional[str]
    image_mime: Optional[str]

# 2. Define the node where human interaction happens
def human_node(state: State):
    # The interrupt pauses execution. 
    # We now expect a dictionary containing both text and optional image info.
    user_payload = interrupt("Waiting for user input...")
    
    # We update the state with the text and any attached image metadata
    return {
        "lastinput": user_payload.get("text", ""),
        "image_path": user_payload.get("image_path"),
        "image_mime": user_payload.get("image_mime")
    }

# 3. Define the backend/agent processing node
def agent_node(state: State):
    # In a real app, this is where you would load the image from state["image_path"] 
    # and pass it along with state["lastinput"] to a vision-capable LLM.
    return state 

# 4. Define conditional routing
def route_next(state: State) -> Literal["__end__", "agent_node"]: # type: ignore
    if state.get("lastinput", "").strip().lower() == "quit":
        return END
    return "agent_node"

# 5. Build and compile the graph
builder = StateGraph(State)
builder.add_node("human_node", human_node)
builder.add_node("agent_node", agent_node)
builder.add_edge(START, "human_node")
builder.add_conditional_edges("human_node", route_next)
builder.add_edge("agent_node", "human_node")

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

@cl.on_chat_start
async def on_chat_start():
    # A checkpointer is STRICTLY REQUIRED for interrupts to function
    config = {"configurable": {"thread_id": cl.context.session.id}}
    cl.user_session.set("config", config)
    cl.user_session.set("graph", graph)
    
    await cl.Message(
        content="Starting chat loop... type 'quit' to exit. You can attach images directly to your messages!"
    ).send()
    
    # Immediately hit the humannode and pause at the interrupt
    await graph.ainvoke({"lastinput": "", "image_path": None, "image_mime": None}, config=config)

@cl.on_message
async def on_message(message: cl.Message):
    config = cl.user_session.get("config")
    graph = cl.user_session.get("graph")
    
    # 1. Extract the first image from the message if one exists
    # Chainlit attaches files as elements; we filter for images based on MIME type
    images = [el for el in (message.elements or []) if "image" in getattr(el, "mime", "")]
    image_path = images[0].path if images else None
    image_mime = images[0].mime if images else None
    print(image_path)
    
    # 2. Package the text and image metadata into a payload dict
    resume_payload = {
        "text": message.content,
        "image_path": image_path,
        "image_mime": image_mime
    }
    
    # 3. Resume the paused graph, passing the payload dictionary into the interrupt
    await graph.ainvoke(Command(resume=resume_payload), config=config)
    
    # Check the current state of the graph
    state = graph.get_state(config)
    if not state.next:
        await cl.Message(content="Chat session ended gracefully. Please refresh to start again.").send()
        return
        
    current_state = state.values
    
    # 4. Construct the agent's response to prove it read the state correctly
    reply = f"Agent: I received your message: '{current_state.get('lastinput')}'"
    if current_state.get('image_path'):
        reply += f"\nI also successfully received an image of type {current_state.get('image_mime')}!"
        
    await cl.Message(content=reply).send()

