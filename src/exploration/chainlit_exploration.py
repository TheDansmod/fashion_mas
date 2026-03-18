# this is taken from gemini
import chainlit as cl
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

# 1. Define the state for the graph
class State(TypedDict):
    last_input: str

# 2. Define the node where human interaction happens
def human_node(state: State):
    # The interrupt pauses execution.
    user_input = interrupt("Waiting for user input...")
    # We update the state with whatever the user typed
    return {"last_input": user_input}

# 3. Define the backend/agent processing node
def agent_node(state: State):
    # This acts as the backend process
    message = f"I received your message: '{state['last_input']}'"
    # We can send a message back to the UI here if we want, or handle it in the loop
    return state

# 4. Define conditional routing
def route_next(state: State) -> Literal[END, "agent_node"]:
    # If the user says quit, we exit the loop
    if state.get("last_input", "").strip().lower() == "quit":
        return END
    # Otherwise, pass the input to the agent node
    return "agent_node"

# 5. Build and compile the graph
builder = StateGraph(State)

builder.add_node("human_node", human_node)
builder.add_node("agent_node", agent_node)

builder.add_edge(START, "human_node")
builder.add_conditional_edges("human_node", route_next)
builder.add_edge("agent_node", "human_node")

# A checkpointer is STRICTLY REQUIRED for interrupts to function
memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

@cl.on_chat_start
async def on_chat_start():
    # Define a unique session ID for this user's chat thread
    config = {"configurable": {"thread_id": cl.context.session.id}}
    cl.user_session.set("config", config)
    cl.user_session.set("graph", graph)

    await cl.Message(content="Starting chat loop... (type 'quit' to exit)").send()

    # Initial invocation to start the graph.
    # It will immediately hit the `human_node` and pause at the `interrupt`.
    result = await graph.ainvoke({"last_input": ""}, config=config)

    # Check if we were interrupted
    if len(result.get('__interrupt__', [])) > 0:
        # Get the interrupt message
        interrupt_msg = result['__interrupt__'][0].value
        # No need to send it if we don't want to double message, or we can:
        # await cl.Message(content=interrupt_msg).send()
        pass

@cl.on_message
async def on_message(message: cl.Message):
    config = cl.user_session.get("config")
    graph = cl.user_session.get("graph")

    user_message = message.content

    # Resume the paused graph, passing the human's message into the interrupt
    result = await graph.ainvoke(Command(resume=user_message), config=config)

    # Check the current state of the graph
    state = graph.get_state(config)

    # If there are no 'next' nodes queued, the graph has successfully reached END
    if not state.next:
        await cl.Message(content="Chat session ended gracefully. Please refresh to start again.").send()
        return

    # The agent node ran, let's get the state and reply to the user
    # In a real app, you might want the agent node to yield messages,
    # but here we'll just read from the state since it was updated
    current_state = state.values
    await cl.Message(content=f"[Agent]: I received your message: '{current_state.get('last_input', '')}'").send()

