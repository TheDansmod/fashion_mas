# this is taken from gemini
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables.graph import MermaidDrawMethod

# 1. Define the state for the graph
class State(TypedDict):
    last_input: str

# 2. Define the node where human interaction happens
def human_node(state: State) -> State:
    # The interrupt pauses execution.
    # When we resume the graph later, `user_input` gets the value passed in Command(resume=...)
    user_input = interrupt("Waiting for user input...")
    
    # We update the state with whatever the user typed
    return {"last_input": user_input}

# 3. Define the backend/agent processing node
def agent_node(state: State) -> State:
    # This acts as the backend process (e.g., calling an LLM, searching a database)
    print(f"\n[Agent]: I received your message: '{state['last_input']}'")
    
    # We return the state (or updates to the state)
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
# Route from human directly to END if they quit, or to agent if they continue
builder.add_conditional_edges("human_node", route_next)
# Loop back from agent to human to wait for the next message
builder.add_edge("agent_node", "human_node")

# A checkpointer is STRICTLY REQUIRED for interrupts to function
memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)
png_bytes = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
with open(r'/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas/src/exploration/interrupt_exploration_diagram.png', "wb") as f:
    f.write(png_bytes)

# 6. Run the execution loop
if __name__ == "__main__":
    # Define a unique session ID for this user's chat thread
    config = {"configurable": {"thread_id": "demo_chat_session"}}
    
    print("Starting chat loop... (type 'quit' to exit)")
    
    # Initial invocation to start the graph. 
    # It will immediately hit the `human_node` and pause at the `interrupt`.
    graph.invoke({"last_input": ""}, config=config)
    
    while True:
        # Check the current state of the graph
        state = graph.get_state(config)
        
        # If there are no 'next' nodes queued, the graph has successfully reached END
        if not state.next:
            print("\n[System]: Chat session ended gracefully.")
            break
            
        # Get console input from the human (simulating frontend interaction)
        user_message = input("\nYou: ")
        
        # Resume the paused graph, passing the human's message into the interrupt
        graph.invoke(Command(resume=user_message), config=config)

