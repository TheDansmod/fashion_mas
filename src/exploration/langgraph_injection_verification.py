from dependency_injector.wiring import inject, Provide as PV
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from loguru import logger as log

from src.config.container import Container

cfg = Container.config.provided

class AgentState(BaseModel):
    header: str = ""
    body: str = ""
    footer: str = ""

@inject
def add_header(
    state: AgentState,
    postgres_dsn: str = PV[cfg.orchestration.checkpointer.postgres.dsn],
) -> dict[str, str]:
    log.error("In add_header function")
    log.info(f"{postgres_dsn=}")
    return {"header": "to: my_dear_friend@gmail.com"}

def add_body(
    state: AgentState,
    data_config = PV[cfg.data],
) -> dict[str, str]:
    log.debug("In add_body function")
    log.info(f"All of data config:\n{data_config}")
    return {"body": "please send me the air fryer that I had requested at the earliest possible opportunity."}

def add_footer(state: AgentState) -> dict[str, str]:
    log.debug("In add_footer function")
    return {"footer": "Your Worst Enemy, Me!"}

def build_graph():
    builder = StateGraph(AgentState)
    log.trace('got builder')

    builder.add_node("add_header", add_header)
    builder.add_node("add_body", add_body)
    builder.add_node("add_footer", add_footer)
    log.success('added nodes')

    builder.add_edge(START, "add_header")
    builder.add_edge("add_header", "add_body")
    builder.add_edge("add_body", "add_footer")
    builder.add_edge("add_footer", END)
    log.warning('added edges')

    graph = builder.compile()
    return graph

def run_graph():
    graph = build_graph()
    result = graph.invoke(dict())
    log.info(result)
