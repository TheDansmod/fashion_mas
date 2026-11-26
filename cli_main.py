"""This is the starting point for the project."""

import asyncio
from loguru import logger as log

from dependency_injector.wiring import inject, Provide as PV

# dependency wiring must be done before frag imports
from frag.config.container import Container

container = Container()

from frag.rag_pipeline.rag_agent import run_fashion_agent
from frag.evaluation.llm_as_judge import run_full_evaluation_pipeline

cfg = Container.config.provided


@inject
async def main(
    eval_mode: bool = PV[cfg.evaluation.eval_mode],
):
    """Launch the current main task for the project."""
    try:
        await container.init_resources()
        if eval_mode:
            log.info("Running full evaluation pipeline.")
            await run_full_evaluation_pipeline()
        else:
            log.info("Running the CLI fashion recommendation agent.")
            await run_fashion_agent()
    except Exception as e:
        log.exception("Some Exception in cli_main.py")
    finally:
        await container.shutdown_resources()


if __name__ == "__main__":
    asyncio.run(main())
