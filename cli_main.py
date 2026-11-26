"""Terminal entrypoint: `uv run cli_main.py [chat|eval]`.

Replaced by the `frag` command (src/frag/cli.py) in cleanup.md Phase 8.
"""

import argparse
import asyncio

from dotenv import load_dotenv
from loguru import logger as log

from frag.evaluation.llm_as_judge import run_full_evaluation_pipeline
from frag.rag_pipeline.rag_agent import run_fashion_agent
from frag.runtime import open_resources
from frag.settings import get_settings
from frag.utils.logger_setup import setup_logging


async def main(mode: str) -> None:
    """Run the terminal chat loop or the full evaluation pipeline."""
    settings = get_settings()
    setup_logging(
        settings.logging,
        level=settings.log_level,
        is_dev=settings.is_dev,
        service="eval" if mode == "eval" else "cli",
    )
    try:
        role = "eval" if mode == "eval" else "agent"
        async with open_resources(settings, role) as resources:
            if mode == "eval":
                log.info("Running full evaluation pipeline.")
                await run_full_evaluation_pipeline(resources)
            else:
                log.info("Running the CLI fashion recommendation agent.")
                await run_fashion_agent(resources)
    except Exception:  # noqa: BLE001  top-level: log any failure before exiting
        log.exception("Unhandled exception in cli_main.py")
    finally:
        await log.complete()


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=["chat", "eval"], default="chat")
    asyncio.run(main(parser.parse_args().mode))
