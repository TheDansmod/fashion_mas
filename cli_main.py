"""Terminal entrypoint: `uv run cli_main.py [chat|eval]`.

Replaced by the `frag` command (src/frag/cli.py) in cleanup.md Phase 8.
"""

import argparse
import asyncio
from uuid import uuid4

from dotenv import load_dotenv
from loguru import logger as log

from frag.agent.graph import FashionAgent
from frag.evaluation.llm_as_judge import run_full_evaluation_pipeline
from frag.runtime import Resources, open_resources
from frag.settings import get_settings
from frag.tokens import track_tokens
from frag.utils.logger_setup import setup_logging


async def chat_loop(resources: Resources) -> None:
    """Terminal chat: each run starts a new conversation thread."""
    agent = FashionAgent(
        llm=resources.llm,
        tools=resources.tools,
        checkpointer=resources.checkpointer,
        settings=resources.settings.agent,
    )
    thread_id = str(uuid4())
    csv_path = resources.settings.token_usage_csv
    while True:
        text = await asyncio.to_thread(input, "Enter `quit` to exit. Your request: ")
        if text.strip().lower() == "quit":
            break
        num_images = int(await asyncio.to_thread(input, "Number of images: ") or 0)
        image_paths = [
            await asyncio.to_thread(input, f"Path to image {n + 1}: ")
            for n in range(num_images)
        ]
        with track_tokens(csv_path, "cli", thread_id) as usage:
            result = await agent.run_turn(
                text, image_paths, thread_id, callbacks=[usage]
            )
        for path in result["recommended_clothes_image_paths"]:
            log.info("Recommended image: {}", path)
        log.info("Explanation: {}", result["recommended_clothes_explanation"])


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
                await chat_loop(resources)
    except Exception:  # noqa: BLE001  top-level: log any failure before exiting
        log.exception("Unhandled exception in cli_main.py")
    finally:
        await log.complete()


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=["chat", "eval"], default="chat")
    asyncio.run(main(parser.parse_args().mode))
