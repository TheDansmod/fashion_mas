"""The `frag` command: `frag chat`, `frag eval` and `frag mcp`.

Configuration comes from the environment and `.env` (see `.env.example`); the
arguments here only choose what to run.
"""

import argparse
import asyncio
import contextlib
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from loguru import logger as log

from frag.agent.graph import FashionAgent
from frag.evaluation.llm_as_judge import run_full_evaluation_pipeline
from frag.mcp_server.server import serve
from frag.runtime import Resources, Role, open_resources
from frag.settings import get_settings
from frag.tokens import track_tokens
from frag.utils.logger_setup import setup_logging

QUIT = "quit"


def _ask(prompt: str) -> str:
    return input(prompt).strip()


def _existing_file(path: str) -> str | None:
    resolved = Path(path).expanduser()
    return str(resolved) if resolved.is_file() else None


async def _read_message() -> tuple[str, list[str]] | None:
    """Read one message (text, then image paths) from the terminal; None to quit."""
    text = await asyncio.to_thread(_ask, f"\nYour request (`{QUIT}` to exit): ")
    if text.lower() == QUIT:
        return None
    images = []
    while True:
        path = await asyncio.to_thread(
            _ask, "Path to an image (leave empty when done): "
        )
        if not path:
            return text, images
        # checked here so a typo is caught before the agent runs
        if (image := _existing_file(path)) is not None:
            images.append(image)
        else:
            print(f"No such file: {path}")  # noqa: T201  terminal UI


async def chat(resources: Resources) -> None:
    """Terminal chat; one conversation thread per run, so later messages follow up."""
    agent = FashionAgent(
        llm=resources.llm,
        tools=resources.tools,
        checkpointer=resources.checkpointer,
        settings=resources.settings.agent,
    )
    thread_id = str(uuid4())
    csv_path = resources.settings.token_usage_csv
    while (message := await _read_message()) is not None:
        text, images = message
        with track_tokens(csv_path, "cli", thread_id) as usage:
            result = await agent.run_turn(text, images, thread_id, callbacks=[usage])
        paths = result["recommended_clothes_image_paths"]
        if not paths:
            print("\nNo recommendations could be found for your request.")  # noqa: T201
            continue
        print("\nRecommended images:")  # noqa: T201
        for path in paths:
            print(f"  {path}")  # noqa: T201
        print(f"\n{result['recommended_clothes_explanation']}")  # noqa: T201


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="frag", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("chat", help="chat with the agent in the terminal")
    evaluate = commands.add_parser(
        "eval", help="run the agent over the evaluation queries and judge the outputs"
    )
    evaluate.add_argument(
        "--skip-run", action="store_true", help="only judge the stored outputs"
    )
    evaluate.add_argument(
        "--skip-judge", action="store_true", help="only run the agent"
    )
    evaluate.add_argument(
        "--limit", type=int, metavar="N", help="run only the first N queries"
    )
    commands.add_parser("mcp", help="serve the catalogue tools over MCP (HTTP)")
    return parser


_ROLES: dict[str, Role] = {"chat": "agent", "eval": "eval", "mcp": "mcp"}
_SERVICES = {"chat": "cli", "eval": "eval", "mcp": "mcp"}


async def _run(args: argparse.Namespace) -> None:
    settings = get_settings()
    setup_logging(
        settings.logging,
        level=settings.log_level,
        is_dev=settings.is_dev,
        service=_SERVICES[args.command],
    )
    try:
        async with open_resources(settings, _ROLES[args.command]) as resources:
            match args.command:
                case "chat":
                    await chat(resources)
                case "eval":
                    await run_full_evaluation_pipeline(
                        resources,
                        run=not args.skip_run,
                        judge=not args.skip_judge,
                        limit=args.limit,
                    )
                case "mcp":
                    await serve(resources.catalogue, settings.tools)
    except Exception:  # noqa: BLE001  top level: log any failure, exit non-zero
        log.exception("frag {} failed.", args.command)
        raise SystemExit(1) from None
    finally:
        await log.complete()


def main(argv: list[str] | None = None) -> None:
    """Entry point of the `frag` command."""
    args = _parser().parse_args(argv)
    # makes .env visible to libraries that read os.environ themselves (LangSmith,
    # boto3, the Ollama client); never overrides variables that are already set
    load_dotenv()
    # Ctrl+C ends `frag chat` and `frag mcp`; no traceback for that
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_run(args))


if __name__ == "__main__":
    main()
