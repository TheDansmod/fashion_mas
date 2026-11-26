"""Per-session LLM token-usage tracking."""

import csv
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import UsageMetadataCallbackHandler
from loguru import logger as log


def append_usage_csv(csv_path: str, usage_metadata: dict[str, Any]) -> None:
    """Append one row per model (time, model, input, output, total tokens).

    Short appends from separate processes are effectively atomic on local filesystems,
    so no locking is needed (cleanup.md §3.3).
    """
    if not usage_metadata:
        return
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        for model_name, metadata in usage_metadata.items():
            writer.writerow(
                [
                    datetime.now(UTC).isoformat(timespec="seconds"),
                    model_name,
                    metadata["input_tokens"],
                    metadata["output_tokens"],
                    metadata["total_tokens"],
                ]
            )


@contextmanager
def track_tokens(csv_path: str | None) -> Iterator[UsageMetadataCallbackHandler]:
    """Yield a usage callback handler; log and record its totals on exit.

    Pass the handler to LLM calls via `callbacks=[handler]`. Exceptions propagate.
    """
    handler = UsageMetadataCallbackHandler()
    try:
        yield handler
    finally:
        log.info("token usage: {}", handler.usage_metadata)
        if csv_path:
            append_usage_csv(csv_path, handler.usage_metadata)
