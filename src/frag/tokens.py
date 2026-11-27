"""Per-session LLM token-usage tracking."""

import csv
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import UsageMetadataCallbackHandler
from loguru import logger as log

CSV_HEADER = (
    "timestamp",
    "model",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "service",
    "thread_id",
)


def append_usage_csv(
    csv_path: str,
    usage_metadata: dict[str, Any],
    service: str = "",
    thread_id: str | None = None,
) -> None:
    """Append one row per model.

    Columns: time, model, input, output and total tokens, service, thread id. The two
    newer columns come last so rows written before they existed still line up.

    Short appends from separate processes are effectively atomic on local filesystems,
    so no locking is needed (cleanup.md §3.3).
    """
    if not usage_metadata:
        return
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if is_new:
            writer.writerow(CSV_HEADER)
        for model_name, metadata in usage_metadata.items():
            writer.writerow(
                [
                    datetime.now(UTC).isoformat(timespec="seconds"),
                    model_name,
                    metadata["input_tokens"],
                    metadata["output_tokens"],
                    metadata["total_tokens"],
                    service,
                    thread_id or "",
                ]
            )


@contextmanager
def track_tokens(
    csv_path: str | None, service: str, thread_id: str | None = None
) -> Iterator[UsageMetadataCallbackHandler]:
    """Yield a usage callback handler; log and record its totals on exit.

    Pass the handler to LLM calls via `callbacks=[handler]`. The totals are recorded
    even if the body raises, and the exception propagates.
    """
    handler = UsageMetadataCallbackHandler()
    try:
        yield handler
    finally:
        log.info("token usage ({}): {}", service, handler.usage_metadata)
        if csv_path:
            append_usage_csv(csv_path, handler.usage_metadata, service, thread_id)
