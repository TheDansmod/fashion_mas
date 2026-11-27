"""Loguru logging for one process (cleanup.md §3.2).

- Development (`APP_ENV=development`): a Rich console sink, plus an optional plain-text
  file `{file_dir}/frag-{service}.log` (rotated). Each process has its own `service`,
  so two processes never write to the same rotating file (§3.3).
- Otherwise (staging, production): JSON lines on stdout only, for the container's log
  driver to collect.

Standard-library logging (used by most libraries) is routed into loguru; the root
logger stays at WARNING, and `LogSettings.library_levels` sets individual libraries.
"""

import contextlib
import logging
import site
import sys

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import Traceback

from frag.settings import LogSettings


class InterceptHandler(logging.Handler):
    """Forward standard-library log records to loguru, keeping the caller's location."""

    def emit(self, record: logging.LogRecord) -> None:
        """Log the record through loguru at the same level."""
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        # skip the logging module's own frames so loguru reports the real caller
        frame, depth = logging.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _rich_console_sink(*, show_locals: bool) -> object:
    """A loguru sink that prints through Rich: columns, colours and tracebacks."""
    console = Console()
    handler = RichHandler(
        console=console, rich_tracebacks=True, show_time=True, show_path=True
    )

    def sink(message: object) -> None:
        record = message.record
        handler.emit(
            logging.LogRecord(
                name=record["name"],
                level=logging.getLevelName(record["level"].name),
                pathname=record["file"].path,
                lineno=record["line"],
                msg=record["message"],
                args=(),
                exc_info=None,
            )
        )
        if (exc := record["exception"]) is not None:
            console.print(
                Traceback.from_exception(
                    exc.type,
                    exc.value,
                    exc.traceback,
                    show_locals=show_locals,
                    width=console.width,
                    # hide frames inside installed packages
                    suppress=site.getsitepackages(),
                )
            )

    return sink


# ids of the sinks setup_logging added, so a second call replaces only those and never
# removes sinks added by others (e.g. pytest's caplog)
_handler_ids: list[int] = []


def setup_logging(
    settings: LogSettings, *, level: str, is_dev: bool, service: str
) -> None:
    """Configure logging for this process; calling it again replaces its own sinks.

    Args:
        settings: Logging settings.
        level: Effective log level (see `Settings.log_level`).
        is_dev: Development (Rich console and file) or not (JSON on stdout).
        service: Process name ("agent", "mcp", "eval", "cli", ...), used in the log
            file name.
    """
    for handler_id in _handler_ids:
        with contextlib.suppress(ValueError):
            logger.remove(handler_id)
    _handler_ids.clear()
    # loguru's default stderr sink has id 0
    with contextlib.suppress(ValueError):
        logger.remove(0)

    if not is_dev:
        _handler_ids.append(logger.add(sys.stdout, level=level, serialize=True))
    else:
        if settings.console:
            _handler_ids.append(
                logger.add(
                    _rich_console_sink(show_locals=True),
                    level=level,
                    format="{message}",  # Rich adds the time, level and location
                    colorize=False,
                    backtrace=False,  # Rich renders the traceback
                    diagnose=False,
                )
            )
        if settings.file:
            _handler_ids.append(
                logger.add(
                    f"{settings.file_dir}/frag-{service}.log",
                    level=level,
                    rotation=settings.rotation,
                    retention=settings.retention,
                    compression=settings.compression,
                    # no enqueue: loguru sinks are already thread-safe, the file is this
                    # process's own, and enqueue pickles records, which fails for
                    # exceptions such as botocore's dynamically created error classes
                    backtrace=True,
                    diagnose=True,
                )
            )

    logging.basicConfig(
        handlers=[InterceptHandler()], level=logging.WARNING, force=True
    )
    intercept = InterceptHandler()
    for name, library_level in settings.library_levels.items():
        library_logger = logging.getLogger(name)
        library_logger.handlers = [intercept]
        library_logger.setLevel(library_level)
        library_logger.propagate = False
