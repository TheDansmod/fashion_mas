"""Loguru logging: Rich console, rotating text and JSON files, stdlib intercept."""

import contextlib
import logging
import site

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import Traceback

from frag.settings import LogSettings


# funnel all logging module output for other libraries through loguru
class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # find caller from where originated the logged message
        frame, depth = logging.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def add_console_logging(
    is_dev: bool,
    log_level: str,
) -> int:
    # the rich handler owns console formatting
    console = Console(stderr=False)
    rich_handler = RichHandler(
        console=console,                # stdout, to match your existing setup
        rich_tracebacks=True,           # beautiful tracebacks in dev mode
        tracebacks_show_locals=is_dev,  # show local vars in tracebacks (like diagnose=True)
        show_time=True,
        show_level=True,
        show_path=True,
        markup=True,
    )

    # below function is from claude sonnet
    def rich_sink(message):
        # message.record contains all loguru metadata
        record = message.record
        level = record["level"].name
        loguru_exc = record["exception"]

        # emit through rich's handler by constructing a minimal LogRecord
        log_record = logging.LogRecord(
            name=record["name"],
            level=logging.getLevelName(level),
            pathname=record["file"].path,
            lineno=record["line"],
            msg=record["message"],
            args=(),
            exc_info=None,
        )
        rich_handler.emit(log_record)

        # Render the traceback separately, directly via the Rich console
        if loguru_exc is not None:
            tb = Traceback.from_exception(
                loguru_exc.type,
                loguru_exc.value,
                loguru_exc.traceback,
                show_locals=rich_handler.tracebacks_show_locals,
                width=console.width,
                suppress=site.getsitepackages()  # suppress all install third-party packages
            )
            console.print(tb)

    handler_id = logger.add(
        rich_sink,
        level=log_level,
        diagnose=False,  # rich handles tracebacks
        backtrace=False, # rich handles this
        enqueue=False,   # not needed for console logging
        format="{message}",  # should be plain since rich adds its own columns
        colorize=False,  # rich adds its own styling
    )
    return handler_id


# add local human readable rotating file logging
def add_file_logs(
    log_file: str,
    log_level: str,
    is_dev: bool,
    rotation_interval: str,
    retention_period: str,
    compression_method: str,
) -> int:
    handler_id = logger.add(
        sink=log_file,
        level=log_level,
        colorize=False,
        serialize=False,
        backtrace=is_dev,
        diagnose=is_dev,
        enqueue=True,
        rotation=rotation_interval,
        retention=retention_period,
        compression=compression_method,
    )
    return handler_id


# add json-formatted rotating machine readable logs
def add_machine_logs(
    log_file: str,
    log_level: str,
    rotation_interval: str,
    retention_period: str,
    compression_method: str,
) -> int:
    handler_id = logger.add(
        sink=log_file,
        level=log_level,
        colorize=False,
        serialize=True,
        backtrace=False,
        diagnose=False,
        enqueue=True,
        rotation=rotation_interval,
        retention=retention_period,
        compression=compression_method,
    )
    return handler_id


def set_library_log_levels(library_levels: dict[str, str]) -> None:
    """Route the named third-party loggers through loguru at the given levels."""
    intercept_handler = InterceptHandler()
    for logger_name, log_level in library_levels.items():
        lib_logger = logging.getLogger(logger_name)
        lib_logger.handlers = [intercept_handler]
        lib_logger.setLevel(log_level)
        lib_logger.propagate = False


# handler ids added by setup_logging, so that calling it again replaces only its own
# handlers (idempotent) and never removes sinks added by others, e.g. pytest's caplog
_handler_ids: list[int] = []


def setup_logging(
    settings: LogSettings, *, level: str, is_dev: bool, service: str
) -> None:
    """Configure loguru sinks for this process. Safe to call more than once.

    Args:
        settings: Logging settings.
        level: Effective log level (see `Settings.log_level`).
        is_dev: Enables locals in tracebacks.
        service: Process name ("agent", "mcp", "eval", "cli", ...). Log files are named
            per service so that separate processes never write to the same rotating
            file (cleanup.md §3.3).
    """
    for handler_id in _handler_ids:
        with contextlib.suppress(ValueError):
            logger.remove(handler_id)
    _handler_ids.clear()
    # loguru's default stderr sink has id 0
    with contextlib.suppress(ValueError):
        logger.remove(0)

    if settings.console:
        _handler_ids.append(add_console_logging(is_dev, level))
    if settings.human_readable_file:
        _handler_ids.append(
            add_file_logs(
                f"{settings.file_dir}/human_readable/frag-{service}.log",
                level,
                is_dev,
                settings.rotation,
                settings.retention,
                settings.compression,
            )
        )
    if settings.machine_readable_file:
        _handler_ids.append(
            add_machine_logs(
                f"{settings.file_dir}/machine_readable/frag-{service}.log",
                level,
                settings.rotation,
                settings.retention,
                settings.compression,
            )
        )

    # redirect the standard logging module into loguru; the root logger stays at
    # WARNING and the libraries listed in settings get their own levels
    logging.basicConfig(
        handlers=[InterceptHandler()], level=logging.WARNING, force=True
    )
    set_library_log_levels(settings.library_levels)
