import sys
import traceback
import logging
import json
import site

from loguru import logger
from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import Traceback


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


# def _serialize(record):
#     payload = {
#         "time":     record["time"].isoformat(),
#         "level":    record["level"].name,
#         "message":  record["message"],
#         "module":   record["module"],
#         "line":     record["line"],
#     }
#     if record["extra"]:
#         payload.update(record["extra"])
#     if record["exception"]:
#         exc = record["exception"]
#         payload["exception"] = {
#             "type":      exc.type.__name__,
#             "value":     str(exc.value),
#             "traceback": traceback.format.exception(exc.type, exc.value, exc.traceback)
#         }
#     return json.dumps(payload)


def set_library_log_levels():
    intercept_handler = InterceptHandler()
    library_levels = {
        "httpx": "WARNING",
        "urllib3": "WARNING",
        "mcp": "WARNING",
        "httpcore": "WARNING",
        "botocore": "WARNING",
        "langgraph_checkpoint_aws": "WARNING",
    }
    for logger_name, log_level in library_levels.items():
        lib_logger = logging.getLogger(logger_name)
        lib_logger.handlers = [intercept_handler]
        lib_logger.setLevel(log_level)
        lib_logger.propagate = False


def setup_logging(log_cfg, for_mcp_server=False):
    # if we are doing log setup for mcp server we use different file names, since it is a different process
    # and we don't want multiple-processes simultaneously writing to the same file - it can cause corruption and race conditions

    # clear the default stderr handler
    # during testing logger.remove() without args can create issues by removing all handlers like my pytest log etc
    # so i am changing to remove only the stderr logger (id 0)
    try:
        logger.remove(0)
    except ValueError:
        # already removed - handler called multiple times - likely during testing
        pass

    if log_cfg.write_console_logs:
        add_console_logging(log_cfg.is_dev, log_cfg.log_level)
    if log_cfg.write_human_readable_logs:
        add_file_logs(
            log_cfg.mcp_human_readable_log_file if for_mcp_server else log_cfg.human_readable_log_file,
            log_cfg.log_level,
            log_cfg.is_dev,
            log_cfg.rotation_interval,
            log_cfg.retention_period,
            log_cfg.compression_method,
        )
    if log_cfg.write_machine_readable_logs:
        add_machine_logs(
            log_cfg.mcp_machine_readable_log_file if for_mcp_server else log_cfg.machine_readable_log_file,
            log_cfg.log_level,
            log_cfg.rotation_interval,
            log_cfg.retention_period,
            log_cfg.compression_method,
        )

    # Redirect everything from the standard logging module into loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # TODO: perhaps put the library levels in some pydantic config
    set_library_log_levels()

    # for now set all other levels to WARNING
    logging.getLogger().setLevel(logging.WARNING)
