import sys
import traceback
import logging
import json

from loguru import logger


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
    handler_id = logger.add(
        sys.stdout,
        level=log_level,
        diagnose=is_dev,
        backtrace=is_dev,
        enqueue=True,
        colorize=True,
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


def setup_logging(log_cfg):
    # clear the default stderr handler
    logger.remove()

    if log_cfg.write_console_logs:
        add_console_logging(log_cfg.is_dev, log_cfg.log_level)
    if log_cfg.write_human_readable_logs:
        add_file_logs(
            log_cfg.human_readable_log_file,
            log_cfg.log_level,
            log_cfg.is_dev,
            log_cfg.rotation_interval,
            log_cfg.retention_period,
            log_cfg.compression_method,
        )
    if log_cfg.write_machine_readable_logs:
        add_machine_logs(
            log_cfg.machine_readable_log_file,
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
