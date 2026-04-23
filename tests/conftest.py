import pytest
from loguru import logger
from _pytest.logging import LogCaptureFixture

from tests.mocks.mock_llm_agent import ChatMockLLM, mock_create_agent


@pytest.fixture
def mocked_llm():
    # used to setup the container
    return ChatMockLLM()


@pytest.fixture
def mocked_create_agent(mocker):
    # we will also be doing the substitution into where this is used to make things easier
    # we use side_effect instead of new since that allows verifying the number of invocations (if required)
    return mocker.patch(
        "frag.rag_pipeline.rag_agent.create_agent", side_effect=mock_create_agent
    )


# since we are using loguru and that does not hook into python's canonical logging system, we have to override the caplog built-in fixture to become a sink for loguru
@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    handler_id = logger.add(
        caplog.handler,
        # this ensures that the full logs don't get passed through (which might include stuff like the timestamp and invoking module, etc), only the messages themselves
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        # level of 0 means that the sink does not have its own log level and accepts all log messages
        level=0,
        # you can set a level for caplog as well - the below line respects the caplog level, and only allows those logs through which have either the same or higher log level.
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,  # set to True if your test spawns child processes
    )
    yield caplog
    try:
        logger.remove(handler_id)
    except ValueError:
        pass  # handler already removed

@pytest.fixture(autouse=True, scope="session")
def loguru_file_sink():
    handler_id = logger.add(
        "logs/pytest.log",
        level="DEBUG",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        mode="a",
        enqueue=False,
        colorize=False,
        backtrace=True,
        diagnose=True,
    )
    yield
    try:
        logger.remove(handler_id)
    except ValueError:
        pass  # handler already removed
