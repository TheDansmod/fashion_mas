import pytest
import pytest_asyncio
from dependency_injector import providers

from frag.config.container import Container


@pytest_asyncio.fixture
async def container(mocked_llm):
    c = Container()

    with c.override_providers(
        llm_model=providers.Object(mocked_llm),
    ):
        await c.init_resources()
        c.wire(modules=["tests.integration"])
        yield c
        await c.shutdown_resources()
        c.unwire()
        c.reset_singletons()


@pytest.fixture
def mock_input(monkeypatch):
    def _set_inputs(*inputs):
        answers = iter(inputs)
        monkeypatch.setattr("builtins.input", lambda _: next(answers))

    return _set_inputs
