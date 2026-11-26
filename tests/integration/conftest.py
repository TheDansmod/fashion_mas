import pytest

# The dependency-injector `container` fixture was removed in cleanup.md Phase 4; the
# integration test is rewritten around explicit construction in Phase 5.


@pytest.fixture
def mock_input(monkeypatch):
    def _set_inputs(*inputs):
        answers = iter(inputs)
        monkeypatch.setattr("builtins.input", lambda _: next(answers))

    return _set_inputs
