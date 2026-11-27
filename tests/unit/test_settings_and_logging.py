"""Settings are pure and idempotent; logging setup is idempotent."""

import json
import os
import subprocess
import sys

import pytest
from loguru import logger
from pydantic import ValidationError

from frag import logging_setup
from frag.settings import LogSettings


def test_settings_defaults_with_empty_environment(clean_env):
    settings = clean_env()
    assert settings.checkpointer.backend == "sqlite"
    assert settings.tools.mode == "local"
    assert settings.llm.provider == "ollama"
    assert settings.llm.model_id == "qwen3.5:4b"
    assert settings.llm.reasoning is False
    assert settings.collection == "fashion_gen"
    assert settings.log_level == "DEBUG"


def test_llm_defaults_follow_the_provider(clean_env, monkeypatch):
    monkeypatch.setenv("LLM__PROVIDER", "bedrock")
    llm = clean_env().llm
    assert llm.model_id.startswith("us.anthropic.claude-haiku")
    assert llm.rate_limit_rps == 0.15
    # an explicit value still wins over the provider default
    monkeypatch.setenv("LLM__MODEL_ID", "other-model")
    assert clean_env().llm.model_id == "other-model"


def test_the_judge_llm_is_set_separately(clean_env, monkeypatch):
    assert clean_env().evaluation.judge_llm is None  # same as the agent's LLM
    monkeypatch.setenv("EVALUATION__JUDGE_LLM__PROVIDER", "bedrock")
    settings = clean_env()
    assert settings.llm.provider == "ollama"
    judge = settings.evaluation.judge_llm
    assert (judge.provider, judge.rate_limit_rps) == ("bedrock", 0.15)
    assert judge.model_id.startswith("us.anthropic.claude-haiku")


def test_settings_are_idempotent(clean_env):
    assert clean_env() == clean_env()


def test_nested_env_vars_override(clean_env, monkeypatch):
    monkeypatch.setenv("CHECKPOINTER__BACKEND", "memory")
    monkeypatch.setenv("TOOLS__MODE", "mcp")
    monkeypatch.setenv("APP_ENV", "production")
    settings = clean_env()
    assert settings.checkpointer.backend == "memory"
    assert settings.tools.mode == "mcp"
    assert settings.log_level == "INFO"


@pytest.mark.parametrize(
    ("key", "value"),
    [("TOOLS__MODE", "false"), ("CHECKPOINTER__BACKND", "sqlite")],
)
def test_invalid_or_mistyped_nested_values_fail_loudly(
    clean_env, monkeypatch, key, value
):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValidationError):
        clean_env()


def test_modules_import_without_environment_or_side_effects(tmp_path):
    """Every module imports in an empty environment with no .env (cleanup.md §0.5)."""
    code = (
        "import frag.settings, frag.runtime, frag.llm, frag.checkpointer, frag.tokens, "
        "frag.prompts, frag.messages, frag.logging_setup, frag.agent.graph, "
        "frag.ui.node_updates, "
        "frag.evaluation.run, frag.cli, "
        "frag.mcp_server.server, frag.catalogue.service, frag.catalogue.embedders, "
        "frag.tools.local, frag.tools.mcp_client"
    )
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)}
    result = subprocess.run(  # noqa: S603  fixed interpreter and arguments
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []  # nothing was written on import


def test_setup_logging_is_idempotent(tmp_path):
    settings = LogSettings(console=False, file_dir=str(tmp_path))
    try:
        logging_setup.setup_logging(
            settings, level="DEBUG", is_dev=True, service="test"
        )
        count_after_first = len(logger._core.handlers)  # no public API for this
        logging_setup.setup_logging(
            settings, level="DEBUG", is_dev=True, service="test"
        )
        assert len(logger._core.handlers) == count_after_first
        assert len(logging_setup._handler_ids) == 1  # the file sink
    finally:
        for handler_id in logging_setup._handler_ids:
            logger.remove(handler_id)
        logging_setup._handler_ids.clear()


@pytest.fixture
def restore_logging():
    yield
    for handler_id in logging_setup._handler_ids:
        logger.remove(handler_id)
    logging_setup._handler_ids.clear()


@pytest.mark.usefixtures("restore_logging")
def test_production_logs_json_to_stdout_only(tmp_path, capsys):
    settings = LogSettings(file_dir=str(tmp_path))
    logging_setup.setup_logging(settings, level="INFO", is_dev=False, service="agent")
    logger.info("hello {}", "json")
    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["record"]
    assert record["message"] == "hello json"
    assert list(tmp_path.iterdir()) == []  # no log file outside development


@pytest.mark.usefixtures("restore_logging")
def test_development_logs_to_a_file_per_service(tmp_path):
    settings = LogSettings(console=False, file_dir=str(tmp_path))
    logging_setup.setup_logging(settings, level="DEBUG", is_dev=True, service="mcp")
    logger.debug("to the file")
    logger.complete()
    assert "to the file" in (tmp_path / "frag-mcp.log").read_text()
