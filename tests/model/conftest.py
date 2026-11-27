"""Fixtures and the report for the model-quality tests; see tests/model/harness.py."""

import json
from datetime import UTC, datetime

import pytest

from tests.model.harness import (
    REPORT,
    REPORT_DIR,
    CachedImageStore,
    ModelHarness,
    load_model_env,
)


@pytest.fixture(scope="session")
def model_env():
    """Settings, metadata and the embedder, loaded once per session."""
    return load_model_env()


@pytest.fixture
def harness(model_env, tmp_path):
    """A ModelHarness over the session's settings, metadata and embedder."""
    return ModelHarness(model_env, tmp_path)


def pytest_terminal_summary(terminalreporter):
    """Print a per-case summary and write the full report."""
    if not REPORT:
        return
    out = REPORT_DIR / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(REPORT, indent=2))
    tr = terminalreporter
    tr.section("model quality")
    for row in REPORT:
        scores = ", ".join(f"{t['score']:.2f}" for t in row["turns"])
        secs = sum(t["seconds"] for t in row["turns"])
        mark = "PASS" if row["passed"] else "FAIL"
        tr.write_line(
            f"{mark}  {row['case']} (run {row['run']}): {scores}; {secs:.0f}s"
        )
        for t in row["turns"]:
            failed = [name for name, ok in t["checks"].items() if not ok]
            if failed:
                tr.write_line(f"      failed: {'; '.join(failed)}")
    tr.write_line(f"uncached S3 image GETs: {CachedImageStore.downloads}")
    tr.write_line(f"report: {out / 'report.json'}")
