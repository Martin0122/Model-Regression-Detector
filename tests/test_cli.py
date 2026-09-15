"""CLI entrypoint smoke tests.

These run the modules as scripts, which unit tests never exercise. Regression guard: after the
old-judge runs were archived, `python -m src.comparer` dumped a raw traceback on what is an
ordinary state (no baselines yet) because __main__ didn't handle InsufficientRunHistory.
"""

import os
import subprocess
import sys

import pytest


def run_module(module, cwd, env=None):
    environment = {**os.environ, **(env or {})}
    environment.pop("OPENAI_API_KEY", None)
    return subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=environment,
    )


@pytest.fixture
def empty_runs_repo(tmp_path, monkeypatch):
    """A working copy whose run history is empty, mirroring a fresh clone."""
    repo = os.getcwd()
    (tmp_path / "src").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "datasets").symlink_to(os.path.join(repo, "datasets"))
    (tmp_path / "prompts").symlink_to(os.path.join(repo, "prompts"))
    for name in os.listdir(os.path.join(repo, "src")):
        if name != "runs":
            (tmp_path / "src" / name).symlink_to(os.path.join(repo, "src", name))
    (tmp_path / "src" / "runs").mkdir()
    return tmp_path


@pytest.mark.parametrize("module", ["src.comparer", "src.report"])
def test_cli_reports_missing_history_without_a_traceback(module, empty_runs_repo):
    result = run_module(module, cwd=str(empty_runs_repo))
    assert "Traceback" not in result.stderr, result.stderr
    assert "InsufficientRunHistory" not in result.stderr
    assert "valid runs" in result.stdout
    assert result.returncode == 1


def test_drift_cli_reports_insufficient_history_and_exits_zero(empty_runs_repo):
    """Drift genuinely has nothing to say yet - that is not an error."""
    result = run_module("src.drift", cwd=str(empty_runs_repo))
    assert "Traceback" not in result.stderr, result.stderr
    assert "Not enough run history" in result.stdout
    assert result.returncode == 0


def test_clis_do_not_require_an_api_key(empty_runs_repo):
    """Offline work must never touch credentials."""
    for module in ("src.comparer", "src.report", "src.drift"):
        result = run_module(module, cwd=str(empty_runs_repo))
        assert "OPENAI_API_KEY is required" not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "module", ["src.notifier", "src.report", "src.comparer", "src.drift", "src.pipeline"]
)
def test_every_entry_point_loads_dotenv(module, tmp_path):
    """Regression test: load_dotenv() used to run as a side effect of importing the OpenAI
    client, so .env was only picked up by entry points that happened to reach it. src.pipeline
    did; src.notifier, src.report, src.comparer and src.drift did not - silently ignoring a
    configured SLACK_WEBHOOK_URL and falling back to default thresholds."""
    env_file = tmp_path / ".env"
    env_file.write_text("SLACK_WEBHOOK_URL=https://example.invalid/from-dotenv\n")

    probe = (
        f"import os, importlib; importlib.import_module('{module}'); "
        "print(os.getenv('SLACK_WEBHOOK_URL'))"
    )
    environment = {**os.environ}
    environment.pop("SLACK_WEBHOOK_URL", None)
    environment.pop("OPENAI_API_KEY", None)
    environment["PYTHONPATH"] = os.getcwd()

    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=environment,
    )
    assert "https://example.invalid/from-dotenv" in result.stdout, (
        f"{module} did not load .env\nstdout={result.stdout}\nstderr={result.stderr}"
    )


def test_dataset_validator_cli_passes():
    result = subprocess.run(
        [sys.executable, "datasets/validate_golden_dataset.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Validated 50 cases" in result.stdout
