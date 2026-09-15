"""Whether critical drift blocks a merge is opt-in - cover both settings."""

import pytest

import src.pipeline as pipeline_mod
from src.models import DriftResult


@pytest.fixture
def critical_drift():
    return DriftResult(
        window_size=7,
        baseline_run_id="r1",
        baseline_moving_average=1.0,
        current_run_id="r8",
        current_moving_average=0.85,
        drift_delta=-0.15,
        status="critical",
    )


@pytest.fixture
def pipeline_with(tmp_path, monkeypatch, make_run, make_scored_result, make_comparison):
    monkeypatch.chdir(tmp_path)

    def _setup(drift, comparison_status="pass"):
        async def fake_run_eval(*args, **kwargs):
            return make_run("run_2026-01-01T00-00-00", [make_scored_result("tc_1", True)])

        monkeypatch.setattr(pipeline_mod, "run_eval", fake_run_eval)
        monkeypatch.setattr(
            pipeline_mod.report_module,
            "main",
            lambda: (make_comparison(comparison_status), tmp_path / "r.html", drift),
        )

    return _setup


async def test_critical_drift_does_not_block_by_default(pipeline_with, critical_drift, monkeypatch):
    monkeypatch.delenv("BLOCK_ON_CRITICAL_DRIFT", raising=False)
    pipeline_with(critical_drift)
    assert await pipeline_mod.main() == 0


async def test_critical_drift_blocks_when_opted_in(pipeline_with, critical_drift, monkeypatch):
    monkeypatch.setenv("BLOCK_ON_CRITICAL_DRIFT", "true")
    pipeline_with(critical_drift)
    assert await pipeline_mod.main() == 1


async def test_critical_regression_blocks_regardless_of_drift_setting(pipeline_with, monkeypatch):
    monkeypatch.setenv("BLOCK_ON_CRITICAL_DRIFT", "false")
    pipeline_with(None, comparison_status="critical")
    assert await pipeline_mod.main() == 1


async def test_passing_drift_does_not_block_when_gate_is_on(pipeline_with, monkeypatch):
    monkeypatch.setenv("BLOCK_ON_CRITICAL_DRIFT", "true")
    drift = DriftResult(
        window_size=7,
        baseline_run_id="r1",
        baseline_moving_average=1.0,
        current_run_id="r8",
        current_moving_average=1.0,
        drift_delta=0.0,
        status="pass",
    )
    pipeline_with(drift)
    assert await pipeline_mod.main() == 0
