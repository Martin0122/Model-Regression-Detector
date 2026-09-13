"""Pipeline exit codes, summary output, and run metadata."""
import pytest

import src.pipeline as pipeline_mod
from src.config import CaseFlip, DriftResult
from src.errors import IncompleteEvalRun, InsufficientRunHistory
from src.llm_judge import JUDGE_MODEL


@pytest.fixture
def in_repo(tmp_path, monkeypatch):
    """Run the pipeline with cwd inside a temp dir so reports/ is written there."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# --------------------------------------------------------------------------------------
# Exit codes
# --------------------------------------------------------------------------------------

async def test_exit_0_when_status_is_pass(in_repo, monkeypatch, make_run, make_scored_result, make_comparison):
    monkeypatch.setattr(pipeline_mod, "run_eval", _fake_run_eval(make_run, make_scored_result))
    monkeypatch.setattr(pipeline_mod.report_module, "main", lambda: (make_comparison("pass"), in_repo / "r.html", None))
    assert await pipeline_mod.main() == 0


async def test_exit_0_when_status_is_warning(in_repo, monkeypatch, make_run, make_scored_result, make_comparison):
    """A warning should be visible but must not block the merge."""
    monkeypatch.setattr(pipeline_mod, "run_eval", _fake_run_eval(make_run, make_scored_result))
    monkeypatch.setattr(pipeline_mod.report_module, "main", lambda: (make_comparison("warning"), in_repo / "r.html", None))
    assert await pipeline_mod.main() == 0


async def test_exit_1_when_status_is_critical(in_repo, monkeypatch, make_run, make_scored_result, make_comparison):
    monkeypatch.setattr(pipeline_mod, "run_eval", _fake_run_eval(make_run, make_scored_result))
    monkeypatch.setattr(pipeline_mod.report_module, "main", lambda: (make_comparison("critical"), in_repo / "r.html", None))
    assert await pipeline_mod.main() == 1


async def test_exit_1_and_no_run_saved_when_completion_gate_trips(in_repo, monkeypatch):
    """A partial run must not be persisted - it would skew future baselines and drift."""
    async def boom(*args, **kwargs):
        raise IncompleteEvalRun("Only 6/10 case(s) completed (60.0%), below the minimum completion rate of 95.0%.")

    monkeypatch.setattr(pipeline_mod, "run_eval", boom)
    assert await pipeline_mod.main() == 1
    assert not (in_repo / "src" / "runs").exists()
    assert "ABORTED" in (in_repo / "reports" / "pipeline_summary.md").read_text()


async def test_exit_0_on_first_run_with_no_baseline(in_repo, monkeypatch, make_run, make_scored_result):
    monkeypatch.setattr(pipeline_mod, "run_eval", _fake_run_eval(make_run, make_scored_result))

    def no_history():
        raise InsufficientRunHistory("Need at least 2 valid runs to build a report, found 1.")

    monkeypatch.setattr(pipeline_mod.report_module, "main", no_history)
    assert await pipeline_mod.main() == 0
    assert "FIRST RUN" in (in_repo / "reports" / "pipeline_summary.md").read_text()


async def test_unexpected_error_is_not_swallowed_as_first_run(in_repo, monkeypatch, make_run, make_scored_result):
    """Regression test: Pydantic's ValidationError IS-A ValueError. A broad `except ValueError`
    used to swallow a corrupt-run-file crash, report it as 'first run', and return exit 0 -
    silently skipping the regression gate."""
    monkeypatch.setattr(pipeline_mod, "run_eval", _fake_run_eval(make_run, make_scored_result))

    def corrupt():
        raise ValueError("simulated corrupt run file")

    monkeypatch.setattr(pipeline_mod.report_module, "main", corrupt)
    with pytest.raises(ValueError, match="simulated corrupt run file"):
        await pipeline_mod.main()


# --------------------------------------------------------------------------------------
# Summary output
# --------------------------------------------------------------------------------------

def test_summary_lists_regressed_cases(in_repo, make_comparison):
    comparison = make_comparison(
        status="critical",
        previous=0.94,
        current=0.89,
        regressions=[CaseFlip(test_case_id="tc_7", category="billing", previous_passed=True, current_passed=False)],
    )
    pipeline_mod.write_pipeline_summary(comparison, in_repo / "report.html", None)
    summary = (in_repo / "reports" / "pipeline_summary.md").read_text()
    assert "CRITICAL" in summary
    assert "tc_7" in summary
    assert "94.0%" in summary and "89.0%" in summary


def test_summary_includes_drift_warning(in_repo, make_comparison):
    drift = DriftResult(
        window_size=7, baseline_run_id="r1", baseline_moving_average=1.0,
        current_run_id="r8", current_moving_average=0.9, drift_delta=-0.1, status="critical",
    )
    pipeline_mod.write_pipeline_summary(make_comparison("pass"), in_repo / "report.html", drift)
    assert "Slow drift detected" in (in_repo / "reports" / "pipeline_summary.md").read_text()


def test_summary_written_to_github_step_summary(in_repo, monkeypatch, make_comparison):
    step_summary = in_repo / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step_summary))
    pipeline_mod.write_pipeline_summary(make_comparison("pass"), None, None)
    assert "Eval pipeline result" in step_summary.read_text()


# --------------------------------------------------------------------------------------
# Run metadata
# --------------------------------------------------------------------------------------

def test_run_metadata_records_the_actual_judge_model(make_run, make_scored_result):
    """Regression test: run metadata used to record the classifier model as the judge model."""
    import inspect
    source = inspect.getsource(pipeline_mod.run_eval)
    assert "judge_model=JUDGE_MODEL" in source
    assert JUDGE_MODEL


def _fake_run_eval(make_run, make_scored_result):
    async def _run(*args, **kwargs):
        return make_run("run_2026-01-01T00-00-00", [make_scored_result("tc_1", True)])
    return _run
