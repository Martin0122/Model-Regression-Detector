"""Comparison logic: pass-rate deltas, per-category deltas, flipped cases, threshold status.

These preserve the edge cases found during the deep-testing pass, including the float-precision
boundary bug where an exact 8% drop computed as 0.07999999999999996 and was silently downgraded
from critical to warning.
"""

import pytest

from src.comparer import compare_runs, list_runs, load_runs_safe, pass_rate
from src.errors import IncompleteEvalRun, InsufficientRunHistory
from src.settings import ThresholdConfig


@pytest.fixture
def cat_map_100():
    return {f"tc_{i}": "billing" for i in range(100)}


def build_runs(make_run, make_scored_result, n_total, n_failing_in_current):
    baseline = make_run("r1", [make_scored_result(f"tc_{i}", True) for i in range(n_total)])
    current_results = [make_scored_result(f"tc_{i}", True) for i in range(n_total)]
    for i in range(n_failing_in_current):
        current_results[i] = make_scored_result(f"tc_{i}", False)
    return baseline, make_run("r2", current_results)


# --------------------------------------------------------------------------------------
# Threshold boundaries
# --------------------------------------------------------------------------------------


def test_exact_3_percent_drop_is_warning(make_run, make_scored_result, cat_map_100):
    baseline, current = build_runs(make_run, make_scored_result, 100, 3)
    result = compare_runs(baseline, current, cat_map_100, ThresholdConfig())
    assert result.status == "warning"
    assert len(result.regressions) == 3


def test_exact_8_percent_drop_is_critical(make_run, make_scored_result, cat_map_100):
    """Regression test: 1.0 - 0.92 == 0.07999999999999996 in IEEE 754, which used to fall
    below the 0.08 threshold and silently downgrade a critical regression to a warning."""
    baseline, current = build_runs(make_run, make_scored_result, 100, 8)
    result = compare_runs(baseline, current, cat_map_100, ThresholdConfig())
    assert result.status == "critical"


def test_small_drop_stays_pass(make_run, make_scored_result, cat_map_100):
    baseline, current = build_runs(make_run, make_scored_result, 100, 1)
    result = compare_runs(baseline, current, cat_map_100, ThresholdConfig())
    assert result.status == "pass"


def test_thresholds_are_configurable(make_run, make_scored_result, cat_map_100):
    baseline, current = build_runs(make_run, make_scored_result, 100, 2)
    strict = compare_runs(
        baseline,
        current,
        cat_map_100,
        ThresholdConfig(warning_threshold=0.01, critical_threshold=0.02),
    )
    assert strict.status == "critical"


# --------------------------------------------------------------------------------------
# Flip detection
# --------------------------------------------------------------------------------------


def test_pure_improvement_is_pass_with_no_regressions(make_run, make_scored_result):
    baseline = make_run("r1", [make_scored_result(f"tc_{i}", False) for i in range(10)])
    current = make_run("r2", [make_scored_result(f"tc_{i}", True) for i in range(10)])
    result = compare_runs(
        baseline, current, {f"tc_{i}": "billing" for i in range(10)}, ThresholdConfig()
    )
    assert result.status == "pass"
    assert len(result.improvements) == 10
    assert result.regressions == []


def test_case_added_in_current_is_not_counted_as_regression(make_run, make_scored_result):
    baseline = make_run("r1", [make_scored_result("tc_1", True)])
    current = make_run("r2", [make_scored_result("tc_1", True), make_scored_result("tc_2", False)])
    result = compare_runs(
        baseline, current, {"tc_1": "billing", "tc_2": "technical"}, ThresholdConfig()
    )
    assert result.regressions == []


def test_case_removed_from_current_does_not_crash(make_run, make_scored_result):
    baseline = make_run("r1", [make_scored_result("tc_1", True), make_scored_result("tc_2", True)])
    current = make_run("r2", [make_scored_result("tc_1", False)])
    result = compare_runs(
        baseline, current, {"tc_1": "billing", "tc_2": "technical"}, ThresholdConfig()
    )
    assert [f.test_case_id for f in result.regressions] == ["tc_1"]


def test_case_id_missing_from_category_map_is_tolerated(make_run, make_scored_result):
    baseline = make_run("r1", [make_scored_result("ghost", True)])
    current = make_run("r2", [make_scored_result("ghost", False)])
    result = compare_runs(baseline, current, {}, ThresholdConfig())
    assert result.regressions[0].category is None


# --------------------------------------------------------------------------------------
# Pass rate / per-category accuracy
# --------------------------------------------------------------------------------------


def test_pass_rate_of_empty_results_is_zero_not_zero_division():
    assert pass_rate([]) == 0.0


def test_empty_runs_compare_without_crashing(make_run):
    empty = make_run("r1", [])
    result = compare_runs(empty, empty, {}, ThresholdConfig())
    assert result.previous_pass_rate == 0.0
    assert result.current_pass_rate == 0.0


def test_per_category_deltas_are_computed_per_category(make_run, make_scored_result):
    baseline = make_run(
        "r1",
        [
            make_scored_result("b1", True, category_match=True),
            make_scored_result("t1", True, category_match=True),
        ],
    )
    current = make_run(
        "r2",
        [
            make_scored_result("b1", False, category_match=False),
            make_scored_result("t1", True, category_match=True),
        ],
    )
    result = compare_runs(
        baseline, current, {"b1": "billing", "t1": "technical"}, ThresholdConfig()
    )
    deltas = {d.category: d for d in result.category_deltas}
    assert deltas["billing"].delta == pytest.approx(-1.0)
    assert deltas["technical"].delta == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# Corrupt-file resilience
# --------------------------------------------------------------------------------------


def test_load_runs_safe_skips_corrupt_file_and_keeps_valid_ones(
    tmp_path, make_run, make_scored_result
):
    """Regression test: a single truncated run file used to raise a Pydantic ValidationError
    (which IS-A ValueError) and get misreported by the pipeline as 'first run', returning
    exit 0 and skipping the regression gate entirely."""
    good = make_run("run_2026-01-01T00-00-00", [make_scored_result("tc_1", True)])
    (tmp_path / "run_2026-01-01T00-00-00.json").write_text(good.model_dump_json())
    (tmp_path / "run_2026-01-02T00-00-00.json").write_text('{"run_metadata": {"run_i')
    good2 = make_run("run_2026-01-03T00-00-00", [make_scored_result("tc_1", False)])
    (tmp_path / "run_2026-01-03T00-00-00.json").write_text(good2.model_dump_json())

    runs = load_runs_safe(list_runs(str(tmp_path)))
    assert len(runs) == 2
    assert [r.run_metadata.run_id for r in runs] == [
        "run_2026-01-01T00-00-00",
        "run_2026-01-03T00-00-00",
    ]


# --------------------------------------------------------------------------------------
# Judge prompt version guard
# --------------------------------------------------------------------------------------


def test_warns_when_runs_used_different_judge_prompt_versions(make_run, make_scored_result, capsys):
    """A changed judge prompt moves scores on its own, so a delta across that boundary mixes
    'the feature got worse' with 'we changed the ruler'."""
    baseline = make_run("r1", [make_scored_result("tc_1", True)])
    current = make_run("r2", [make_scored_result("tc_1", True)])
    baseline.run_metadata.judge_prompt_version = None  # pre-versioning run
    current.run_metadata.judge_prompt_version = "v2"

    compare_runs(baseline, current, {"tc_1": "billing"}, ThresholdConfig())

    output = capsys.readouterr().out
    assert "different judge prompt versions" in output
    assert "unversioned" in output


def test_no_warning_when_judge_versions_match(make_run, make_scored_result, capsys):
    baseline = make_run("r1", [make_scored_result("tc_1", True)])
    current = make_run("r2", [make_scored_result("tc_1", True)])
    baseline.run_metadata.judge_prompt_version = "v2"
    current.run_metadata.judge_prompt_version = "v2"

    compare_runs(baseline, current, {"tc_1": "billing"}, ThresholdConfig())

    assert "different judge prompt versions" not in capsys.readouterr().out


def test_list_runs_does_not_descend_into_subdirectories(tmp_path, make_run, make_scored_result):
    """Runs scored under the old judge are parked in src/runs/archive/. They stay out of history
    only because list_runs() globs one level - if that became recursive they would silently
    return as baselines."""
    active = make_run("run_2026-01-01T00-00-00", [make_scored_result("tc_1", True)])
    (tmp_path / "run_2026-01-01T00-00-00.json").write_text(active.model_dump_json())

    archive = tmp_path / "archive"
    archive.mkdir()
    stale = make_run("run_2025-01-01T00-00-00", [make_scored_result("tc_1", False)])
    (archive / "run_2025-01-01T00-00-00.json").write_text(stale.model_dump_json())

    found = [p.name for p in list_runs(str(tmp_path))]
    assert found == ["run_2026-01-01T00-00-00.json"]


# --------------------------------------------------------------------------------------
# Absolute quality floor
# --------------------------------------------------------------------------------------


def test_floor_is_off_by_default_so_a_flat_low_quality_run_passes(make_run, make_scored_result):
    """The exact gap the floor exists to close: 33/50 passing, nothing regressed, so the
    relative check reports PASS. Off by default, this is still the behaviour."""
    results = [make_scored_result(f"tc_{i}", i < 33) for i in range(50)]
    baseline = make_run("r1", results)
    current = make_run("r2", list(results))
    cat_map = {f"tc_{i}": "billing" for i in range(50)}

    result = compare_runs(baseline, current, cat_map, ThresholdConfig())
    assert result.current_pass_rate == pytest.approx(0.66)
    assert result.status == "pass"
    assert result.below_minimum_pass_rate is False


def test_floor_fails_a_low_quality_run_even_with_no_regression(make_run, make_scored_result):
    results = [make_scored_result(f"tc_{i}", i < 33) for i in range(50)]
    baseline = make_run("r1", results)
    current = make_run("r2", list(results))
    cat_map = {f"tc_{i}": "billing" for i in range(50)}

    result = compare_runs(baseline, current, cat_map, ThresholdConfig(minimum_pass_rate=0.80))
    assert result.status == "critical"
    assert result.below_minimum_pass_rate is True
    assert result.pass_rate_delta == pytest.approx(0.0)  # critical despite zero regression


def test_floor_allows_a_run_at_or_above_the_minimum(make_run, make_scored_result):
    results = [make_scored_result(f"tc_{i}", i < 40) for i in range(50)]  # exactly 80%
    baseline = make_run("r1", results)
    current = make_run("r2", list(results))
    cat_map = {f"tc_{i}": "billing" for i in range(50)}

    result = compare_runs(baseline, current, cat_map, ThresholdConfig(minimum_pass_rate=0.80))
    assert result.status == "pass"
    assert result.below_minimum_pass_rate is False


def test_regression_still_reported_when_floor_also_breached(make_run, make_scored_result):
    """A run can be both below the floor and regressing - the regressed cases must still
    be listed, not swallowed by the floor verdict."""
    baseline = make_run("r1", [make_scored_result(f"tc_{i}", True) for i in range(50)])
    current = make_run("r2", [make_scored_result(f"tc_{i}", i < 30) for i in range(50)])
    cat_map = {f"tc_{i}": "billing" for i in range(50)}

    result = compare_runs(baseline, current, cat_map, ThresholdConfig(minimum_pass_rate=0.80))
    assert result.status == "critical"
    assert result.below_minimum_pass_rate is True
    assert len(result.regressions) == 20


def test_control_flow_exceptions_are_not_valueerrors():
    """The pipeline catches these specifically. If either were a ValueError subclass, Pydantic's
    ValidationError (which IS-A ValueError) would be swallowed by the same handler - the bug
    that made a corrupt run file report as 'first run' and exit 0."""
    assert not issubclass(InsufficientRunHistory, ValueError)
    assert not issubclass(IncompleteEvalRun, ValueError)
