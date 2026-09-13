"""Drift detection: rolling moving averages and the slow-decay threshold."""
import pytest

from src.config import DriftConfig
from src.drift import detect_drift, moving_averages


def test_moving_average_values_are_correct(make_run, make_scored_result):
    # pass rates alternate 1, 0, 1, 0, 1
    runs = [make_run(f"r{i}", [make_scored_result("tc_1", i % 2 == 0)]) for i in range(5)]
    points = moving_averages(runs, window_size=3)
    assert [p[1] for p in points] == pytest.approx([2 / 3, 1 / 3, 2 / 3])


@pytest.mark.parametrize("n_runs,window,expected_points", [
    (3, 7, 0),   # fewer runs than the window - nothing to say yet
    (7, 7, 1),   # exactly one full window
    (8, 7, 2),   # first point that can be compared against a baseline
    (5, 3, 3),
])
def test_point_count_follows_runs_minus_window_plus_one(make_run, make_scored_result, n_runs, window, expected_points):
    runs = [make_run(f"r{i}", [make_scored_result("tc_1", True)]) for i in range(n_runs)]
    assert len(moving_averages(runs, window_size=window)) == expected_points


def test_detect_drift_returns_none_without_enough_history(tmp_path, make_run, make_scored_result):
    for i in range(3):
        run = make_run(f"run_2026-01-0{i + 1}T00-00-00", [make_scored_result("tc_1", True)])
        (tmp_path / f"run_2026-01-0{i + 1}T00-00-00.json").write_text(run.model_dump_json())
    assert detect_drift(str(tmp_path), DriftConfig(window_size=7)) is None


def _write_drift_history(tmp_path, make_run, make_scored_result, failing_in_second_half):
    """7 perfect runs, then 7 runs where `failing_in_second_half` of 100 cases fail."""
    index = 0
    for _ in range(7):
        results = [make_scored_result(f"tc_{j}", True) for j in range(100)]
        run_id = f"run_2026-01-{index + 1:02d}T00-00-00"
        (tmp_path / f"{run_id}.json").write_text(make_run(run_id, results).model_dump_json())
        index += 1
    for _ in range(7):
        results = [make_scored_result(f"tc_{j}", True) for j in range(100)]
        for j in range(failing_in_second_half):
            results[j] = make_scored_result(f"tc_{j}", False)
        run_id = f"run_2026-01-{index + 1:02d}T00-00-00"
        (tmp_path / f"{run_id}.json").write_text(make_run(run_id, results).model_dump_json())
        index += 1


def test_slow_drift_is_flagged_critical(tmp_path, make_run, make_scored_result):
    _write_drift_history(tmp_path, make_run, make_scored_result, failing_in_second_half=8)
    result = detect_drift(str(tmp_path), DriftConfig(window_size=7))
    assert result is not None
    assert result.baseline_moving_average == pytest.approx(1.0)
    assert result.current_moving_average == pytest.approx(0.92)
    assert result.status == "critical"


def test_stable_history_does_not_flag_drift(tmp_path, make_run, make_scored_result):
    _write_drift_history(tmp_path, make_run, make_scored_result, failing_in_second_half=0)
    result = detect_drift(str(tmp_path), DriftConfig(window_size=7))
    assert result is not None
    assert result.status == "pass"
    assert result.drift_delta == pytest.approx(0.0)
