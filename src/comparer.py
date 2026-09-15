from pathlib import Path

from .errors import InsufficientRunHistory
from .formatting import floor_line, pass_rate_line
from .loaders import load_golden_dataset
from .models import CaseFlip, CategoryDelta, ComparisonResult, EvalRun, ScoredResult
from .settings import ThresholdConfig


def list_runs(runs_dir: str = "./src/runs") -> list[Path]:
    """Run files oldest -> newest. Filenames are ISO8601, so string sort is chronological."""
    return sorted(Path(runs_dir).glob("run_*.json"))


def load_run(path: Path) -> EvalRun:
    return EvalRun.model_validate_json(path.read_text())


def load_runs_safe(paths: list[Path]) -> list[EvalRun]:
    """Loads each run, skipping corrupt files with a warning rather than letting one bad
    historical file break comparison, trend and drift for every run after it."""
    runs = []
    for path in paths:
        try:
            runs.append(load_run(path))
        except Exception as e:
            print(f"WARNING: skipping unreadable run file {path.name}: {type(e).__name__}: {e}")
    return runs


def build_category_map(dataset_path: str = "./datasets/golden_dataset_v1.json") -> dict[str, str]:
    dataset = load_golden_dataset(dataset_path)
    return {case.id: case.expected_output.category for case in dataset.cases}


def pass_rate(results: list[ScoredResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.passed) / len(results)


def _category_accuracy(
    results: list[ScoredResult], category_map: dict[str, str]
) -> dict[str, float]:
    totals: dict[str, int] = {}
    matches: dict[str, int] = {}
    for result in results:
        category = category_map.get(result.test_case_id)
        if category is None:
            continue
        totals[category] = totals.get(category, 0) + 1
        if result.category_match:
            matches[category] = matches.get(category, 0) + 1
    return {category: matches.get(category, 0) / total for category, total in totals.items()}


def warn_on_judge_mismatch(baseline: EvalRun, current: EvalRun) -> bool:
    """Warns when two runs were scored under different judge instructions.

    A changed judge prompt can move scores on its own, so a delta across that boundary mixes
    'the feature got worse' with 'we changed the ruler'. Returns True if a mismatch was found.
    """
    baseline_version = baseline.run_metadata.judge_prompt_version
    current_version = current.run_metadata.judge_prompt_version
    if baseline_version == current_version:
        return False

    print(
        f"WARNING: comparing runs scored under different judge prompt versions "
        f"(baseline={baseline_version or 'unversioned'}, "
        f"current={current_version or 'unversioned'}). "
        f"The delta partly reflects the changed judge, not just the feature. "
        f"Regenerate the baseline with the current judge to get a clean comparison."
    )
    return True


def compare_runs(
    baseline: EvalRun,
    current: EvalRun,
    category_map: dict[str, str],
    thresholds: ThresholdConfig | None = None,
) -> ComparisonResult:
    thresholds = thresholds or ThresholdConfig()
    warn_on_judge_mismatch(baseline, current)

    baseline_by_id = {r.test_case_id: r for r in baseline.results}
    current_by_id = {r.test_case_id: r for r in current.results}

    previous_pass_rate = pass_rate(baseline.results)
    current_pass_rate = pass_rate(current.results)
    pass_rate_delta = current_pass_rate - previous_pass_rate

    previous_accuracy = _category_accuracy(baseline.results, category_map)
    current_accuracy = _category_accuracy(current.results, category_map)
    category_deltas = [
        CategoryDelta(
            category=category,
            previous_accuracy=previous_accuracy.get(category, 0.0),
            current_accuracy=current_accuracy.get(category, 0.0),
            delta=current_accuracy.get(category, 0.0) - previous_accuracy.get(category, 0.0),
        )
        for category in sorted(set(previous_accuracy) | set(current_accuracy))
    ]

    regressions: list[CaseFlip] = []
    improvements: list[CaseFlip] = []
    for test_case_id, current_result in current_by_id.items():
        baseline_result = baseline_by_id.get(test_case_id)
        if baseline_result is None:
            continue
        if baseline_result.passed and not current_result.passed:
            regressions.append(
                CaseFlip(
                    test_case_id=test_case_id,
                    category=category_map.get(test_case_id),
                )
            )
        elif not baseline_result.passed and current_result.passed:
            improvements.append(
                CaseFlip(
                    test_case_id=test_case_id,
                    category=category_map.get(test_case_id),
                )
            )

    # round to avoid float artifacts (e.g. 1.0 - 0.92 == 0.07999999999999996) landing
    # on the wrong side of a threshold boundary
    drop = round(-pass_rate_delta, 9)
    if drop >= thresholds.critical_threshold:
        status = "critical"
    elif drop >= thresholds.warning_threshold:
        status = "warning"
    else:
        status = "pass"

    # Absolute floor, checked after the relative comparison. Without it a suite can hold a
    # poor pass rate forever and still report PASS, because nothing got *worse* this run.
    below_floor = (
        thresholds.minimum_pass_rate > 0
        and round(current_pass_rate, 9) < thresholds.minimum_pass_rate
    )
    if below_floor:
        status = "critical"

    return ComparisonResult(
        baseline_run_id=baseline.run_metadata.run_id,
        current_run_id=current.run_metadata.run_id,
        previous_pass_rate=previous_pass_rate,
        current_pass_rate=current_pass_rate,
        pass_rate_delta=pass_rate_delta,
        category_deltas=category_deltas,
        regressions=regressions,
        improvements=improvements,
        status=status,
        below_minimum_pass_rate=below_floor,
        minimum_pass_rate=thresholds.minimum_pass_rate,
    )


def compare_latest_two(
    runs_dir: str = "./src/runs", dataset_path: str = "./datasets/golden_dataset_v1.json"
) -> ComparisonResult:
    runs = load_runs_safe(list_runs(runs_dir))
    if len(runs) < 2:
        raise InsufficientRunHistory(
            f"Need at least 2 valid runs to compare, found {len(runs)} in {runs_dir}."
        )

    baseline = runs[-2]
    current = runs[-1]
    category_map = build_category_map(dataset_path)
    thresholds = ThresholdConfig.from_env()

    return compare_runs(baseline, current, category_map, thresholds)


if __name__ == "__main__":
    try:
        result = compare_latest_two()
    except InsufficientRunHistory as e:
        # An ordinary, expected state (fresh clone, archived baselines) - report it plainly
        # rather than dumping a traceback.
        print(f"{e} Run `python -m src.pipeline` to record runs first.")
        raise SystemExit(1) from None

    print("=======================================================")
    print(f"Comparing {result.baseline_run_id} -> {result.current_run_id}")
    print(pass_rate_line(result))
    print(f"Status: {result.status.upper()}")
    if floor := floor_line(result):
        print(f"  {floor}")
    print("\nPer-category accuracy:")
    for delta in result.category_deltas:
        print(
            f"  {delta.category}: {delta.previous_accuracy:.1%} -> "
            f"{delta.current_accuracy:.1%} ({delta.delta:+.1%})"
        )

    if result.regressions:
        print(f"\nRegressions ({len(result.regressions)}):")
        for flip in result.regressions:
            print(f"  {flip.test_case_id} ({flip.category})")

    if result.improvements:
        print(f"\nImprovements ({len(result.improvements)}):")
        for flip in result.improvements:
            print(f"  {flip.test_case_id} ({flip.category})")

    baseline_id = result.baseline_run_id.replace(":", "-")
    current_id = result.current_run_id.replace(":", "-")
    output_path = Path("./reports") / f"comparison_{baseline_id}_to_{current_id}.json"
    output_path.write_text(result.model_dump_json(indent=2))
    print(f"\nSaved comparison to {output_path}")
