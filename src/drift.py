from .config import EvalRun, DriftConfig, DriftResult
from .comparer import list_runs, load_runs_safe, pass_rate


def moving_averages(runs: list[EvalRun], window_size: int) -> list[tuple[str, float]]:
    """One (run_id, trailing moving average) point per run, once `window_size` runs are available."""
    rates = [pass_rate(r.results) for r in runs]
    points = []
    for i in range(window_size - 1, len(runs)):
        window = rates[i - window_size + 1: i + 1]
        points.append((runs[i].run_metadata.run_id, sum(window) / len(window)))
    return points


def detect_drift(runs_dir: str = "./src/runs", config: DriftConfig | None = None) -> DriftResult | None:
    """Compares the current N-run moving average pass rate against the earliest available N-run
    moving average. Catches gradual degradation that per-run diffs miss because no single run
    crosses the regression threshold. Returns None if there isn't enough run history yet."""
    config = config or DriftConfig.from_env()
    runs = load_runs_safe(list_runs(runs_dir))

    points = moving_averages(runs, config.window_size)
    if len(points) < 2:
        return None

    baseline_run_id, baseline_avg = points[0]
    current_run_id, current_avg = points[-1]
    drift_delta = current_avg - baseline_avg
    drop = round(-drift_delta, 9)

    if drop >= config.critical_threshold:
        status = "critical"
    elif drop >= config.warning_threshold:
        status = "warning"
    else:
        status = "pass"

    return DriftResult(
        window_size=config.window_size,
        baseline_run_id=baseline_run_id,
        baseline_moving_average=baseline_avg,
        current_run_id=current_run_id,
        current_moving_average=current_avg,
        drift_delta=drift_delta,
        status=status,
    )


if __name__ == "__main__":
    result = detect_drift()

    print("=======================================================")
    if result is None:
        config = DriftConfig.from_env()
        print(f"Not enough run history yet to compute drift (need at least {config.window_size + 1} runs).")
    else:
        print(f"Drift check (window={result.window_size})")
        print(f"Moving average: {result.baseline_moving_average:.1%} ({result.baseline_run_id}) -> {result.current_moving_average:.1%} ({result.current_run_id})")
        print(f"Drift: {result.drift_delta:+.1%}")
        print(f"Status: {result.status.upper()}")
        if result.status != "pass":
            print("SLOW DRIFT WARNING: rolling average has degraded beyond threshold with no single-run alert.")
