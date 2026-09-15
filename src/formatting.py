"""Shared phrasing for a comparison result.

The console, the PR summary and the Slack alert all report the same facts. Building the
sentences separately let them drift apart in wording and in which facts they bothered to
mention - the absolute-floor breach reached Slack but not the console, for instance.
"""

from .models import ComparisonResult, DriftResult


def pass_rate_line(comparison: ComparisonResult) -> str:
    return (
        f"Pass rate: {comparison.previous_pass_rate:.1%} -> "
        f"{comparison.current_pass_rate:.1%} ({comparison.pass_rate_delta:+.1%})"
    )


def floor_line(comparison: ComparisonResult) -> str | None:
    """Why a run is critical when nothing regressed. None when the floor was not breached."""
    if not comparison.below_minimum_pass_rate:
        return None
    return (
        f"Below the absolute quality floor: {comparison.current_pass_rate:.1%} is under the "
        f"{comparison.minimum_pass_rate:.1%} minimum (MIN_PASS_RATE)"
    )


def flip_counts_line(comparison: ComparisonResult) -> str:
    return (
        f"Regressions: {len(comparison.regressions)} | Improvements: {len(comparison.improvements)}"
    )


def regressed_case_ids(comparison: ComparisonResult, limit: int | None = None) -> str:
    flips = comparison.regressions if limit is None else comparison.regressions[:limit]
    return ", ".join(flip.test_case_id for flip in flips)


def drift_line(drift: DriftResult) -> str:
    return (
        f"{drift.window_size}-run moving average: {drift.baseline_moving_average:.1%} "
        f"({drift.baseline_run_id}) -> {drift.current_moving_average:.1%} "
        f"({drift.current_run_id})"
    )
