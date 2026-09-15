"""CI entrypoint: run the eval, score it, save the run, then compare/report/alert.

Usage: python -m src.pipeline
Exit code is 1 when the comparison status is "critical" (used to block PR merges), 0 otherwise.
"""

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from . import report as report_module
from .classifier import CLASSIFIER_TEMPERATURE
from .errors import IncompleteEvalRun, InsufficientRunHistory, MissingAPIKey
from .formatting import flip_counts_line, floor_line, pass_rate_line
from .llm_judge import JUDGE_MODEL, JUDGE_PROMPT_VERSION, JUDGE_TEMPERATURE
from .loaders import load_golden_dataset, load_prompt_config
from .models import ComparisonResult, DriftResult, EvalRun, RunMetadata
from .scoring import DEFAULT_DATASET_PATH, DEFAULT_PROMPT_PATH, scorer


def save_run(eval_run: EvalRun, output_dir: str = "./src/runs") -> Path:
    safe_id = eval_run.run_metadata.run_id.replace(":", "-")
    output_path = Path(output_dir) / f"{safe_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(eval_run.model_dump_json(indent=2))
    return output_path


async def run_eval(
    prompt_path: str = DEFAULT_PROMPT_PATH,
    dataset_path: str = DEFAULT_DATASET_PATH,
    llm_client=None,
    judge_client=None,
) -> EvalRun:
    config = load_prompt_config(prompt_path)
    results = await scorer(
        prompt_path, dataset_path, llm_client=llm_client, judge_client=judge_client
    )
    total_cases = len(load_golden_dataset(dataset_path).cases)
    timestamp = datetime.now(UTC).isoformat()
    return EvalRun(
        run_metadata=RunMetadata(
            run_id=f"run_{timestamp}",
            prompt_version=config.version_id,
            model=config.model,
            judge_model=JUDGE_MODEL,
            judge_prompt_version=JUDGE_PROMPT_VERSION,
            judge_temperature=JUDGE_TEMPERATURE,
            classifier_temperature=CLASSIFIER_TEMPERATURE,
            timestamp=timestamp,
            total_cases=total_cases,
            completed_cases=len(results),
        ),
        results=results,
    )


def write_pipeline_summary(
    comparison: ComparisonResult | None,
    report_path: Path | None,
    drift: DriftResult | None,
    aborted_reason: str | None = None,
) -> None:
    if aborted_reason is not None:
        lines = ["## Eval pipeline result: ABORTED", "", aborted_reason]
    elif comparison is None:
        lines = [
            "## Eval pipeline result: FIRST RUN",
            "",
            "No prior run exists yet, so there's nothing to compare against.",
        ]
    else:
        lines = [
            f"## Eval pipeline result: {comparison.status.upper()}",
            "",
            pass_rate_line(comparison),
            flip_counts_line(comparison),
        ]
        if floor := floor_line(comparison):
            lines.append("")
            lines.append(
                f"**{floor}.** This fails the run regardless of whether anything "
                f"regressed since the last one."
            )
        if comparison.regressions:
            lines.append("")
            lines.append("### Regressed cases")
            lines.extend(
                f"- `{flip.test_case_id}` ({flip.category})" for flip in comparison.regressions
            )
        if drift is not None and drift.status != "pass":
            lines.append("")
            lines.append(
                f"**Slow drift detected:** {drift.status.upper()} "
                f"({drift.drift_delta:+.1%} over {drift.window_size} runs)"
            )
        if report_path is not None:
            lines.append("")
            lines.append(f"[Full HTML report]({report_path})")

    summary = "\n".join(lines) + "\n"

    Path("./reports").mkdir(exist_ok=True)
    Path("./reports/pipeline_summary.md").write_text(summary)

    step_summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary_path:
        with open(step_summary_path, "a") as f:
            f.write(summary)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Run the eval, score it, save the run, then compare/report/alert.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT_PATH,
        help=f"Prompt config to evaluate (default: {DEFAULT_PROMPT_PATH}). Point this at a "
        f"candidate prompt to test it against the existing baseline before promoting it.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_PATH,
        help=f"Golden dataset to evaluate against (default: {DEFAULT_DATASET_PATH}).",
    )
    return parser.parse_args(argv)


async def main(
    prompt_path: str = DEFAULT_PROMPT_PATH, dataset_path: str = DEFAULT_DATASET_PATH
) -> int:
    try:
        eval_run = await run_eval(prompt_path, dataset_path)
    except IncompleteEvalRun as e:
        # Too much of the dataset failed to evaluate for the result to mean anything. The run is
        # deliberately NOT saved - persisting a partial run would skew future baselines and drift.
        print(f"ABORTED: {e}")
        write_pipeline_summary(None, None, None, aborted_reason=str(e))
        return 1

    run_path = save_run(eval_run)
    print(f"Saved run to {run_path}")

    try:
        comparison, report_path, drift = report_module.main()
    except InsufficientRunHistory as e:
        print(f"{e} Skipping regression/drift gate.")
        write_pipeline_summary(None, None, None)
        return 0

    write_pipeline_summary(comparison, report_path, drift)

    print(f"Pipeline status: {comparison.status.upper()}")
    if comparison.status == "critical":
        return 1

    # Drift is a property of the trend, not of this PR - by default it alerts loudly but does
    # not block, because the PR author usually cannot fix a slow decay originating upstream and
    # blocking every PR until the trend recovers would stall the whole team. Teams that want a
    # hard stop can opt in with BLOCK_ON_CRITICAL_DRIFT=true.
    if drift is not None and drift.status == "critical":
        if os.getenv("BLOCK_ON_CRITICAL_DRIFT", "false").lower() == "true":
            print("Critical drift detected and BLOCK_ON_CRITICAL_DRIFT is set; failing the run.")
            return 1
        print(
            "Critical drift detected (reported, not blocking; "
            "set BLOCK_ON_CRITICAL_DRIFT=true to gate on it)."
        )

    return 0


if __name__ == "__main__":
    args = parse_args()
    try:
        sys.exit(asyncio.run(main(args.prompt, args.dataset)))
    except MissingAPIKey as e:
        # The most common first-run failure, especially in the container. A traceback here
        # buries the one line that tells you what to do.
        print(f"ERROR: {e}")
        print("Set it in .env, or pass it to the container with: docker run -e OPENAI_API_KEY=...")
        sys.exit(1)
