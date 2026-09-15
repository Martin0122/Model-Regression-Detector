"""CI entrypoint: run the eval, score it, save the run, then compare/report/alert.

Usage: python -m src.pipeline
Exit code is 1 when the comparison status is "critical" (used to block PR merges), 0 otherwise.
"""
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import ComparisonResult, DriftResult, EvalRun, RunMetadata
from .errors import IncompleteEvalRun, InsufficientRunHistory, MissingAPIKey
from .llm_judge import JUDGE_MODEL, JUDGE_PROMPT_VERSION
from .scoring import scorer, DEFAULT_PROMPT_PATH, DEFAULT_DATASET_PATH
from .services.load_configs import load_prompt_config
from .services.load_json import load_golden_dataset
from . import report as report_module


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
    results = await scorer(prompt_path, dataset_path, llm_client=llm_client, judge_client=judge_client)
    total_cases = len(load_golden_dataset(dataset_path).cases)
    timestamp = datetime.now(timezone.utc).isoformat()
    return EvalRun(
        run_metadata=RunMetadata(
            run_id=f"run_{timestamp}",
            prompt_version=config.version_id,
            model=config.model,
            judge_model=JUDGE_MODEL,
            judge_prompt_version=JUDGE_PROMPT_VERSION,
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
        lines = ["## Eval pipeline result: FIRST RUN", "", "No prior run exists yet, so there's nothing to compare against."]
    else:
        lines = [
            f"## Eval pipeline result: {comparison.status.upper()}",
            "",
            f"Pass rate: {comparison.previous_pass_rate:.1%} -> {comparison.current_pass_rate:.1%} ({comparison.pass_rate_delta:+.1%})",
            f"Regressions: {len(comparison.regressions)} | Improvements: {len(comparison.improvements)}",
        ]
        if comparison.regressions:
            lines.append("")
            lines.append("### Regressed cases")
            lines.extend(f"- `{flip.test_case_id}` ({flip.category})" for flip in comparison.regressions)
        if drift is not None and drift.status != "pass":
            lines.append("")
            lines.append(f"**Slow drift detected:** {drift.status.upper()} ({drift.drift_delta:+.1%} over {drift.window_size} runs)")
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


async def main() -> int:
    try:
        eval_run = await run_eval()
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
        print("Critical drift detected (reported, not blocking; set BLOCK_ON_CRITICAL_DRIFT=true to gate on it).")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except MissingAPIKey as e:
        # The most common first-run failure, especially in the container. A traceback here
        # buries the one line that tells you what to do.
        print(f"ERROR: {e}")
        print("Set it in .env, or pass it to the container with: docker run -e OPENAI_API_KEY=...")
        sys.exit(1)
