import asyncio

from .clients.llm_client import get_client
from .errors import IncompleteEvalRun
from .eval_runner import eval_runner
from .llm_judge import llm_as_judge
from .loaders import load_golden_dataset, load_prompt_config
from .models import GoldenCase, RawResult, ScoredResult
from .settings import CompletionConfig

DEFAULT_PROMPT_PATH = "./prompts/v6_classifier.yaml"
DEFAULT_DATASET_PATH = "./datasets/golden_dataset_v1.json"


def completion_rate(completed: int, total: int) -> float:
    """Fraction of dataset cases that produced a usable score. 0.0 for an empty dataset."""
    if total <= 0:
        return 0.0
    return completed / total


async def score_one(
    generated_result: RawResult,
    golden_case: GoldenCase,
    semaphore: asyncio.Semaphore,
    judge_client=None,
) -> ScoredResult | None:
    """Scores each result against its golden counter part"""

    async with semaphore:
        expected = golden_case.expected_output
        try:
            summary_score = await llm_as_judge(
                generated_result.summary, expected.summary, judge_client
            )
        except Exception as e:
            print(f"SKIPPED scoring case {golden_case.id}: {type(e).__name__}: {e}")
            return None
        category_match = generated_result.category == expected.category
        print(f"Successfully scored an eval for case: {golden_case.id}")
        return ScoredResult(
            test_case_id=golden_case.id,
            category_match=category_match,
            summary_score=summary_score,
            latency_seconds=generated_result.latency,
            prompt_tokens=generated_result.prompt_tokens,
            completion_tokens=generated_result.completion_tokens,
            passed=category_match and summary_score >= 4,
            generated_category=generated_result.category,
            generated_summary=generated_result.summary,
            expected_category=expected.category,
            expected_summary=expected.summary,
        )


async def scorer(
    prompt_path: str = DEFAULT_PROMPT_PATH,
    dataset_path: str = DEFAULT_DATASET_PATH,
    max_concurrency: int = 10,
    completion: CompletionConfig | None = None,
    llm_client=None,
    judge_client=None,
) -> list[ScoredResult]:
    """Runs the eval, then scores each result against its golden case, matched by id.

    Matched by id rather than list position because eval_runner() can return fewer results than
    the dataset has cases, so positions no longer line up. Skipped cases are excluded rather
    than failed - an API timeout is not a quality regression - and the completion gate stops a
    heavily-degraded run from reporting a confident pass rate over a small surviving sample.
    """
    completion = completion or CompletionConfig.from_env()
    # Resolve up front: score_one() treats any exception as a skipped case, so a missing API key
    # would otherwise surface as N skipped cases instead of one clear startup error.
    judge_client = judge_client or get_client()
    config = load_prompt_config(prompt_path)
    generated_results = await eval_runner(dataset_path, config, max_concurrency, llm_client)
    golden_dataset = load_golden_dataset(dataset_path)
    golden_by_id = {case.id: case for case in golden_dataset.cases}
    semaphore = asyncio.Semaphore(max_concurrency)
    print(f"Successfully ran evaluation on {dataset_path}!")
    print("=======================================================\n")

    print("=======================================================")
    print("Scoring evaluations with llm judge.......")

    tasks = []
    for generated_result in generated_results:
        golden_case = golden_by_id.get(generated_result.test_id)
        if golden_case is None:
            print(f"SKIPPED scoring case {generated_result.test_id}: no matching golden case found")
            continue
        tasks.append(score_one(generated_result, golden_case, semaphore, judge_client))

    scored_results = await asyncio.gather(*tasks)
    results = [r for r in scored_results if r is not None]
    skipped = len(scored_results) - len(results)
    if skipped:
        print(f"WARNING: {skipped}/{len(scored_results)} case(s) failed scoring and were skipped.")

    total_cases = len(golden_dataset.cases)
    rate = completion_rate(len(results), total_cases)
    print(f"Completed {len(results)}/{total_cases} case(s) ({rate:.1%}).")
    if rate < completion.minimum_completion_rate:
        raise IncompleteEvalRun(
            f"Only {len(results)}/{total_cases} case(s) completed ({rate:.1%}), below the minimum "
            f"completion rate of {completion.minimum_completion_rate:.1%}. Refusing to report a "
            f"pass rate over this sample - set MIN_COMPLETION_RATE to change this gate."
        )

    return results
