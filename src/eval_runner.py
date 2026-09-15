import asyncio
import time

from .classifier import classify_email
from .clients.llm_client import get_client
from .loaders import load_golden_dataset
from .models import GoldenCase, PromptConfig, RawResult


async def run_single_case(
    semaphore: asyncio.Semaphore, test_case: GoldenCase, config: PromptConfig, llm_client
) -> RawResult | None:
    async with semaphore:
        start = time.perf_counter()
        try:
            result = await classify_email(test_case.input, config, llm_client)
        except Exception as e:
            print(f"SKIPPED case {test_case.id}: {type(e).__name__}: {e}")
            return None
        latency = round(time.perf_counter() - start, 2)
        print(f"Successfully evaluated case: {test_case.id}")
        return RawResult(
            test_id=test_case.id,
            category=result["category"],
            summary=result["summary"],
            latency=latency,
            prompt_tokens=result["input_tokens"],
            completion_tokens=result["output_tokens"],
        )


async def eval_runner(
    test_cases_path: str, config: PromptConfig, max_concurrency: int = 10, llm_client=None
) -> list[RawResult]:
    """Returns a list of the raw results of the test cases ran against email classifier.
    A case whose API call fails (timeout, rate limit, etc.) is logged and skipped rather than
    aborting the whole batch - a transient error on one case shouldn't lose the other 49.

    Pass llm_client to inject a double in tests; it is resolved lazily otherwise, so importing
    this module never requires an API key."""

    if not test_cases_path:
        raise ValueError(f"ERROR: No path provided for test cases, path: {test_cases_path}")

    llm_client = llm_client or get_client()
    dataset = load_golden_dataset(test_cases_path)
    semaphore = asyncio.Semaphore(max_concurrency)

    print("=======================================================")
    print(f"Running evaluation on {test_cases_path}...")
    test_cases = [
        run_single_case(semaphore, test_case, config, llm_client) for test_case in dataset.cases
    ]
    raw_results = await asyncio.gather(*test_cases)

    results = [r for r in raw_results if r is not None]
    skipped = len(raw_results) - len(results)
    if skipped:
        print(f"WARNING: {skipped}/{len(raw_results)} case(s) failed and were skipped.")
    return results
