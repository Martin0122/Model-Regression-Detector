from .llm_judge import llm_as_judge
from .config import ScoredResult, RawResult
from .eval_runner import eval_runner
from .services.load_configs import load_prompt_config
from .services.load_json import load_dataset
import asyncio

async def score_one(generated_result: RawResult, golden_result: dict, semaphore: asyncio.Semaphore) -> ScoredResult:
    """Scores each result against its golden counter part"""

    async with semaphore:
        summary_score = await llm_as_judge(generated_result.summary, golden_result["expected_summary"])
        category_match = generated_result.category == golden_result["expected_category"]
        if isinstance(summary_score, int):
            print(f"Successfully scored an eval for case: {golden_result["id"]}")
        return ScoredResult(
            test_case_id=golden_result["id"],
            category_match=category_match,
            summary_score=summary_score,
            latency_seconds=generated_result.latency,
            prompt_tokens=generated_result.prompt_tokens,
            completion_tokens=generated_result.completion_tokens,
            passed=category_match and summary_score >= 4
        )

async def scorer() -> list[ScoredResult]:
    prompt_path = "./prompts/v1_classifier.yaml"
    dataset_path = './data/golden_dataset_v1.json'
    config = load_prompt_config(prompt_path)
    generated_result = await eval_runner(dataset_path, config) # Returns list of RawResult
    golden_result = load_dataset(dataset_path) # Loads golden result
    semaphore = asyncio.Semaphore(10)
    print(f'Successfully ran evaluation on {dataset_path}!')
    print('=======================================================\n')

    print('=======================================================')
    print(f'Scoring evaluations with llm judge.......')

    tasks = []
    for i in range(len(generated_result)):
        tasks.append(score_one(generated_result[i], golden_result["test_cases"][i], semaphore))
    
    return await asyncio.gather(*tasks)





