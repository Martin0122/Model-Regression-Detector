
from .config import PromptConfig, RawResult
from .clients.llm_client import client
from .classifier import classify_email
from .services.load_json import load_dataset
import asyncio
import time


async def run_single_case(semaphore: asyncio.Semaphore, test_case: dict, config: PromptConfig) -> RawResult:
    # Use the semaphore asynchronously, will only allow X amount of coroutines concurrently
    async with semaphore:
        start = time.perf_counter()
        result = await classify_email(test_case["email_text"], config, client)
        latency = round(time.perf_counter() - start, 2)
        print(f"Successfully evaluated case: {test_case["id"]}")
        return RawResult(
            test_id=test_case["id"], 
            category=result["category"], 
            summary=result["summary"], 
            latency=latency, 
            prompt_tokens=result["input_tokens"], 
            completion_tokens=result["output_tokens"])

async def eval_runner(test_cases_path: str, config: PromptConfig, max_concurrency: int = 10) -> list[RawResult]:
    """Returns a list of the raw results of the test cases ran against email classifier"""

    if not test_cases_path:
        raise ValueError(f"ERROR: No path provided for test cases, path: {test_cases_path}")
    
    dataset = load_dataset(test_cases_path)
    semaphore = asyncio.Semaphore(max_concurrency)

    print('=======================================================')
    print(f"Running evaluation on {test_cases_path}...")
    # Function 'run_single_case' isn't automatically executed since its just a coroutine now since we made the function async
    test_cases = [run_single_case(semaphore, test_case, config) for test_case in dataset["test_cases"]]
    return await asyncio.gather(*test_cases)