"""Manual end-to-end runner: executes a real eval against the live API and saves the run.

This makes real (paid) OpenAI calls. It delegates to src.pipeline so there is exactly one
implementation of "run an eval and write a run file" - duplicating that here previously let
the two copies drift apart (this one recorded the wrong judge model).

Usage: python -m tests.test_scorer
"""
import asyncio

from src.errors import IncompleteEvalRun
from src.pipeline import run_eval, save_run


async def main():
    try:
        eval_run = await run_eval()
    except IncompleteEvalRun as e:
        print(f"ABORTED: {e}")
        return

    print("Successfully scored evaluations!")
    print('=======================================================')
    save_run(eval_run)


if __name__ == "__main__":
    asyncio.run(main())
