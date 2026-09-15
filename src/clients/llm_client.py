from openai import AsyncOpenAI
from functools import lru_cache
import os

# .env is loaded once in src/__init__.py, which runs before any module in this package.

MISSING_KEY_MESSAGE = "OPENAI_API_KEY is required to run live evaluations."


@lru_cache(maxsize=1)
def get_client() -> AsyncOpenAI:
    """Create and cache the OpenAI client on first use."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(MISSING_KEY_MESSAGE)
    return AsyncOpenAI(api_key=api_key)
