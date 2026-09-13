from openai import AsyncOpenAI
from dotenv import load_dotenv
from functools import lru_cache
import os

load_dotenv()

MISSING_KEY_MESSAGE = "OPENAI_API_KEY is required to run live evaluations."


@lru_cache(maxsize=1)
def get_client() -> AsyncOpenAI:
    """Builds the OpenAI client on first use.

    Deliberately lazy: constructing the client at import time meant that merely importing a
    pipeline module crashed without a key, so offline work (unit tests, report generation,
    run comparison) needed a dummy key just to get past the import.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(MISSING_KEY_MESSAGE)
    return AsyncOpenAI(api_key=api_key)
