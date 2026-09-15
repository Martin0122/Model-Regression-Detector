"""Load .env once, for every entry point.

This lives here rather than in a module further down the tree because `python -m src.<anything>`
imports this package first. It used to run as a side effect of importing the OpenAI client, which
meant configuration silently depended on the import chain: `src.pipeline` reached the client and
saw .env, while `src.notifier`, `src.report`, `src.comparer` and `src.drift` did not - so a
configured SLACK_WEBHOOK_URL was ignored and env-based thresholds fell back to their defaults.

Real environment variables take precedence (override=False), so CI secrets win over a local .env.
"""
from dotenv import load_dotenv

load_dotenv()
