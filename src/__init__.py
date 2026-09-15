"""Load .env once, for every entry point.

Lives here because `python -m src.<anything>` imports this package first. Previously it ran as a
side effect of importing the OpenAI client, so only entry points that reached the client saw
.env - a configured SLACK_WEBHOOK_URL was silently ignored by the rest.

Real environment variables win over .env, so CI secrets take precedence.
"""

from dotenv import load_dotenv

load_dotenv()
