"""Pipeline-specific exceptions.

These deliberately subclass Exception rather than ValueError. Pydantic's ValidationError
is itself a ValueError subclass, so catching ValueError to detect an expected control-flow
condition (like "not enough runs yet") would also silently swallow genuine data-corruption
errors and misreport them as that expected condition.
"""


class InsufficientRunHistory(Exception):
    """Raised when there aren't enough valid runs on disk to perform a comparison."""


class IncompleteEvalRun(Exception):
    """Raised when too few test cases completed for a run's results to be trustworthy."""


class MissingAPIKey(RuntimeError):
    """Raised when a live evaluation is attempted without OPENAI_API_KEY.

    Subclasses RuntimeError so callers that only care that something went wrong keep working,
    while entry points can catch this specifically and print a one-line fix instead of a
    traceback - this is the first thing anyone hits running the container.
    """
