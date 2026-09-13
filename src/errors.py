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
