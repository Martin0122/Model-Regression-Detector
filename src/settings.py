"""Runtime configuration read from the environment.

Every threshold is env-overridable so behaviour can change without a rebuild - set them in
.env locally, as repo variables in CI, or with -e on the container. Defaults live here.
"""

import os

from pydantic import BaseModel


class ThresholdConfig(BaseModel):
    warning_threshold: float = 0.03
    critical_threshold: float = 0.08
    # Absolute floor, disabled by default. The relative thresholds only ask whether a change
    # made things worse, so a suite can sit at a poor pass rate indefinitely and still report
    # PASS. Enabling this above the current pass rate fails every run, so it is a ratchet.
    minimum_pass_rate: float = 0.0

    @classmethod
    def from_env(cls) -> "ThresholdConfig":
        return cls(
            warning_threshold=float(os.getenv("REGRESSION_WARNING_THRESHOLD", 0.03)),
            critical_threshold=float(os.getenv("REGRESSION_CRITICAL_THRESHOLD", 0.08)),
            minimum_pass_rate=float(os.getenv("MIN_PASS_RATE", 0.0)),
        )


class CompletionConfig(BaseModel):
    """How much of the dataset must complete for a run to be trustworthy."""

    minimum_completion_rate: float = 0.95

    @classmethod
    def from_env(cls) -> "CompletionConfig":
        return cls(minimum_completion_rate=float(os.getenv("MIN_COMPLETION_RATE", 0.95)))


class DriftConfig(BaseModel):
    window_size: int = 7
    warning_threshold: float = 0.03
    critical_threshold: float = 0.08

    @classmethod
    def from_env(cls) -> "DriftConfig":
        return cls(
            window_size=int(os.getenv("DRIFT_WINDOW_SIZE", 7)),
            warning_threshold=float(os.getenv("DRIFT_WARNING_THRESHOLD", 0.03)),
            critical_threshold=float(os.getenv("DRIFT_CRITICAL_THRESHOLD", 0.08)),
        )
