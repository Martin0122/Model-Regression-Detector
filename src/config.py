from typing import Literal
from pydantic import BaseModel, Field

Category = Literal["billing", "technical", "account", "general"]

class ClassificationOutput(BaseModel):
    category: Category
    summary: str

class FewShotExample(BaseModel):
    input: str
    output: ClassificationOutput

class PromptConfig(BaseModel):
    version_id: str
    created_at: str
    model: str
    system_prompt: str
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)

class GoldenExpectedOutput(BaseModel):
    category: Category
    summary: str

class GoldenCase(BaseModel):
    id: str
    input: str
    expected_output: GoldenExpectedOutput
    expected_difficulty: str
    edge_case_tags: list[str] = Field(default_factory=list)
    notes: str

class GoldenDataset(BaseModel):
    """The authoritative golden dataset schema (datasets/golden_dataset_v1.json).
    Parsing through this model means a malformed or drifted dataset fails loudly at load
    time instead of surfacing as a KeyError deep inside the eval loop."""
    dataset_version: str
    created_at: str
    feature: str
    status: str
    expected_categories: list[str]
    expected_difficulties: list[str]
    cases: list[GoldenCase]

class RawResult(BaseModel):
    test_id: str
    category: Category
    summary: str
    latency: float
    prompt_tokens: int
    completion_tokens: int

class ScoredResult(BaseModel):
    test_case_id: str
    category_match: bool
    summary_score: int
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    passed: bool
    generated_category: Category | None = None
    generated_summary: str | None = None
    expected_category: Category | None = None
    expected_summary: str | None = None

class RunMetadata(BaseModel):
    run_id: str
    prompt_version: str
    model: str
    judge_model: str
    timestamp: str
    # Optional so run files written before these fields existed still load.
    total_cases: int | None = None
    completed_cases: int | None = None
    # None means "written before judge prompts were versioned" - i.e. scored under the old,
    # self-contradictory judge instructions, so not comparable with current runs.
    judge_prompt_version: str | None = None

class EvalRun(BaseModel):
    run_metadata: RunMetadata
    results: list[ScoredResult]

class ThresholdConfig(BaseModel):
    warning_threshold: float = 0.03
    critical_threshold: float = 0.08

    @classmethod
    def from_env(cls) -> "ThresholdConfig":
        import os
        return cls(
            warning_threshold=float(os.getenv("REGRESSION_WARNING_THRESHOLD", 0.03)),
            critical_threshold=float(os.getenv("REGRESSION_CRITICAL_THRESHOLD", 0.08)),
        )

class CompletionConfig(BaseModel):
    """Gate on how much of the dataset must actually complete for a run to be trustworthy."""
    minimum_completion_rate: float = 0.95

    @classmethod
    def from_env(cls) -> "CompletionConfig":
        import os
        return cls(minimum_completion_rate=float(os.getenv("MIN_COMPLETION_RATE", 0.95)))

class CaseFlip(BaseModel):
    test_case_id: str
    category: str | None

class CategoryDelta(BaseModel):
    category: str
    previous_accuracy: float
    current_accuracy: float
    delta: float

Status = Literal["pass", "warning", "critical"]

class ComparisonResult(BaseModel):
    baseline_run_id: str
    current_run_id: str
    previous_pass_rate: float
    current_pass_rate: float
    pass_rate_delta: float
    category_deltas: list[CategoryDelta]
    regressions: list[CaseFlip]
    improvements: list[CaseFlip]
    status: Status

class DriftConfig(BaseModel):
    window_size: int = 7
    warning_threshold: float = 0.03
    critical_threshold: float = 0.08

    @classmethod
    def from_env(cls) -> "DriftConfig":
        import os
        return cls(
            window_size=int(os.getenv("DRIFT_WINDOW_SIZE", 7)),
            warning_threshold=float(os.getenv("DRIFT_WARNING_THRESHOLD", 0.03)),
            critical_threshold=float(os.getenv("DRIFT_CRITICAL_THRESHOLD", 0.08)),
        )

class DriftResult(BaseModel):
    window_size: int
    baseline_run_id: str
    baseline_moving_average: float
    current_run_id: str
    current_moving_average: float
    drift_delta: float
    status: Status
