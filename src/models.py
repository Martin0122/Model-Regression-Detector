"""Data schemas: the prompt, the golden dataset, and the shape of a run and its comparison."""

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["billing", "technical", "account", "general"]
Status = Literal["pass", "warning", "critical"]


# --- The feature under test -----------------------------------------------------------


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


# --- Golden dataset -------------------------------------------------------------------


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
    dataset_version: str
    created_at: str
    feature: str
    status: str
    expected_categories: list[str]
    expected_difficulties: list[str]
    cases: list[GoldenCase]


# --- A run ----------------------------------------------------------------------------


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
    # All optional: None means "recorded before this field was tracked". Runs missing
    # judge_prompt_version or the temperatures are not comparable with current ones - the
    # judge instructions and the sampler both changed under them.
    total_cases: int | None = None
    completed_cases: int | None = None
    judge_prompt_version: str | None = None
    judge_temperature: float | None = None
    classifier_temperature: float | None = None


class EvalRun(BaseModel):
    run_metadata: RunMetadata
    results: list[ScoredResult]


# --- Comparing runs -------------------------------------------------------------------


class CaseFlip(BaseModel):
    test_case_id: str
    category: str | None


class CategoryDelta(BaseModel):
    category: str
    previous_accuracy: float
    current_accuracy: float
    delta: float


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
    # Distinguishes failing the absolute floor from regressing - they mean different things.
    below_minimum_pass_rate: bool = False
    minimum_pass_rate: float = 0.0


class DriftResult(BaseModel):
    window_size: int
    baseline_run_id: str
    baseline_moving_average: float
    current_run_id: str
    current_moving_average: float
    drift_delta: float
    status: Status
