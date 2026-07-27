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

class RunMetadata(BaseModel):
    run_id: str
    prompt_version: str
    model: str
    judge_model: str
    timestamp: str

class EvalRun(BaseModel):
    run_metadata: RunMetadata
    results: list[ScoredResult]