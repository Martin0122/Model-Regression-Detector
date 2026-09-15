"""Shared fixtures.

No dummy API key is needed: the OpenAI client is built lazily by
src/clients/llm_client.get_client(), so importing pipeline modules never touches credentials.
Tests inject fake clients explicitly through the llm_client / judge_client parameters.
"""

import pytest

from src.models import (
    ClassificationOutput,
    ComparisonResult,
    EvalRun,
    GoldenCase,
    GoldenExpectedOutput,
    PromptConfig,
    RunMetadata,
    ScoredResult,
)

# --------------------------------------------------------------------------------------
# Fake OpenAI doubles
# --------------------------------------------------------------------------------------


class FakeUsage:
    def __init__(self, input_tokens=100, output_tokens=20):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeClassifyResponse:
    def __init__(self, category, summary):
        self.output_parsed = ClassificationOutput(category=category, summary=summary)
        self.usage = FakeUsage()


class FakeResponses:
    """Stands in for client.responses - used by classify_email()."""

    def __init__(self, answer_fn):
        self.answer_fn = answer_fn
        self.calls = 0
        self.last_kwargs = {}

    async def parse(self, model, input, text_format, **kwargs):
        self.calls += 1
        self.last_kwargs = {"model": model, **kwargs}
        email = input[-1]["content"]
        category, summary = self.answer_fn(email)
        return FakeClassifyResponse(category, summary)


class FakeClassifierClient:
    def __init__(self, answer_fn):
        self.responses = FakeResponses(answer_fn)


class _FakeJudgeMessage:
    def __init__(self, score):
        self.parsed = type("Parsed", (), {"score": score})()
        self.refusal = None


class _FakeJudgeCompletions:
    def __init__(self, score_fn):
        self.score_fn = score_fn
        self.calls = 0
        self.last_kwargs = {}

    async def parse(self, model, messages, response_format, **kwargs):
        self.calls += 1
        self.last_kwargs = {"model": model, **kwargs}
        score = self.score_fn(messages[0]["content"])
        message = _FakeJudgeMessage(score)
        return type("Resp", (), {"choices": [type("Choice", (), {"message": message})()]})()


class FakeJudgeClient:
    """Stands in for the client used by llm_as_judge()."""

    def __init__(self, score_fn):
        completions = _FakeJudgeCompletions(score_fn)
        self.completions = completions
        chat = type("Chat", (), {"completions": completions})()
        self.beta = type("Beta", (), {"chat": chat})()


@pytest.fixture
def fake_classifier_client():
    def _make(answer_fn=None):
        answer_fn = answer_fn or (lambda email: ("billing", "a generated summary"))
        return FakeClassifierClient(answer_fn)

    return _make


@pytest.fixture
def fake_judge_client():
    def _make(score_fn=None):
        score_fn = score_fn or (lambda prompt: 5)
        return FakeJudgeClient(score_fn)

    return _make


# --------------------------------------------------------------------------------------
# Domain object builders
# --------------------------------------------------------------------------------------


@pytest.fixture
def make_scored_result():
    def _make(test_case_id, passed, category_match=None, summary_score=5, **kwargs):
        return ScoredResult(
            test_case_id=test_case_id,
            category_match=category_match if category_match is not None else passed,
            summary_score=summary_score,
            latency_seconds=kwargs.get("latency_seconds", 1.0),
            prompt_tokens=kwargs.get("prompt_tokens", 10),
            completion_tokens=kwargs.get("completion_tokens", 5),
            passed=passed,
            generated_category=kwargs.get("generated_category"),
            generated_summary=kwargs.get("generated_summary"),
            expected_category=kwargs.get("expected_category"),
            expected_summary=kwargs.get("expected_summary"),
        )

    return _make


@pytest.fixture
def make_run():
    def _make(run_id, results, prompt_version="v1", model="gpt-4o-mini", judge_model="gpt-4o-mini"):
        return EvalRun(
            run_metadata=RunMetadata(
                run_id=run_id,
                prompt_version=prompt_version,
                model=model,
                judge_model=judge_model,
                timestamp=run_id,
            ),
            results=results,
        )

    return _make


@pytest.fixture
def make_comparison():
    def _make(
        status="pass",
        previous=1.0,
        current=1.0,
        regressions=None,
        improvements=None,
        category_deltas=None,
    ):
        return ComparisonResult(
            baseline_run_id="r1",
            current_run_id="r2",
            previous_pass_rate=previous,
            current_pass_rate=current,
            pass_rate_delta=current - previous,
            category_deltas=category_deltas or [],
            regressions=regressions or [],
            improvements=improvements or [],
            status=status,
        )

    return _make


@pytest.fixture
def make_golden_case():
    def _make(
        case_id, category="billing", summary="expected summary", text=None, difficulty="easy"
    ):
        return GoldenCase(
            id=case_id,
            input=text or f"EMAIL_{case_id}",
            expected_output=GoldenExpectedOutput(category=category, summary=summary),
            expected_difficulty=difficulty,
            edge_case_tags=[],
            notes="fixture case",
        )

    return _make


@pytest.fixture
def prompt_config():
    return PromptConfig(
        version_id="test_v1",
        created_at="2026-01-01",
        model="gpt-4o-mini",
        system_prompt="Classify the email.",
        few_shot_examples=[],
    )
