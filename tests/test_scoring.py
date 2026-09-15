"""Eval running, judging, id-based pairing, and the completion-rate gate. All mocked."""

import json

import pytest

import src.eval_runner as eval_runner_mod
import src.scoring as scoring_mod
from src.clients.llm_client import MISSING_KEY_MESSAGE, get_client
from src.errors import IncompleteEvalRun
from src.llm_judge import llm_as_judge
from src.settings import CompletionConfig


def write_dataset(tmp_path, n_cases, categories=None):
    categories = categories or ["billing"] * n_cases
    dataset = {
        "dataset_version": "test_v1",
        "created_at": "2026-01-01",
        "feature": "test",
        "status": "active",
        "source_policy": "human written",
        "expected_categories": ["billing", "technical", "account", "general"],
        "expected_difficulties": ["easy", "medium", "hard", "edge"],
        "target_case_count": {"minimum": 1, "maximum": 100},
        "cases": [
            {
                "id": f"t{i}",
                "input": f"EMAIL_{i}",
                "expected_output": {"category": categories[i], "summary": f"summary {i}"},
                "expected_difficulty": "easy",
                "edge_case_tags": [],
                "notes": "test case",
            }
            for i in range(n_cases)
        ],
    }
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(dataset))
    return str(path)


@pytest.fixture
def patched_clients(monkeypatch, fake_classifier_client, fake_judge_client, prompt_config):
    """Builds the fake clients and returns them for explicit injection. Only the prompt-config
    loader is patched, since the prompt path is not a parameter of the call under test."""

    def _apply(answer_fn=None, score_fn=None):
        monkeypatch.setattr(scoring_mod, "load_prompt_config", lambda path: prompt_config)
        return {
            "llm_client": fake_classifier_client(answer_fn),
            "judge_client": fake_judge_client(score_fn),
        }

    return _apply


# --------------------------------------------------------------------------------------
# eval_runner resilience
# --------------------------------------------------------------------------------------


async def test_results_preserve_dataset_order(patched_clients, prompt_config, tmp_path):
    clients = patched_clients(answer_fn=lambda email: ("billing", f"summary for {email}"))
    path = write_dataset(tmp_path, 3)
    results = await eval_runner_mod.eval_runner(
        path, prompt_config, llm_client=clients["llm_client"]
    )
    assert [r.test_id for r in results] == ["t0", "t1", "t2"]


async def test_one_failing_case_does_not_abort_the_batch(patched_clients, prompt_config, tmp_path):
    """Regression test: asyncio.gather without return_exceptions used to lose all 50 results
    when a single case hit a transient timeout."""

    def answer(email):
        if "EMAIL_1" in email:
            raise TimeoutError("simulated transient API failure")
        return ("billing", "ok")

    clients = patched_clients(answer_fn=answer)
    path = write_dataset(tmp_path, 3)
    results = await eval_runner_mod.eval_runner(
        path, prompt_config, llm_client=clients["llm_client"]
    )
    assert {r.test_id for r in results} == {"t0", "t2"}


# --------------------------------------------------------------------------------------
# id-based pairing
# --------------------------------------------------------------------------------------


async def test_scoring_pairs_by_id_not_position(patched_clients, tmp_path):
    """Worst case for positional pairing: the FIRST case fails, shifting every later index.
    Results must still be scored against their own golden case."""

    def answer(email):
        if "EMAIL_0" in email:
            raise TimeoutError("first case fails")
        return ("technical" if "EMAIL_1" in email else "account", "generated")

    clients = patched_clients(answer_fn=answer)
    path = write_dataset(tmp_path, 3, categories=["billing", "technical", "account"])

    results = await scoring_mod.scorer(
        "prompt.yaml", path, completion=CompletionConfig(minimum_completion_rate=0.0), **clients
    )
    by_id = {r.test_case_id: r for r in results}

    assert "t0" not in by_id
    assert by_id["t1"].generated_category == "technical" and by_id["t1"].category_match
    assert by_id["t2"].generated_category == "account" and by_id["t2"].category_match


async def test_judge_failure_skips_only_that_case(patched_clients, tmp_path):
    def score(prompt):
        if "summary 1" in prompt:
            raise RuntimeError("judge unavailable")
        return 5

    clients = patched_clients(score_fn=score)
    path = write_dataset(tmp_path, 3)
    results = await scoring_mod.scorer(
        "prompt.yaml", path, completion=CompletionConfig(minimum_completion_rate=0.0), **clients
    )
    assert {r.test_case_id for r in results} == {"t0", "t2"}


# --------------------------------------------------------------------------------------
# pass criteria
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score,category,expected_pass",
    [
        (5, "billing", True),
        (4, "billing", True),
        (3, "billing", False),
        (5, "technical", False),
    ],
)
async def test_pass_requires_category_match_and_score_at_least_4(
    patched_clients, tmp_path, score, category, expected_pass
):
    clients = patched_clients(
        answer_fn=lambda email: (category, "generated"), score_fn=lambda prompt: score
    )
    path = write_dataset(tmp_path, 1, categories=["billing"])
    results = await scoring_mod.scorer(
        "prompt.yaml", path, completion=CompletionConfig(minimum_completion_rate=0.0), **clients
    )
    assert results[0].passed is expected_pass


# --------------------------------------------------------------------------------------
# completion-rate gate
# --------------------------------------------------------------------------------------


async def test_gate_aborts_when_too_many_cases_are_skipped(patched_clients, tmp_path):
    def answer(email):
        if any(f"EMAIL_{i}" in email for i in (0, 1, 2, 3)):
            raise TimeoutError("simulated")
        return ("billing", "ok")

    clients = patched_clients(answer_fn=answer)
    path = write_dataset(tmp_path, 10)
    with pytest.raises(IncompleteEvalRun, match="below the minimum"):
        await scoring_mod.scorer(
            "prompt.yaml",
            path,
            completion=CompletionConfig(minimum_completion_rate=0.95),
            **clients,
        )


async def test_gate_passes_when_completion_is_above_threshold(patched_clients, tmp_path):
    clients = patched_clients()
    path = write_dataset(tmp_path, 10)
    results = await scoring_mod.scorer(
        "prompt.yaml", path, completion=CompletionConfig(minimum_completion_rate=0.95), **clients
    )
    assert len(results) == 10


# --------------------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------------------


async def test_judge_is_called_with_temperature_zero(fake_judge_client):
    """The judge is the measuring instrument. At the API default of 1.0 it re-rolls its score
    each call, and since passing hinges on `summary_score >= 4`, any summary it genuinely rates
    near that boundary becomes a coin flip. Measured across four identical runs, 12 of 50 cases
    flipped pass/fail on nothing else - enough to trip the regression thresholds by itself."""
    client = fake_judge_client(lambda prompt: 5)
    await llm_as_judge("generated", "reference", judge_client=client)
    assert client.completions.last_kwargs.get("temperature") == 0.0


async def test_classifier_is_called_with_temperature_zero(fake_classifier_client, prompt_config):
    from src.classifier import classify_email

    client = fake_classifier_client(lambda email: ("billing", "s"))
    await classify_email("an email", prompt_config, client)
    assert client.responses.last_kwargs.get("temperature") == 0.0


# --------------------------------------------------------------------------------------
# lazy client initialization
# --------------------------------------------------------------------------------------


def test_missing_api_key_raises_a_clear_error(monkeypatch):
    """Regression test: the client used to be constructed at import time, so merely importing
    a pipeline module crashed without a key - which is why the suite needed a dummy key."""
    get_client.cache_clear()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    try:
        with pytest.raises(RuntimeError, match=MISSING_KEY_MESSAGE):
            get_client()
    finally:
        get_client.cache_clear()


async def test_scorer_surfaces_missing_key_once_not_as_n_skipped_cases(
    monkeypatch, tmp_path, prompt_config
):
    """score_one() treats any exception as a skipped case, so the key check has to happen
    before the per-case loop or a missing key looks like 50 flaky cases."""
    get_client.cache_clear()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(scoring_mod, "load_prompt_config", lambda path: prompt_config)
    path = write_dataset(tmp_path, 3)
    try:
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
            await scoring_mod.scorer("prompt.yaml", path)
    finally:
        get_client.cache_clear()


# --------------------------------------------------------------------------------------
# judge prompt correctness
# --------------------------------------------------------------------------------------


async def test_judge_prompt_states_higher_is_better(fake_judge_client):
    """The original prompt said '1 being the most relevant and 5 being the least relevant',
    contradicting both its own next sentence and the summary_score >= 4 pass criterion."""
    captured = {}

    def score(prompt):
        captured["prompt"] = prompt
        return 5

    await llm_as_judge("generated", "reference", judge_client=fake_judge_client(score))

    prompt = captured["prompt"].lower()
    assert "higher score is better" in prompt
    assert "1 = worst" in prompt
    assert "5 = best" in prompt
    assert "1 being the most relevant" not in prompt
