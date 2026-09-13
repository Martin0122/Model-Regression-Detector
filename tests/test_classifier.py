"""Classifier message construction (mocked) plus opt-in live API checks.

The live tests are marked `live` and deselected by default - run them with `pytest -m live`
and a real OPENAI_API_KEY. Everything else runs offline.
"""
import pytest

from src.classifier import build_messages, classify_email
from src.config import ClassificationOutput, FewShotExample, PromptConfig
from src.services.load_configs import load_prompt_config


# --------------------------------------------------------------------------------------
# Message construction (offline)
# --------------------------------------------------------------------------------------

def test_system_prompt_is_first_and_email_is_last(prompt_config):
    messages = build_messages("the customer email", prompt_config)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == prompt_config.system_prompt
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "the customer email"


def test_few_shot_examples_expand_to_user_assistant_pairs():
    config = PromptConfig(
        version_id="v", created_at="2026-01-01", model="gpt-4o-mini", system_prompt="sys",
        few_shot_examples=[
            FewShotExample(input="ex1", output=ClassificationOutput(category="billing", summary="s1")),
            FewShotExample(input="ex2", output=ClassificationOutput(category="technical", summary="s2")),
        ],
    )
    messages = build_messages("real email", config)

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert messages[1]["content"] == "ex1"
    assert "billing" in messages[2]["content"]
    assert messages[-1]["content"] == "real email"


async def test_classify_email_returns_parsed_fields_and_token_usage(fake_classifier_client, prompt_config):
    client = fake_classifier_client(lambda email: ("technical", "a summary"))
    result = await classify_email("my app crashes", prompt_config, client)
    assert result["category"] == "technical"
    assert result["summary"] == "a summary"
    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 20


# --------------------------------------------------------------------------------------
# Prompt config loading (offline)
# --------------------------------------------------------------------------------------

def test_live_prompt_file_loads_and_is_well_formed():
    config = load_prompt_config("./prompts/v1_classifier.yaml")
    assert config.version_id
    assert config.model
    assert config.system_prompt.strip()
    for example in config.few_shot_examples:
        assert example.output.category in ("billing", "technical", "account", "general")


def test_missing_prompt_file_raises_filenotfound():
    with pytest.raises(FileNotFoundError):
        load_prompt_config("./prompts/does_not_exist.yaml")


# --------------------------------------------------------------------------------------
# Live API (opt-in: pytest -m live)
# --------------------------------------------------------------------------------------

LIVE_CASES = [
    ("I was charged twice this month, please refund the duplicate.", "billing"),
    ("The app crashes every time I upload a photo larger than 5MB.", "technical"),
    ("I can't log in anymore, it says my email isn't recognized.", "account"),
    ("Do you offer student discounts? Just curious before I sign up.", "general"),
]


@pytest.mark.live
@pytest.mark.parametrize("email,expected_category", LIVE_CASES)
async def test_live_classification_matches_expected_category(email, expected_category):
    config = load_prompt_config("./prompts/v1_classifier.yaml")
    result = await classify_email(email, config)
    assert result["category"] == expected_category
    assert result["summary"].strip()
