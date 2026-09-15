import os

from pydantic import BaseModel, Field

from .clients.llm_client import get_client

# Single source of truth for the judge model: the same value is sent to the API and recorded
# in run metadata, so a recorded run can never claim a judge model that wasn't actually used.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

# Bump whenever the judge prompt changes in a way that could shift scores; recorded in run
# metadata so runs scored under different instructions are not silently compared.
# v1: self-contradictory (claimed 1=most relevant AND 1=worst). v2: 1=worst .. 5=best.
JUDGE_PROMPT_VERSION = "v2"

# The judge must not sample. At the API default of 1.0 it re-rolls each call, and since
# passing hinges on `summary_score >= 4`, anything near that boundary becomes a coin flip:
# 12 of 50 cases flipped across four identical runs before this was pinned.
JUDGE_TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", 0.0))


class JudgeScore(BaseModel):
    score: int = Field(ge=1, le=5, description="Relevance score from 1 (worst) to 5 (best)")


async def llm_as_judge(generated_summary: str, actual_summary: str, judge_client=None) -> int:
    """Takes in two summaries and returns a score of 1-5, where 1 is the worst match against
    the reference summary and 5 is the best.

    Pass judge_client to inject a double in tests; resolved lazily otherwise."""
    judge_client = judge_client or get_client()

    prompt = f"""
    You are grading how well a generated email summary matches an ideal reference summary.

    Score from 1 to 5 (inclusive), where a HIGHER score is BETTER:
      1 = worst: the generated summary is not relevant to the reference summary at all
      2 = poor: barely related, or misses the main point of the reference
      3 = fair: partially captures the reference, but omits or distorts important details
      4 = good: captures the main point accurately, with only minor differences in wording or detail
      5 = best: fully equivalent in meaning to the reference summary

    Return only the score.

    <generated-summary>
    {generated_summary}
    </generated-summary>

    <reference-summary>
    {actual_summary}
    </reference-summary>
    """

    response = await judge_client.beta.chat.completions.parse(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=JudgeScore,
        temperature=JUDGE_TEMPERATURE,
    )

    message = response.choices[0].message

    if message.refusal:
        raise ValueError(f"Model refused to respond: {message.refusal}")

    if message.parsed is None:
        raise ValueError("Failed to parse structured output from model response")

    result = message.parsed.score

    return result
