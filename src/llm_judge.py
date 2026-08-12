from .clients.llm_client import client
from pydantic import BaseModel, Field

class JudgeScore(BaseModel):
    score: int = Field(ge=1, le=5, description="Relevance score from 1-5")

async def llm_as_judge(generated_summary: str, actual_summary: str) -> int:
    """Takes in two summaries and returns a score of 1-5 based on how closely related they are."""

    prompt = f"""
    You are a judge that will be given two email summaries that you will eventually use to rate with a score from 1-5 (inclusive), 1 being the most relevant
    and 5 being the least relevant. The first summary is the generated summary of an LLM feature that generated a summary of an email, and the 
    second summary is an ideal summary of the exact email. You will assign a score from 1-5 in which a 1 signifies the worst score, which means that
    the first summary (generated summary) is not relevant whatsover to the second summary (ideal summary). A score of 5 means the two summaries are extremely
    relevant to one another. Only return the score.

    <generated-summary>
    {generated_summary}
    <generated-summary>

    <actual-summary>
    {actual_summary}
    <actual-summary>
    """

    response = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format=JudgeScore
    )

    message = response.choices[0].message

    if message.refusal:
        raise ValueError(f"Model refused to respond: {message.refusal}")

    if message.parsed is None:
        raise ValueError("Failed to parse structured output from model response")

    result = message.parsed.score
    
    return result