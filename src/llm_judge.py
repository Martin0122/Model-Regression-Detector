from .clients.llm_client import client

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

    response = await client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )
    result = int(response.output_text)
    if result < 0 or result > 5:
        raise ValueError(f"ERROR: Result not in 1-5 range, result: {result}")
    
    return result