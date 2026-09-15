import os
from typing import cast

from openai.types.responses import ResponseInputParam

from .models import ClassificationOutput, PromptConfig

CLASSIFIER_TEMPERATURE = float(os.getenv("CLASSIFIER_TEMPERATURE", 0.0))


def build_messages(email: str, config: PromptConfig) -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": config.system_prompt,
        }
    ]

    for example in config.few_shot_examples:
        messages.append(
            {
                "role": "user",
                "content": example.input,
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": example.output.model_dump_json(),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": email,
        }
    )

    return messages


async def classify_email(
    email: str,
    config: PromptConfig,
    llm_client=None,
) -> dict:
    if llm_client is None:
        from .clients.llm_client import get_client

        llm_client = get_client()

    # Pinned so a run measures the prompt, not the sampler.
    response = await llm_client.responses.parse(
        model=config.model,
        input=cast(ResponseInputParam, build_messages(email, config)),
        text_format=ClassificationOutput,
        temperature=CLASSIFIER_TEMPERATURE,
    )

    if not response.output_parsed:
        raise ValueError("Model did not return valid classification output.")

    input_tokens = response.usage.input_tokens if response.usage else -1
    output_tokens = response.usage.output_tokens if response.usage else -1
    category = response.output_parsed.category
    summary = response.output_parsed.summary

    result = {
        "category": category,
        "summary": summary,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }

    return result
