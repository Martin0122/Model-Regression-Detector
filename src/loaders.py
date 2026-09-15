"""Loading and validating the two files the pipeline reads: the prompt and the dataset.

Both parse through pydantic models so a malformed file fails at load time rather than as a
KeyError somewhere inside the eval loop.
"""

from pathlib import Path

import yaml

from src.models import GoldenDataset, PromptConfig


def load_prompt_config(path: str | Path) -> PromptConfig:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Prompt configuration file not found: {source}")
    return PromptConfig(**yaml.safe_load(source.read_text()))


def load_golden_dataset(path: str | Path) -> GoldenDataset:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Golden dataset not found: {source}")
    return GoldenDataset.model_validate_json(source.read_text())
