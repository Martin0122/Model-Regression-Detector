import yaml
from src.config import PromptConfig
from pathlib import Path

def load_prompt_config(path: str | Path) -> PromptConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt configuration file not found: {path}")
    
    with open(path, "r") as f:
        raw_data = yaml.safe_load(f)

    return PromptConfig(**raw_data)