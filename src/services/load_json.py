from pathlib import Path

from src.config import GoldenDataset


def load_golden_dataset(path: str) -> GoldenDataset:
    """Loads and validates the golden dataset. Raises pydantic.ValidationError if the file
    doesn't match the authoritative schema, rather than failing later with a KeyError."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Golden dataset not found: {source}")
    return GoldenDataset.model_validate_json(source.read_text())
