"""The golden dataset is ground truth - guard its integrity in CI."""

import subprocess
import sys
from collections import Counter

import pytest

from src.loaders import load_golden_dataset
from src.scoring import DEFAULT_DATASET_PATH


@pytest.fixture(scope="module")
def dataset():
    return load_golden_dataset(DEFAULT_DATASET_PATH)


def test_dataset_parses_against_the_authoritative_schema(dataset):
    assert dataset.cases


def test_case_count_is_within_target_range(dataset):
    assert 50 <= len(dataset.cases) <= 100


def test_case_ids_are_unique(dataset):
    ids = [case.id for case in dataset.cases]
    duplicates = [i for i, count in Counter(ids).items() if count > 1]
    assert duplicates == []


def test_every_category_is_covered(dataset):
    covered = {case.expected_output.category for case in dataset.cases}
    assert covered == set(dataset.expected_categories)


def test_difficulties_are_from_the_declared_set(dataset):
    for case in dataset.cases:
        assert case.expected_difficulty in dataset.expected_difficulties


def test_required_text_fields_are_non_empty(dataset):
    for case in dataset.cases:
        assert case.input.strip(), f"{case.id} has empty input"
        assert case.expected_output.summary.strip(), f"{case.id} has empty summary"
        assert case.notes.strip(), f"{case.id} has empty notes"


def test_inputs_are_not_duplicated(dataset):
    inputs = [case.input.strip() for case in dataset.cases]
    duplicates = [i for i, count in Counter(inputs).items() if count > 1]
    assert duplicates == []


def test_dataset_contains_deliberate_edge_cases(dataset):
    tagged = [
        case for case in dataset.cases if case.edge_case_tags or case.expected_difficulty == "edge"
    ]
    assert len(tagged) >= 8


def test_strict_validator_passes_on_the_production_dataset():
    """The validator and the eval pipeline must agree on one file - run it the way CI does."""
    result = subprocess.run(
        [sys.executable, "datasets/validate_golden_dataset.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"validator failed:\n{result.stdout}\n{result.stderr}"
