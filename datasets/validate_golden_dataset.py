import argparse
import json
import re
import sys
from pathlib import Path

# The single authoritative dataset - the same file the eval pipeline consumes, so validation
# and production can never drift apart.
DEFAULT_DATASET_PATH = "datasets/golden_dataset_v1.json"

REQUIRED_TOP_LEVEL_FIELDS = {
    "dataset_version",
    "created_at",
    "feature",
    "status",
    "source_policy",
    "expected_categories",
    "expected_difficulties",
    "target_case_count",
    "cases",
}

REQUIRED_CASE_FIELDS = {
    "id",
    "input",
    "expected_output",
    "expected_difficulty",
    "notes",
}

REQUIRED_OUTPUT_FIELDS = {"category", "summary"}
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def add_error(errors: list[str], message: str) -> None:
    errors.append(f"ERROR: {message}")


def add_warning(warnings: list[str], message: str) -> None:
    warnings.append(f"WARNING: {message}")


def is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_dataset(path: Path, allow_draft: bool) -> int:
    errors: list[str] = []
    warnings: list[str] = []

    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    missing_top_level = REQUIRED_TOP_LEVEL_FIELDS - dataset.keys()
    if missing_top_level:
        add_error(errors, f"Missing top-level fields: {sorted(missing_top_level)}")

    categories = dataset.get("expected_categories", [])
    difficulties = dataset.get("expected_difficulties", [])
    cases = dataset.get("cases", [])

    if not isinstance(categories, list) or not categories:
        add_error(errors, "expected_categories must be a non-empty list.")
    if not isinstance(difficulties, list) or not difficulties:
        add_error(errors, "expected_difficulties must be a non-empty list.")
    if not isinstance(cases, list):
        add_error(errors, "cases must be a list.")
        cases = []

    target_count = dataset.get("target_case_count", {})
    min_cases = target_count.get("minimum", 50) if isinstance(target_count, dict) else 50
    max_cases = target_count.get("maximum", 100) if isinstance(target_count, dict) else 100

    if len(cases) < min_cases:
        message = f"Dataset has {len(cases)} cases; Phase 2 requires at least {min_cases}."
        if allow_draft:
            add_warning(warnings, message)
        else:
            add_error(errors, message)
    if len(cases) > max_cases:
        add_error(errors, f"Dataset has {len(cases)} cases; expected no more than {max_cases}.")

    seen_ids: set[str] = set()
    covered_categories: set[str] = set()
    has_edge_case = False

    for index, case in enumerate(cases, start=1):
        label = f"case #{index}"
        if not isinstance(case, dict):
            add_error(errors, f"{label} must be an object.")
            continue

        missing_case_fields = REQUIRED_CASE_FIELDS - case.keys()
        if missing_case_fields:
            add_error(errors, f"{label} missing fields: {sorted(missing_case_fields)}")

        case_id = case.get("id")
        if not is_non_empty_string(case_id):
            add_error(errors, f"{label} id must be a non-empty string.")
        elif not ID_PATTERN.match(case_id):
            add_error(
                errors,
                f"{case_id} has an unstable ID format. Use lowercase letters, numbers, "
                f"underscores, or hyphens.",
            )
        elif case_id in seen_ids:
            add_error(errors, f"Duplicate case id: {case_id}")
        else:
            seen_ids.add(case_id)
            label = case_id

        if not is_non_empty_string(case.get("input")):
            add_error(errors, f"{label} input must be a non-empty string.")

        output = case.get("expected_output")
        if not isinstance(output, dict):
            add_error(errors, f"{label} expected_output must be an object.")
            output = {}

        missing_output_fields = REQUIRED_OUTPUT_FIELDS - output.keys()
        if missing_output_fields:
            add_error(
                errors, f"{label} expected_output missing fields: {sorted(missing_output_fields)}"
            )

        category = output.get("category")
        if category not in categories:
            add_error(errors, f"{label} category must be one of {categories}; got {category!r}.")
        else:
            covered_categories.add(category)

        summary = output.get("summary")
        if not is_non_empty_string(summary):
            add_error(errors, f"{label} summary must be a non-empty string.")

        difficulty = case.get("expected_difficulty")
        if difficulty not in difficulties:
            add_error(
                errors,
                f"{label} expected_difficulty must be one of {difficulties}; got {difficulty!r}.",
            )
        if difficulty == "edge":
            has_edge_case = True

        edge_case_tags = case.get("edge_case_tags", [])
        if edge_case_tags is None:
            edge_case_tags = []
        if not isinstance(edge_case_tags, list):
            add_error(errors, f"{label} edge_case_tags must be a list when present.")
        elif edge_case_tags:
            has_edge_case = True

        if not is_non_empty_string(case.get("notes")):
            add_error(errors, f"{label} notes must explain why the case matters.")

    missing_categories = set(categories) - covered_categories
    if cases and missing_categories:
        message = f"Dataset does not cover categories: {sorted(missing_categories)}"
        if allow_draft:
            add_warning(warnings, message)
        else:
            add_error(errors, message)

    if cases and not has_edge_case:
        message = (
            "Dataset has no deliberate edge cases. "
            "Add expected_difficulty='edge' or edge_case_tags."
        )
        if allow_draft:
            add_warning(warnings, message)
        else:
            add_error(errors, message)

    for warning in warnings:
        print(warning)
    for error in errors:
        print(error, file=sys.stderr)

    if errors:
        return 1

    print(f"Validated {len(cases)} cases from {path}.")
    if allow_draft:
        print("Draft mode allowed incomplete Phase 2 case count.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Phase 2 golden dataset JSON file.")
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_DATASET_PATH,
        help="Path to the golden dataset JSON file.",
    )
    parser.add_argument(
        "--allow-draft",
        action="store_true",
        help="Allow fewer than 50 cases while the human-curated dataset is still in progress.",
    )
    args = parser.parse_args()

    return validate_dataset(Path(args.path), args.allow_draft)


if __name__ == "__main__":
    raise SystemExit(main())
