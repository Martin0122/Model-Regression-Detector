# Golden Dataset V1

`golden_dataset_v1.json` is the authoritative golden dataset. It is the single file the eval
pipeline runs against and the single file CI validates — there is no second copy.

These 50 cases are hand-written and human-verified. Keep them that way: do not use an LLM to
generate inputs, labels, or ideal summaries. If the ground truth comes from the same kind of
model being tested, the eval measures self-agreement rather than correctness.

## Case schema

- `id` — stable unique ID (e.g. `tc_051`); lowercase letters, digits, `_`, `-`
- `input` — the full customer email text
- `expected_output.category` — one of `billing`, `technical`, `account`, `general`
- `expected_output.summary` — ideal one-sentence summary
- `expected_difficulty` — one of `easy`, `medium`, `hard`, `edge`
- `edge_case_tags` — optional list: `ambiguous`, `typo`, `mixed_language`, `sarcastic`, `extremely_short`
- `notes` — why this case matters

`case_template.json` is a copy-pasteable skeleton.

## Current composition

50 cases: 17 billing, 13 technical, 12 account, 8 general. 16 easy, 20 medium, 14 hard.
36 carry at least one `edge_case_tags` entry.

## Validation

```bash
python datasets/validate_golden_dataset.py
```

Defaults to this file, which is the same file the pipeline consumes — validation and production
cannot drift apart. CI runs this on every PR touching `datasets/`, `src/`, `prompts/`, or
`tests/`. `--allow-draft` relaxes the minimum-count and coverage rules while a new dataset
version is still being written.

The pipeline additionally parses this file through the `GoldenDataset` Pydantic model in
`src/models.py`, so a schema violation fails at load time rather than mid-eval.

## Growing the dataset

The intended way this grows is from real failures: when a bad output reaches users, add it here
as a case so the suite catches that class of failure from then on. Front-loading cases once and
never adding to them is how an eval suite goes stale.
