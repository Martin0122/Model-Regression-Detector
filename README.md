# Model Regression Detection System

CI for prompt changes. Every PR that touches `/prompts` runs the customer-support email
classifier against a 50-case hand-labeled golden dataset, scores it on category accuracy and
summary quality, diffs the result against the previous run, and posts the outcome to the PR and
to Slack. A separate rolling-average check watches for slow quality decay that no single run
would trigger on its own. The goal is to catch a bad prompt change before it reaches users, the
same way a test suite catches a bad code change.

This file is the operational reference. For why the system is built this way — the problem it
solves and the reasoning behind the main design decisions — see **[WRITEUP.md](WRITEUP.md)**.

## Architecture

```
prompts/*.yaml          -> versioned prompt configs (the "code" under test)
datasets/golden_dataset_v1.json -> hand-labeled ground truth (50 cases); the ONE authoritative dataset
datasets/validate_golden_dataset.py -> strict schema validator, runs in CI against that same file
src/models.py           -> data schemas (prompt, dataset, run, comparison)
src/settings.py         -> env-driven thresholds and gates
src/loaders.py          -> loads and validates the prompt and the dataset
src/formatting.py       -> shared phrasing for console, PR summary and Slack
src/classifier.py       -> the LLM feature under test
src/eval_runner.py      -> async batch runner, produces RawResult per case
src/scoring.py          -> category exact-match + LLM-as-judge summary score -> ScoredResult
src/comparer.py         -> diffs two runs: pass-rate delta, per-category delta, flipped cases
src/drift.py            -> rolling N-run moving average, catches gradual decline
src/report.py           -> renders the HTML diff report
src/notifier.py         -> Slack webhook alerts
src/pipeline.py         -> ties the above together; the CI entrypoint
src/runs/*.json         -> every eval run, committed to git (this is the run history)
reports/                -> generated HTML reports + the PR comment body
```

### Why these decisions

- **Runs are committed JSON files, not a database.** The whole system needs is "what did the
  last run look like," and a JSON file per run gives that for free, with zero infrastructure and
  a readable git history of quality over time. SQLite would add a moving part for no real
  benefit at this scale (dozens to low hundreds of runs).
- **Scoring has two independent dimensions, not one pass/fail.** Category is checked with an
  exact match because it's a closed set with a right answer. Summary quality is checked with an
  LLM-as-judge 1-5 score because "is this a good one-sentence summary" doesn't have a single
  correct string. A case only passes if both hold (`category_match and summary_score >= 4`).
  Collapsing these into one number would hide *which* dimension broke when a prompt change
  regresses.
- **Regression thresholds (3%/8%) are separate from drift thresholds.** Per-run diffing catches
  a prompt change that immediately makes things worse. It does not catch a slow bleed where each
  individual run only drops half a point but the trend over 7 runs is clearly down. Those are
  different failure modes with different causes (a bad prompt edit vs. e.g. upstream model
  drift), so they get separate, independently configurable thresholds instead of one blended
  metric.
- **The pipeline exits 1 on `critical`, not on `warning`.** A warning should be visible (PR
  comment + Slack) without blocking work; only a critical regression should actually gate the
  merge. This is enforced by making `critical` fail the CI job and relying on branch protection
  to require that check — see [Wiring up branch protection](#wiring-up-branch-protection) below.
- **Golden dataset cases are hand-written, not LLM-generated.** If the ground truth comes from
  the same kind of model being tested, the eval just measures self-agreement, not correctness.
  The dataset is deliberately the most manual, least automatable part of this system.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate.bat
pip install -r requirements.txt
```

Create a `.env` file in the repo root:

```bash
OPENAI_API_KEY=your_api_key
SLACK_WEBHOOK_URL=your_slack_incoming_webhook_url   # optional; alerts are skipped without it
```

Run a single eval + score + report cycle locally:

```bash
python -m src.pipeline
```

This runs the classifier against `datasets/golden_dataset_v1.json` using
`prompts/v6_classifier.yaml`, saves the result to `src/runs/`, and (once at least two runs
exist) writes an HTML report to `reports/`, posts a Slack alert if a webhook is configured,
and checks for drift.

Run the individual stages directly if you're debugging one piece:

```bash
python -m src.comparer   # diff the two most recent runs, print to stdout
python -m src.report     # diff + HTML report + Slack alert + drift check
python -m src.drift      # drift check only
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest                 # full suite, no API key needed - OpenAI is mocked
pytest -m live         # opt-in, hits the real API and costs money
```

OpenAI is mocked by default via fixtures in `tests/conftest.py`, so the suite runs offline and
in CI without credentials. Tests that hit the real API are marked `live` and deselected by
default (see `pytest.ini`).

## Adding new test cases to the golden dataset

Edit `datasets/golden_dataset_v1.json` and add an entry to `cases`. Each case needs:

- `id` — stable, unique (e.g. `tc_051`)
- `input` — realistic customer email, hand-written
- `expected_output.category` — one of `billing`, `technical`, `account`, `general`
- `expected_output.summary` — the ideal one-sentence summary
- `expected_difficulty` — `easy`, `medium`, `hard`, or `edge`
- `edge_case_tags` — optional, e.g. `ambiguous`, `typo`, `mixed_language`, `sarcastic`, `extremely_short`
- `notes` — why this case is in the dataset

`datasets/case_template.json` has a copy-pasteable skeleton. Validate before committing:

```bash
python datasets/validate_golden_dataset.py
```

CI runs that same validator against that same file, so validation and production can't drift
apart. The dataset is also parsed through a Pydantic model (`GoldenDataset` in `src/models.py`),
so a malformed dataset fails loudly at load time instead of surfacing as a `KeyError` deep in
the eval loop.

**Do not generate cases with an LLM.** The dataset's entire value is being an independent,
human-verified source of truth; if it's LLM-generated it just measures whether two models agree
with each other. When a bug or edge case slips through to production, the standard move is to
add it here as a new case so the regression suite catches it next time — the dataset should grow
from real failures, not just be front-loaded once.

## Adjusting thresholds

All thresholds are env vars with defaults in `src/settings.py`, so nothing needs to be
rebuilt to change them — set them in `.env` locally, as repo/environment secrets in CI, or as
`-e` flags to the Docker container.

| Variable | Default | Meaning |
|---|---|---|
| `REGRESSION_WARNING_THRESHOLD` | `0.03` | Per-run pass-rate drop that triggers a warning |
| `REGRESSION_CRITICAL_THRESHOLD` | `0.08` | Per-run pass-rate drop that triggers critical (blocks merge) |
| `DRIFT_WINDOW_SIZE` | `7` | Number of runs in the rolling average |
| `DRIFT_WARNING_THRESHOLD` | `0.03` | Rolling-average drop (vs. the earliest available window) that triggers a drift warning |
| `DRIFT_CRITICAL_THRESHOLD` | `0.08` | Rolling-average drop that triggers critical drift |
| `MIN_COMPLETION_RATE` | `0.95` | Fraction of the dataset that must complete, or the run aborts |
| `MIN_PASS_RATE` | `0.0` (off) | Absolute quality floor — a run below this is critical even with no regression |
| `BLOCK_ON_CRITICAL_DRIFT` | `false` | Whether critical drift fails the CI job (see below) |
| `JUDGE_MODEL` | `gpt-4o-mini` | Model used by the LLM-as-judge; recorded in run metadata |

### Why drift doesn't block merges by default

A critical per-run regression blocks the merge: the PR in front of you caused it, and the author
can fix it. Critical *drift* only alerts. Drift is a property of the trend, often originating
upstream (a silently updated provider model, gradual data shift) rather than in any one PR, so
blocking on it would stall every PR until the trend recovers, for something no individual author
can fix. Set `BLOCK_ON_CRITICAL_DRIFT=true` if your team wants the hard stop instead.

### Relative thresholds vs. the absolute floor

The regression and drift thresholds are all *relative* — they ask "did this change make things
worse". That leaves a real gap: a suite sitting at a poor pass rate reports `PASS` forever,
because nothing got worse this run. `MIN_PASS_RATE` closes it by failing any run whose absolute
pass rate is under the floor, regardless of the delta.

It ships disabled (`0.0`). Turning it on above the current pass rate fails every run immediately,
so set it once the suite is where you want it, and raise it as quality improves — a ratchet
rather than an aspiration.

### Testing a candidate prompt

Don't promote a prompt before the eval has judged it. Point the pipeline at the candidate and let
it be compared against the existing baseline:

```bash
python -m src.pipeline --prompt prompts/v4_candidate.yaml
```

If the comparison is favourable, promote it by updating `DEFAULT_PROMPT_PATH` in `src/scoring.py`.

### Incomplete runs

Cases dropped to API errors are infrastructure failures, not quality failures, so they're
excluded from results rather than scored as regressions — counting them as failures would fire
false alerts and block merges for unrelated reasons. To stop a badly degraded run from reporting
a confident-looking pass rate over a small surviving sample, a run that completes less than
`MIN_COMPLETION_RATE` of the dataset aborts outright (exit 1) and is **not saved**, since
persisting a partial run would skew future baselines and drift.

## Docker

```bash
docker build -t model-regression-detector .
docker run --rm \
  -e OPENAI_API_KEY=your_api_key \
  -e SLACK_WEBHOOK_URL=your_webhook_url \
  -v $(pwd)/src/runs:/app/src/runs \
  -v $(pwd)/reports:/app/reports \
  model-regression-detector
```

The volume mounts matter: `src/runs` is where run history lives. Without mounting it, every
container invocation starts from zero history and there's nothing to diff against.

## CI/CD

`.github/workflows/prompt-eval.yml` runs on every PR touching `prompts/**`, `datasets/**`,
`src/**`, `tests/**`, or the requirements files. It has two jobs:

**`checks`** — free, no API key, runs first:
1. Validates the golden dataset (`datasets/validate_golden_dataset.py`).
2. Runs the unit tests (`pytest`).

**`eval`** — needs `checks` to pass, since it spends real money (one classifier call plus one
judge call per case):
1. Seeds a fallback summary so an early crash still produces a readable PR comment.
2. Runs `python -m src.pipeline` (eval -> score -> save -> compare -> report -> alert -> drift).
3. Posts/updates a single PR comment with the summary.
4. Uploads the HTML report and summary as workflow artifacts.
5. Exits non-zero on a `critical` comparison (and on critical drift if `BLOCK_ON_CRITICAL_DRIFT`
   is enabled).

Required repo secrets: `OPENAI_API_KEY`, and optionally `SLACK_WEBHOOK_URL`. Optional repo
variable: `BLOCK_ON_CRITICAL_DRIFT`.

### Wiring up branch protection

A failing job doesn't block a merge by itself — that's a separate GitHub setting. To actually
enforce it: repo Settings -> Branches -> branch protection rule on `main` -> Require status
checks to pass -> select the `checks` and `eval` jobs. Without this, the workflow still reports
status correctly, it just doesn't stop anyone from merging past it.

### A known limitation worth knowing about

Runs generated during a CI job aren't committed back to the repo automatically — GitHub Actions
runners are ephemeral, and the workflow doesn't push to the branch. So each PR's CI run compares
against whatever run was last *committed* to `src/runs/`, not against other in-flight PRs. Run
history only grows when someone runs `python -m src.pipeline` locally (or in a scheduled job) and
commits the result. Auto-committing from CI was left out deliberately: it's easy to get subtly
wrong (race conditions between concurrent PRs, noisy bot commits) for a benefit that doesn't
matter much at this scale.

## Operator setup

`docs/SETUP_CHECKLIST.md` walks through everything that has to be configured outside the
codebase — building up run history, GitHub Actions secrets, branch protection (without which a
failing job does not actually block a merge), Docker verification, and the optional Slack
webhook. Start there when standing this up on a new repo or machine.

## Demo

`docs/DEMO_SCRIPT.md` is a timed shot list for recording a 3-minute walkthrough: make a prompt
change, watch the regression get caught, show the Slack alert and the diff report. It includes
the pre-recording setup, since a real 50-case run takes longer than the recording budget.

## Run history persistence

`src/runs/*.json` is the regression baseline and **is committed to git** — a comparison is only
meaningful against a run someone kept. This is why generated reports are gitignored but runs are
not. Two consequences worth knowing:

- A new clone with fewer than 2 runs can't compare; the pipeline says so and exits 0 rather than
  failing (`InsufficientRunHistory`).
- Drift needs `DRIFT_WINDOW_SIZE + 1` runs (8 by default) before it reports anything.

In Docker, mount a volume over `/app/src/runs` or every container start begins with no history.
