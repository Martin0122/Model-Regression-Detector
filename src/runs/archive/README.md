# Archived runs

## Superseded prompt versions and unpinned sampling (2026-09-15)

Eight further runs were archived because they cannot serve as valid baselines:

- **Four v1-prompt runs** (66–74% pass). Superseded by v2 and then v3; a comparison across a
  prompt change measures the change, not quality.
- **Four v2-prompt runs recorded before `temperature` was pinned** (92%, 92%, 92%, 86%). The
  judge was sampling at the API default of 1.0, so it re-rolled its score on every call. Across
  those four identical runs, 12 of 50 cases flipped pass/fail purely on that — the 92% readings
  were partly lucky rolls, and the same prompt measured 80–84% once sampling was pinned to 0.

Runs from this point record `judge_temperature` and `classifier_temperature` in their metadata,
so this distinction is visible rather than inferred from timestamps.

The two kept runs are v2 at temperature 0. They stop counting toward drift automatically once
v3 runs accumulate, because drift only considers the trailing block sharing the newest run's
prompt version.

---


These runs were scored under **judge prompt v1**, which was self-contradictory: it told the
judge "1 being the most relevant and 5 being the least relevant" in one sentence and "a 1
signifies the worst score ... 5 means the two summaries are extremely relevant" in the next,
while the pass criterion (`summary_score >= 4`) assumed higher-is-better.

Because the judge instructions changed in v2, scores in these runs are not comparable with
current runs — a delta across that boundary would mix "the feature changed" with "the ruler
changed". They are kept for history, not as baselines.

`list_runs()` globs `src/runs/run_*.json` non-recursively, so nothing in this directory is
picked up as a baseline.

Runs written from v2 onward record `judge_prompt_version` in their metadata, and `compare_runs()`
warns when a baseline and current run disagree on it.

## Re-establishing a baseline

The first run after archiving has nothing to compare against — the pipeline reports "first run"
and exits 0. The run after that produces the first real comparison.
