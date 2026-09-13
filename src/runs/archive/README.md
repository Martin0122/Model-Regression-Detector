# Archived runs

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
