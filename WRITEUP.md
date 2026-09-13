# CI for prompts: catching LLM regressions before users do

## The problem

We would never merge a refactor without running the test suite. We merge prompt rewrites after
spot-checking two examples in a playground.

That asymmetry is the whole problem. A prompt is code — it has inputs, outputs, and regressions —
but it has no CI. Someone tweaks a system prompt to fix a misclassified billing email, ships it,
and quietly breaks five account cases that used to work. Nobody finds out from a dashboard. They
find out from a support ticket three weeks later, and by then nobody remembers which change did it.

The failure is invisible because it's *partial*. The feature doesn't crash. It doesn't 500. It
just gets a bit worse, on a subset of inputs nobody happened to check.

## What I built

A CI/CD pipeline for model behavior. On every PR that touches a prompt, the dataset, or the eval
code, it runs a customer-support email classifier against 50 hand-labeled cases, scores each one
on four dimensions, diffs the result against the last accepted run, comments the summary on the
PR, alerts Slack, and fails the job when quality drops past a critical threshold.

Two decisions shaped everything else.

**The golden dataset is hand-written and never LLM-generated.** Fifty real-looking emails across
billing, technical, account, and general, each with a human-written ideal summary and a note
explaining why the case is in the set. Thirty-six carry edge-case tags: ambiguous between two
categories, mixed-language, sarcastic, typo-ridden, or three words long. This is slow, and it is
the point. If the ground truth comes from the same class of model you're testing, you aren't
measuring correctness — you're measuring self-agreement between two models that share the same
blind spots.

**Scoring is multi-dimensional.** Exact category match is binary and easy. Summary quality isn't,
so an LLM-as-judge rates relevance 1–5, and a case passes only if the category matches *and* the
summary scores ≥ 4. Latency and token usage are recorded per case too. Collapsing this into one
number would tell you quality dropped without telling you which dimension broke — and those have
completely different fixes.

## The decision I'm proudest of: drift doesn't block merges

The system tracks two things that both look like "quality went down," and treats them as
different problems with different consequences.

A **per-run regression** is sharp and attributable: this PR's pass rate dropped 8% against the
previous run, and here are the six specific cases that flipped from pass to fail. A **slow drift**
is the 7-run moving average sagging while no individual run ever crosses the per-run threshold —
half a point at a time, which is exactly the shape of an upstream model being silently updated, or
real-world inputs shifting away from a dataset written eight months ago.

The naive design is one threshold on one number, and it fails in both directions. Set it tight
enough to catch a slow bleed and every PR trips it on noise. Set it loose enough to be quiet and
the bleed is invisible for months. So there are two detectors with independently configurable
thresholds.

The part I actually care about is that **only one of them blocks the merge.**

A critical per-run regression fails the CI job. The person who opened the PR caused it, can see
which cases broke, and can fix it before merging. Blocking is appropriate because blocking is
actionable.

Critical drift alerts loudly and lets the merge through. Nobody can fix a sagging 7-run average
from a feature branch. The cause is usually outside the PR entirely — a provider updating a model
under a stable version string, or a dataset that's aged out of its distribution. Blocking on it
would stall every PR in the repo until the trend recovered, punishing whoever happened to open a PR
that morning for something they didn't do and can't undo. So it pages the team instead, and the
owner decides whether to regenerate the baseline, refresh the dataset, or pin a model version.
(`BLOCK_ON_CRITICAL_DRIFT=true` turns on the hard stop for teams that want it.)

The general principle: **gating should map to who can act, not just to how bad the number is.**
Severity and actionability are different axes, and conflating them is how you end up with a
CI check everyone has learned to click past.

The same reasoning governs API failures. When a case times out, it's dropped rather than scored as
a failure — scoring it as a failure would fire a regression alert and block a merge over a network
blip that says nothing about prompt quality. But silently dropping is also wrong, because then a
run where 38 of 50 cases errored out happily reports "96% pass rate" over the survivors. So a
completion-rate gate aborts any run that finishes under 95% of the dataset, and deliberately does
not save it — persisting a partial run would poison every future baseline and drift window. Again:
separate "the system broke" from "the output got worse," because they demand different responses.

## What testing the thing taught me

Building it was the easy half. Two findings from testing changed how I think about eval tooling.

**The judge was grading against contradictory instructions.** The prompt said scores ran from "1
being the most relevant and 5 being the least relevant," then two sentences later said "a 1
signifies the worst score." The pass criterion assumed higher-is-better. Every historical score was
produced under instructions that contradicted themselves — and I had been treating those runs as a
baseline. Run metadata now records a `judge_prompt_version`, and comparing across versions emits a
warning, because a changed judge moves scores on its own and a delta across that boundary conflates
"the feature got worse" with "I changed the ruler."

**The safety system was failing open.** Pydantic's `ValidationError` subclasses `ValueError`. A
broad `except ValueError`, written to handle the ordinary "no previous run to compare against"
case, was also swallowing corrupt-run-file crashes — reporting them as "first run, nothing to
compare," and exiting 0. A single truncated JSON file would have disabled the regression gate
entirely while CI stayed green.

That's the worst possible failure mode for this kind of tool, and it's worse than having no tool at
all, because a green check you trust is more dangerous than no check. Both control-flow exceptions
are now dedicated types that deliberately do not inherit from `ValueError`, and a test asserts that
property so it can't quietly come back.

---

*Setup, architecture, and operational docs are in the [README](README.md).*
