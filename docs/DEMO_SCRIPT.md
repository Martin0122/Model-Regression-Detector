# 3-minute walkthrough script

A shot-by-shot script for the Loom. The hard constraint is that a real 50-case run takes 1–2
minutes of API calls — most of your budget — so the run that produces the regression happens
**before** you hit record. You demo the *diff*, not the waiting.

## Before recording

1. **Have a baseline on disk.** With a real `OPENAI_API_KEY` set:
   ```bash
   python -m src.pipeline      # run 1 - reports "first run", nothing to compare
   python -m src.pipeline      # run 2 - first real comparison, should be PASS
   ```
2. **Set a Slack webhook** (`SLACK_WEBHOOK_URL` in `.env`) and confirm one alert lands, so the
   channel isn't empty on camera.
3. **Pre-stage the degraded prompt edit** but don't apply it yet. In
   `prompts/v6_classifier.yaml`, the edit is appending one sentence to `system_prompt`:
   > `When in doubt, classify the email as general.`

   That reliably drags the ambiguous and edge-tagged cases into `general` and produces a real
   regression rather than a staged one.
4. **Have three tabs ready:** terminal, the Slack channel, and a file browser pointed at
   `reports/`.
5. Close anything with credentials visible. `.env` stays off screen.

## Shot list

### 0:00–0:25 — The problem

Terminal, repo open.

> "Every team I've worked on ships prompt changes blind. We'd never merge a refactor without
> running tests, but we merge prompt rewrites after eyeballing two examples. This is CI for
> prompts — it runs an LLM feature against a hand-labeled golden dataset on every change and
> blocks the merge if quality drops."

### 0:25–0:50 — The golden dataset

Open `datasets/golden_dataset_v1.json`, scroll through two or three cases. Land on one with
`edge_case_tags`.

> "Fifty cases, hand-written, never LLM-generated — if the ground truth comes from the same kind
> of model you're testing, you're measuring self-agreement, not correctness. Thirty-six are
> deliberate edge cases: ambiguous between two categories, sarcastic, mixed-language, three words
> long. Each one has a note explaining why it's in the set."

### 0:50–1:20 — Make the change and kick off the run

Apply the one-line prompt edit on camera. Save. Then:

```bash
python -m src.pipeline
```

Let it scroll for ~10 seconds so the per-case output is visible, then cut.

> "I'm making the kind of change that looks harmless in review — one extra instruction about what
> to do when the model's unsure. The pipeline runs all fifty cases concurrently, scores each on
> category match plus an LLM-as-judge rating of the summary, and saves the run."

**Cut the wait in the edit.** Resume on the final output.

### 1:20–1:50 — The regression is caught

Show the terminal tail: status line, the regressed case IDs, the exit code.

> "Pass rate dropped, and it's flagged critical — over the 8% threshold. It's not just telling me
> the number moved, it's naming the exact cases that flipped from pass to fail. Those are mostly
> the ambiguous ones, which is exactly what that instruction would break. The job exits non-zero,
> so with branch protection on, this PR can't merge."

### 1:50–2:15 — The Slack alert

Switch to Slack.

> "Same result lands in the team channel — status, the headline numbers, which cases broke, and a
> link to the full report. This is the part that matters after deployment: you don't have to be
> watching a dashboard to find out."

### 2:15–2:50 — The diff report

Open the newest `reports/report_*.html` in the browser. Scroll: scorecard → per-category table →
the regressed-case table → trend chart.

> "The HTML report shows the scorecard against the baseline, accuracy broken out per category —
> so you can see *which* category took the hit — and then every regressed case with the old output
> next to the new one, side by side. That's usually enough to see the cause without rerunning
> anything. And the trend chart is the other half of the system: separately from per-run diffs, it
> tracks a 7-run moving average to catch slow drift that never trips a single-run threshold."

### 2:50–3:00 — Close

> "Drift alerts but doesn't block, because nobody can fix a sagging average from a feature branch —
> regressions block, because the author can. Full write-up's in the repo."

## After recording

```bash
git checkout prompts/v6_classifier.yaml
```

Then decide what to do with the degraded run in `src/runs/` — delete it, or keep it as the
"before" side of a future demo. Don't leave it as the live baseline.

## Notes

- Rehearse once. The temptation is to explain the architecture; don't. Show the regression being
  caught and let the artifacts speak.
- If a take runs long, the dataset section (0:25–0:50) compresses best — the report walkthrough is
  the part that actually differentiates the project.
- Mention the drift/regression distinction even if you cut everything else. It's the design
  decision interviewers follow up on.
