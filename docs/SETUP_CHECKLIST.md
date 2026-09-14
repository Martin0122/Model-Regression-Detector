# Operator setup checklist

Everything that has to be done outside the codebase to take this from "the code works" to
"the gate is actually enforcing." Work top to bottom — later steps depend on earlier ones.

Each step says what it unlocks and how to confirm it worked.

---

## 1. Build up run history

**Status:** 1 of 8 runs. Committed baseline: `run_2026-09-14T01-47-32` (68% pass, 94% category
accuracy, judge prompt v2).

Run history is the entire basis of comparison — with one run there is nothing to diff against.

### 1a. Second run → unlocks regression comparison

```bash
python -m src.pipeline
```

Costs ~100 API calls (50 classify + 50 judge). After this you get a real comparison, an HTML
report in `reports/`, and a Slack alert if configured.

Confirm:
```bash
python -m src.comparer     # should print a comparison, not "Need at least 2 valid runs"
```

Then commit it — runs are only baselines if they're in git:
```bash
git add src/runs/run_*.json
git commit -m "Add eval run <date>"
```

> **Only commit runs you accept as a baseline.** A run recorded while the prompt was
> deliberately broken (e.g. during the Loom demo) should be deleted, not committed — future
> comparisons and drift windows are measured against whatever is in `src/runs/`.

### 1b. Eight runs → unlocks drift detection

Drift needs `DRIFT_WINDOW_SIZE + 1` runs (7 + 1 = 8) before it can compare one full window
against another. Until then it prints "not enough run history" and exits 0 — expected, not a bug.

Confirm:
```bash
python -m src.drift        # should report a moving average, not "Not enough run history"
```

Don't burn 8 runs in one sitting just to switch the feature on. Drift is meant to measure change
over *time* — eight runs generated back-to-back against an unchanged prompt will show a flat line,
which tells you nothing. Let history accumulate naturally as you iterate, or schedule a nightly
run (see step 6).

---

## 2. Add `OPENAI_API_KEY` to GitHub Actions secrets

**Unlocks:** the `eval` job in CI. Without it that job fails on every PR.

1. Go to your repo on GitHub → **Settings** (repo settings, not account)
2. Left sidebar → **Secrets and variables** → **Actions**
3. **New repository secret**
4. Name: `OPENAI_API_KEY` — exactly this, it's case-sensitive and matches
   `.github/workflows/prompt-eval.yml`
5. Secret: paste the key
6. **Add secret**

Confirm: the secret is listed under *Repository secrets*. GitHub will never show the value again —
that's expected. If you need to change it, use **Update**.

> Use a separate key from your local one if you can, so you can revoke CI access independently
> and attribute spend. The `checks` job (dataset validation + tests) needs no key and runs
> regardless.

---

## 3. Configure branch protection

**Unlocks:** actual merge blocking. Until this is set, a failing job shows a red ✗ and the
**Merge** button still works — the pipeline reports, but enforces nothing.

**The `eval` job must have run at least once before GitHub will offer it as a status check.** So
open a throwaway PR touching `prompts/` first, let the workflow run, then come back here.

1. Repo → **Settings** → **Branches**
2. **Add branch protection rule** (or edit the existing rule for `main`)
3. Branch name pattern: `main`
4. Tick **Require status checks to pass before merging**
5. Tick **Require branches to be up to date before merging** (recommended — otherwise a PR can
   pass against a stale baseline)
6. In the search box, add both:
   - `checks` — dataset validation + unit tests
   - `eval` — the live regression gate
7. **Create** / **Save changes**

Confirm: open a PR that touches `prompts/`. You should see both checks listed, and the Merge
button disabled while they're running or failing.

> If you're the repo owner, also consider **Do not allow bypassing the above settings** —
> otherwise admins (you) can merge past a failing gate, which quietly defeats the purpose.

### What blocks and what doesn't

| Condition | CI result | Merge |
|---|---|---|
| Critical regression (≥8% pass-rate drop) | exit 1 | **blocked** |
| Warning (≥3% drop) | exit 0 | allowed — visible in PR comment |
| Critical drift | exit 0 by default | allowed — set `BLOCK_ON_CRITICAL_DRIFT=true` to block |
| Run completed <95% of dataset | exit 1 | **blocked** (run not saved) |
| First run, no baseline | exit 0 | allowed |

---

## 4. Verify the Docker image

**Unlocks:** confidence the container actually builds. The Dockerfile has never been built — the
daemon was stopped every time it was attempted, so this is unverified.

1. Launch **Docker Desktop** and wait for the whale icon to stop animating
2. Confirm the daemon is up:
   ```bash
   docker info
   ```
   Should print system info, not `Cannot connect to the Docker daemon`.
3. Build:
   ```bash
   docker build -t model-regression-detector .
   ```
4. Run it, mounting run history so it isn't lost when the container exits:
   ```bash
   docker run --rm \
     -e OPENAI_API_KEY=your_key \
     -v $(pwd)/src/runs:/app/src/runs \
     -v $(pwd)/reports:/app/reports \
     model-regression-detector
   ```

Confirm: a new run file appears in `src/runs/` on the **host**. If it doesn't, the volume mount
is wrong — without it the container starts with zero history every time and can never compare.

> Watch for this specifically: `.dockerignore` excludes `src/runs` and `reports` from the image
> on purpose (they're mounted, not baked in). If the build fails on a missing `datasets/`
> directory, check that `.dockerignore` doesn't list it.

---

## 5. Slack webhook (optional)

**Unlocks:** alerts outside CI. Everything works without it — the pipeline prints
`SLACK_WEBHOOK_URL not set; skipping Slack alert.` and carries on.

1. https://api.slack.com/apps → **Create New App** → **From scratch**
2. Name it, pick your workspace
3. **Incoming Webhooks** → toggle **On**
4. **Add New Webhook to Workspace** → choose a channel → **Allow**
5. Copy the URL (`https://hooks.slack.com/services/T.../B.../...`)

Local — add to `.env`:
```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

CI — add as a repository secret named `SLACK_WEBHOOK_URL` (same steps as section 2).

Test it against a dedicated channel first:
```bash
python -m src.report      # re-alerts on the existing comparison without a new eval run
```

> Treat the URL as a credential — anyone holding it can post to your channel. It's in
> `.gitignore` via `.env`; never paste it into a commit or the workflow file directly.

### Report links in Slack

Slack alerts link to the HTML report. Locally that's a file path, which isn't clickable for
teammates. In CI, set `REPORT_PUBLIC_URL` to wherever you publish reports (Pages, S3, an
artifact URL) if you want the link to resolve for others.

---

## 6. Optional: scheduled runs

Per-PR runs only happen when someone changes a prompt. Drift is about decay over calendar time —
a provider silently updating a model won't touch your repo at all. A nightly run is what makes
drift detection meaningful, and it accumulates the 8-run history from step 1b on its own.

Add to `.github/workflows/prompt-eval.yml`:

```yaml
on:
  schedule:
    - cron: "0 6 * * *"    # 06:00 UTC daily
```

Note the caveat already documented in the README: CI-generated runs are **not** committed back to
the repo, so a scheduled run alerts but doesn't grow the committed history. To actually accumulate
history you either run locally and commit, or add a commit-back step (deliberately left out —
it's easy to get wrong with concurrent PRs).

---

## Completion check

- [ ] Second run recorded and committed → `python -m src.comparer` prints a comparison
- [ ] Eight runs accumulated → `python -m src.drift` reports a moving average
- [ ] `OPENAI_API_KEY` in repo secrets → `eval` job passes on a PR
- [ ] Branch protection requires `checks` + `eval` → Merge button disabled on a failing PR
- [ ] `docker build` succeeds and a mounted run appears on the host
- [ ] (Optional) Slack alert lands in the channel
