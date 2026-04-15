---
name: ml-experiment-loop
description: ML experiment tracking and iteration loop for iterative model improvement on the CreditSense Kaggle project (classification + regression pipeline via `python main.py --mode cv`). Use this skill whenever the user is iterating on ML models — adding features, swapping algorithms, tuning hyperparameters, changing preprocessing, engineering targets — even if they don't explicitly mention "experiments" or "tracking". Phrases like "试试 XGBoost", "加个特征", "调一下参数", "这次效果怎么样", "why are we stuck", or any approved edit to files under `modeling/`, `feature_engineering/`, `hyperparameter_tuning/`, or `configs/config.py` should trigger this skill. It runs a disciplined loop — propose hypothesis → approve edit → run CV → log to `experiments.md` → judge vs noise → either continue or reflect when stuck — and saves durable lessons via the `/learn` skill so future sessions benefit.
---

# ML Experiment Loop

This skill turns model iteration into a disciplined loop: every ML code change becomes a logged experiment with hypothesis, metric delta, and a judgment call. The loop runs automatically after any approved edit to ML code — no need for the user to ask.

Core idea: iteration without tracking leads to flailing. Tracking with too much ceremony leads to avoidance. This skill aims for the middle — one markdown table row per experiment, one CV run per change, one sentence of conclusion.

## Project context (read once per session)

- Entry point: `python main.py --mode {eda|cv|train|tune|submit} --task {classification|regression}`
- Venv: activate with `source .venv/bin/activate` before any `python` call
- Metrics files (read by the logging script):
  - `outputs/metrics/classification_cv_summary.json` → `accuracy_mean`, `accuracy_std`, `accuracy_per_fold`
  - `outputs/metrics/classification_metrics.json` → single-split `val_metrics.accuracy`, `f1_macro`, `f1_weighted`
  - `outputs/metrics/regression_metrics.json` → single-split `val_metrics.r2`, `rmse`, `mae`
- EDA outputs: `outputs/eda/*.png`
- Experiment log: `experiments.md` at project root
- Primary scores to track (the log script picks these by default):
  - Classification: **CV `accuracy_mean` ± std** (from cv_summary); fall back to `val_metrics.accuracy` if CV wasn't run
  - Regression: **val `r2`**

## The loop

```
propose hypothesis → make edit → user approves → run CV → log row → judge vs noise → next
                                                                            ↓
                                                                 3 in a row without gain
                                                                            ↓
                                                                       reflect
```

### Step 0 — First-time session setup

When a session starts or the user opens a new conversation, do this quick check (no more than 30 seconds of tool calls):

1. **EDA done?** Run `ls outputs/eda/ 2>/dev/null | head -5`. If PNGs exist, EDA has been run — skim the filenames (e.g., `num_AnnualIncome.png`, `cat_LoanPurpose.png`) to learn what's been visualized. Don't re-run.
2. **`experiments.md` exists?** If yes, read the last 5–10 rows to load context on where things stand. If no, scaffold it from `assets/experiments_template.md`.
3. **Data understanding notes?** Look for `data_notes.md` or similar. If none and this is clearly an early-stage session, offer (don't force) to spend 5 minutes reading the EDA plots and writing down 3–5 observations. Grounded hypotheses come from grounded data understanding.

Skip this check on later turns of the same session — it's one-time.

### Step 1 — Propose with a hypothesis, bundled

Before editing any ML code, tell the user:

- **Hypothesis** — why should this help? Tie it to something: an EDA observation, a confusion-matrix pattern, a known technique for this problem shape, a feature interaction you noticed. "Let's just try XGBoost" is a weak hypothesis — push back gently and ask "what specifically do you expect XGBoost to capture that the current model misses?"
- **Expected effect size** — rough guess is fine. "+0.5 to +2 accuracy points" or "should move r2 by 0.005–0.01". Anchors judgment in Step 5.
- **Bundle the supporting changes** — don't ship the smallest possible diff. Before editing, list every change the hypothesis plausibly needs to show its effect, then ship them together. No hard cap on bundle size — if a hypothesis wants 3 changes, ship 3; if it wants 10, ship 10. The constraint is *coherence* (every item in the bundle must support the same hypothesis), not count. Example: "preserve signal currently lost in preprocessing" might naturally include 6 missingness indicators + NetWorth + RecentInquiriesRatio + LoanToAssets + DependentsOver18 + IncomePerMember — all one bundle, one hypothesis. Serial micro-steps waste iterations and produce misleading failures where each isolated piece looks bad.
- **Plan** — which files, which params, all listed together.

If the user just says "do X", still verbalize the hypothesis + the bundle yourself in one line each before editing. The atomic-step instinct is wrong for ML — too many effects are conditional on a companion change.

Exception: when the user explicitly asks for an isolation test ("just this one change, I want to know if it matters alone"), honor it. Bundling is the default, not the mandate.

### Step 2 — Edit, wait for approval

Normal Edit/Write flow. No ceremony.

### Step 3 — Run CV

Right after the approval lands, run CV for the affected task. Don't wait for the user to ask:

```bash
source .venv/bin/activate && python main.py --mode cv --task classification
```

If the change plausibly affects both tasks (shared preprocessing, shared feature engineering), run both. If CV errors out, fix the error and retry — don't log a failed run.

If the change is a bug fix, refactor, or infrastructure-only edit with no expected metric movement, skip CV and skip logging (see "When NOT to log" below).

### Step 4 — Log the experiment

Run the helper:

```bash
python .claude/skills/ml-experiment-loop/scripts/log_experiment.py \
  --task classification \
  --hypothesis "LoanToIncomeRatio should separate high-risk applicants" \
  --change "feature_engineering/credit_features.py: added LoanToIncomeRatio feature" \
  --notes "guarded against AnnualIncome==0"
```

The script reads latest metrics, finds the prior baseline from the last row of the log, appends a new row with id, timestamp, task, hypothesis, change, baseline, new score, delta, and notes. It leaves the **结论** column blank — you fill that in next.

See `scripts/log_experiment.py --help` for all flags. For regression runs pass `--task regression`.

### Step 5 — Judge vs noise

Open `experiments.md`, find the row just added, fill in **结论** and any extra color in **备注**:

- **`work`** — delta > 2× CV std (for classification: typically > ~0.5 pts absolute; for regression r2: > ~0.01)
- **`待定`** — |delta| < 1× CV std, within random variation
- **`not work`** — delta < −2× CV std (actively worse)

Say why in one sentence in 备注. If the change regressed, offer to revert before proposing the next one — but don't auto-revert; the user may want to keep it (e.g., simpler code, infrastructure prerequisite for a later change).

### Step 6 — Continue, commit, or reflect

Read the last 3 rows of `experiments.md`:

- At least one `work` → momentum is there, propose the next logical step.
- All 3 are `待定` or `not work` → stop guessing, trigger **reflection** (see below).
- User expresses frustration ("又没变", "怎么老是这样", "是不是走错方向了") → trigger reflection regardless of row count.

**Checkpoint rule — commit + push every 3 wins**: after filling in 结论 as `work`, count the **total** `work` rows in `experiments.md`. If that count is a multiple of 3 (3, 6, 9, ...), `git commit` + `git push` on the current branch without re-asking. Commit message should reference the three winning rows and the cumulative score delta — look at `git log -3` first to match the repo's existing style. If the working tree has unrelated uncommitted changes (half-done refactors in modules untouched by the experiment), pause and ask the user before committing so nothing extra gets swept in. `not work` / `待定` rows don't count toward the 3.

## Reflection mode (when stuck)

Read `references/reflection_guide.md` for the full procedure. Core idea: when you're stuck, don't try harder within the current frame — question the frame.

Output of a reflection:
1. A short `## Reflection YYYY-MM-DD` block appended to `experiments.md` describing what's common across the recent failures and what new direction was chosen.
2. A concrete next experiment with a hypothesis grounded in the reflection.
3. If the reflection surfaced a durable lesson about this dataset/problem, save it via `/learn` so future sessions inherit it.

## /learn skill integration

When an experiment or reflection yields a lesson that generalizes beyond this single change, call `/learn` to persist it. Good learnings look like:

- "CV std on classification is ~0.003 — anything under 0.5 pt delta is noise on this dataset"
- "Tree-based models plateau around 0.79 accuracy here — breaking through probably needs a linear/interaction component or target engineering"
- "RiskTier and InterestRate targets are highly correlated — improvements on regression features often carry to classification"

Avoid saving every row — `experiments.md` already captures per-experiment detail. `/learn` is for things you want a fresh Claude instance in next week's session to know.

## When NOT to log an experiment

Skip logging (and skip CV) for:
- Bug fixes with no expected metric change
- Pure refactors / code cleanup
- Infra, logging, tooling edits
- Edits to `.claude/`, `README.md`, docs, tests

If in doubt, ask: "log this or skip?"

## Optional: PostToolUse reminder hook

If the user wants an extra nudge against forgetting to log, they can add to `.claude/settings.local.json`:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [{
          "type": "command",
          "command": "if echo \"$CLAUDE_TOOL_INPUT\" | grep -qE '(modeling/|feature_engineering/|configs/config\\.py|hyperparameter_tuning/)'; then echo '→ ML code touched. Run CV + log_experiment.py before moving on.'; fi"
        }]
      }
    ]
  }
}
```

The hook prints a reminder into Claude's context after matching edits. It's belt-and-suspenders — the skill is the primary driver.

## Files in this skill

- `scripts/log_experiment.py` — append a row to `experiments.md` with metrics delta auto-computed
- `references/reflection_guide.md` — what to do when stuck
- `references/eda_checklist.md` — what to look for in `outputs/eda/` on first session
- `assets/experiments_template.md` — initial scaffold for `experiments.md`
