# Reflection guide — when stuck, question the frame

Trigger this when the last 3 experiment rows are all `待定` or `not work`, or when the user says things like "怎么又没动" / "还是差不多" / "是不是方向错了".

The failure mode to avoid: proposing more variations on the same theme. If three tree-based models in a row plateau, the answer is unlikely to be "try a fourth tree-based model with different hyperparameters". The signal is telling you something.

## Procedure

### 1. Read the last 5–10 rows

Open `experiments.md` and read recent history. What changed? What was the hypothesis each time? What ended up as `待定` vs `not work`?

### 2. Find the shared assumption

Ask: what do these recent tries have in common, and what are they collectively assuming?

Examples of shared-but-unexamined assumptions:
- **All tree-based models** → maybe the decision boundary isn't axis-aligned; try linear + interactions, or a neural net with tabular-specific tricks.
- **All same feature set** → maybe information is in an interaction or transformation you haven't added.
- **All same target encoding** → for regression, maybe log/Box-Cox; for classification, maybe a different grouping (binary high-risk vs low-risk).
- **All validated on the same CV split** → maybe the split is misleading — try a different stratification or time-based split if applicable.
- **All tuning the same few params** → maybe the sensitive param is somewhere else (e.g., class weights, sample weights, learning-rate schedule).

### 3. Enumerate what hasn't been tried

Walk through these categories and note which ones the recent history has *not* explored:

- [ ] **Different model family** (e.g., if all tree: try linear-based, kernel-based, or tabular NN like TabNet/FT-Transformer)
- [ ] **Feature geometry** — log transforms, binning, target encoding of categoricals, polynomial/interaction features, embeddings for high-cardinality cats
- [ ] **Target engineering** — for regression: log(y), Box-Cox; for classification: target grouping (5→2 classes), ordinal handling
- [ ] **Preprocessing** — robust scaler vs standard, missing-value strategy, outlier clipping
- [ ] **Loss / class weights** — for imbalanced or ordinal targets, custom loss often unlocks meaningful gains
- [ ] **Ensembling** — stacking, blending, different seeds
- [ ] **Data** — any external data, denoising, relabeling obvious errors

### 4. Pick one direction that's meaningfully different

Not two, not three — one. Reflections that propose "try 5 new directions" turn into flailing by another name. Commit to one experiment, run it, see what happens.

### 5. Write the reflection to experiments.md

Append a section like this (not a table row):

```markdown
## Reflection — 2026-04-14

Last 4 experiments all plateaued around 0.789 ± 0.003 (CV acc) — feature additions
(LoanToIncomeRatio, DebtRatio, AgeGroup) each gained nothing, all used GradientBoosting.

**Shared assumption**: tree-based model is strong enough and the signal is in
individual features. Maybe the signal is in interactions a tree can't easily
build at current depth, or the boundary needs a linear component.

**New direction**: try stacked ensemble — LightGBM + LogisticRegression(with
polynomial interactions on top 5 features) + ridge blender. If still plateau,
next pivot is target engineering (5-class → ordinal regression).
```

### 6. If there's a durable lesson, save it via /learn

Not everything is worth saving. But if the reflection surfaces something that will still be true next week — "CV std is ~0.003, so anything under 0.5 pt is noise", or "tree models plateau here, breakthrough needs interactions" — call `/learn` so a fresh Claude session in the future inherits it.

Rule of thumb: if you'd want next week's Claude to know this without reading all of `experiments.md`, save it. Otherwise let it stay in the log.

## What a reflection should NOT do

- Don't list 5 new directions. Pick one.
- Don't propose "tune the current model harder" — that's not reflecting, that's continuing.
- Don't blame the data unless you've actually inspected it. Usually the frame is the problem, not the dataset.
- Don't declare the project done / impossible. Reflection is a pivot, not a surrender.
