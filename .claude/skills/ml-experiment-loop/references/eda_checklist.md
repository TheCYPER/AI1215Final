# EDA checklist — first session only

Goal: spend 5–10 minutes grounding yourself in the data before proposing any model change. A hypothesis like "LoanToIncomeRatio separates high-risk applicants" is only useful if you've looked at the ratio's distribution across RiskTier.

Skip this whole file if `outputs/eda/` is already populated and the user has already gone through the plots. It's a one-time orientation.

## What's already generated

`python main.py --mode eda` produces PNGs in `outputs/eda/`. Typical contents:

- `00_missing_values.png` — which columns have NaNs and how much
- `01_target_risktier.png` — class distribution for classification target
- `02_target_interestrate.png` — distribution for regression target
- `03_correlation_heatmap.png` — numeric correlations
- `num_<Feature>.png` — distribution per numeric column, often split by target
- `cat_<Feature>.png` — category counts + target relationship per categorical column

## What to look at and note

Don't just flip through the PNGs. For each of these questions, write 1 sentence into `data_notes.md` (create it at project root if missing):

1. **Class balance** — is the classification target roughly balanced, or skewed? → informs whether class weights / sampling matters.
2. **Target variance** — for regression, is the target long-tailed? log skew? → informs target transformation.
3. **Missing-value concentration** — are missings on a few columns or spread out? are missings informative (e.g., "no CoApplicant" encoded as NaN)? → informs imputation choice.
4. **High-correlation pairs** — which numeric features are redundant with each other? → informs feature selection / collinearity concerns for linear models.
5. **Category cardinality** — any categorical with >50 unique values? → informs encoding (target encoding vs one-hot).
6. **Obvious target relationships** — which features show strong separation by target in the per-feature plots? → first-pass feature importance, before any model.
7. **Suspicious stuff** — sentinels (−1, 9999), outliers, duplicate rows, class 0 having 10× the samples of others, etc. Flag anything that looks off.

## Output

`data_notes.md` should be ~15–30 bullet points. Rough template:

```markdown
# Data notes — CreditSense

## Targets
- RiskTier: 5 classes, roughly balanced (14k per class)
- InterestRate: continuous, 2–15%, mild right tail

## Features worth attention
- AnnualIncome: heavy right tail → try log transform
- DebtToIncomeRatio: already correlates strongly with RiskTier visually
- State: 50+ categories → target encoding

## Missing values
- CollateralValue ~40% missing (only loans with collateral)
- CoApplicant* columns missing when HasCoApplicant=0 → informative, keep as indicator

## Flags
- AnnualIncome has a few zeros — treat as missing or cap at 1
- ...
```

Once this exists, experiments can reference it. A hypothesis "log-transform AnnualIncome should help linear models" is much stronger when grounded in "AnnualIncome is right-tailed per EDA".
