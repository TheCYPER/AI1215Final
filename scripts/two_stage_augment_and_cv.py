"""Two-stage: inject cls SOTA OOF probas into reg features, re-run CV.

Hypothesis: RiskTier is a direct causal factor of InterestRate in the data-
generating process. Our cls SOTA (12-TabNet stacking, OOF in
`outputs/oof/stack_12tabnet_reg.npz`) reaches 0.8687 accuracy — a strong
prior that reg models currently ignore.

This script:
1. Aligns OOF probas to original train-CSV row order via the stored `indices`.
2. Writes `data/credit_train_cls_augmented.csv` with 5 new numeric columns
   `cls_prob_0 .. cls_prob_4` appended.
3. Runs reg 5-fold CV for both (a) CatBoost default single model and
   (b) 12-TabNet reg stacking, pointing config.paths.train_csv at the
   augmented file.

Why safe from leakage:
- OOF probas: each sample's 5-vector came from a cls model that did NOT
  train on that sample. Reg's model training can use these freely.
- Reg's CV folds use the same X (augmented) and the same y (InterestRate);
  InterestRate is never an input at any stage, so no y→feature leakage.

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        nohup python -u scripts/two_stage_augment_and_cv.py \\
        > /tmp/two_stage.log 2>&1 &

Writes:
- data/credit_train_cls_augmented.csv (created if missing)
- /tmp/two_stage/results.json + results.md
- /tmp/two_stage.log
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from configs.config import Config, TaskType, pure_tabnet_reg_stacking_components  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
OOF_PATH = REPO_ROOT / "outputs" / "oof" / "stack_12tabnet_reg.npz"
TRAIN_CSV = REPO_ROOT / "data" / "credit_train.csv"
TRAIN_AUG_CSV = REPO_ROOT / "data" / "credit_train_cls_augmented.csv"

CLS_PROB_COLS = [f"cls_prob_{k}" for k in range(5)]

BASELINE_CATBOOST = 0.8367  # row #34, 5-fold default CatBoost
BASELINE_TABNET_STACK = 0.8406  # row #49, 12-TabNet reg stacking


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def build_augmented_train_csv() -> pd.DataFrame:
    """Load train CSV + OOF probas, align by index, write augmented CSV."""
    if TRAIN_AUG_CSV.exists():
        log(f"augmented CSV already exists: {TRAIN_AUG_CSV}")
        df = pd.read_csv(TRAIN_AUG_CSV)
        missing = [c for c in CLS_PROB_COLS if c not in df.columns]
        if not missing:
            log(f"  all {len(CLS_PROB_COLS)} cls_prob_* columns present; reusing.")
            return df
        log(f"  missing cols {missing}; rebuilding.")

    log("building augmented CSV from OOF probas")
    data = np.load(OOF_PATH)
    probas = data["y_proba"]
    indices = data["indices"]
    if probas.shape[0] != indices.shape[0]:
        raise ValueError(
            f"OOF proba shape {probas.shape} vs indices {indices.shape} mismatch"
        )

    # OOF is stored in per-fold concat order; `indices[i]` is the original
    # train-CSV row of the i-th OOF entry. Align by sorting indices.
    order = np.argsort(indices)
    probas_aligned = probas[order]  # row i → original train-CSV row i

    train_df = pd.read_csv(TRAIN_CSV)
    if len(train_df) != probas_aligned.shape[0]:
        raise ValueError(
            f"train rows {len(train_df)} vs OOF rows {probas_aligned.shape[0]} mismatch"
        )

    for k, col in enumerate(CLS_PROB_COLS):
        train_df[col] = probas_aligned[:, k]

    TRAIN_AUG_CSV.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_AUG_CSV, index=False)
    log(f"  wrote {TRAIN_AUG_CSV} ({len(train_df)} rows, +5 cls_prob_* columns)")
    return train_df


def make_reg_config(
    model_type: str,
    n_splits: int = 5,
    ensemble_mode: str = "stacking",
) -> Config:
    cfg = Config()
    cfg.training.task_type = TaskType.REGRESSION
    cfg.models.reg_model_type = model_type
    cfg.training.n_splits = n_splits
    cfg.paths.train_csv = str(TRAIN_AUG_CSV)

    if model_type == "ensemble":
        cfg.models.ensemble_reg_mode = ensemble_mode
        cfg.models.ensemble_meta_learner_type = "ridge"
        cfg.models.ensemble_reg_components = pure_tabnet_reg_stacking_components()
    return cfg


def run_reg_cv(cfg: Config, label: str) -> dict:
    from training.cross_validator import CrossValidator

    log(f"=== CV: {label} ===")
    log(f"  train_csv: {cfg.paths.train_csv}")
    log(f"  reg_model_type: {cfg.models.reg_model_type}")

    t0 = time.time()
    cv = CrossValidator(cfg)
    results = cv.run()
    elapsed = time.time() - t0
    log(
        f"  r2={results['r2_mean']:.4f} ± {results['r2_std']:.4f} "
        f"({elapsed/60:.1f} min)"
    )
    return {
        "label": label,
        "model_type": cfg.models.reg_model_type,
        "r2_mean": results["r2_mean"],
        "r2_std": results["r2_std"],
        "r2_per_fold": results["r2_per_fold"],
        "elapsed_min": round(elapsed / 60, 1),
    }


def write_summary(results: list, out_dir: Path) -> None:
    lines = [
        "# Two-Stage Regression Results (cls OOF probas as features)",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        f"Baselines: CatBoost default = **{BASELINE_CATBOOST:.4f}** (row #34); "
        f"12-TabNet reg stacking = **{BASELINE_TABNET_STACK:.4f}** (row #49).",
        "",
        "| Label | Model | r2 mean | r2 std | Δ vs CatBoost | Δ vs TabNet stack | Time |",
        "|------|-------|---------|--------|---------------|-------------------|------|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['label']} | — | ERROR: {r['error']} | — | — | — | — |")
            continue
        d_cb = r["r2_mean"] - BASELINE_CATBOOST
        d_tn = r["r2_mean"] - BASELINE_TABNET_STACK
        lines.append(
            f"| {r['label']} | {r['model_type']} | "
            f"**{r['r2_mean']:.4f}** | {r['r2_std']:.4f} | "
            f"{d_cb:+.4f} | {d_tn:+.4f} | {r['elapsed_min']}m |"
        )
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    log("two-stage autorun start")

    # Step 0: build / verify augmented training CSV
    build_augmented_train_csv()

    out_dir = Path("/tmp/two_stage")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results: list = []

    # --- Experiment 1: CatBoost default + augmented features --------------
    # Fast, cheap sanity check. If cls OOF provides signal, even a plain
    # single CatBoost with default params should move past 0.8367.
    try:
        cfg = make_reg_config("catboost", n_splits=5)
        r = run_reg_cv(cfg, "catboost_default + cls_oof")
        all_results.append(r)
    except Exception as exc:  # noqa: BLE001
        log(f"  ERROR in catboost_default: {exc}")
        import traceback; traceback.print_exc()
        all_results.append({"label": "catboost_default + cls_oof", "error": str(exc)})
    (out_dir / "results.json").write_text(json.dumps(all_results, indent=2))
    write_summary(all_results, out_dir)

    # --- Experiment 2: 12-TabNet reg stacking + augmented features -------
    # Current best reg stacking (row #49 = 0.8406). Does cls-prior help on
    # top of the NN diversity we already have?
    try:
        cfg = make_reg_config("ensemble", n_splits=5, ensemble_mode="stacking")
        r = run_reg_cv(cfg, "12-TabNet stacking + cls_oof")
        all_results.append(r)
    except Exception as exc:  # noqa: BLE001
        log(f"  ERROR in tabnet_stack: {exc}")
        import traceback; traceback.print_exc()
        all_results.append({"label": "12-TabNet stacking + cls_oof", "error": str(exc)})
    (out_dir / "results.json").write_text(json.dumps(all_results, indent=2))
    write_summary(all_results, out_dir)

    log("=" * 60)
    log("TWO-STAGE SUMMARY")
    log("=" * 60)
    for r in all_results:
        if "error" in r:
            log(f"  {r['label']}: ERROR ({r['error']})")
        else:
            d_cb = r["r2_mean"] - BASELINE_CATBOOST
            d_tn = r["r2_mean"] - BASELINE_TABNET_STACK
            log(
                f"  {r['label']}: r2={r['r2_mean']:.4f} ± {r['r2_std']:.4f} "
                f"(vs CatBoost {d_cb:+.4f}, vs TabNet-stack {d_tn:+.4f})"
            )


if __name__ == "__main__":
    main()
