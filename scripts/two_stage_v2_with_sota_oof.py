"""Two-stage v2: SOTA-quality gated cls OOF → augmented reg CV.

The actual cls SOTA (row #48 OOF acc 0.8718) is a gated mixture, not pure
stacking. To regenerate matching OOF we run two independent cls 5-fold CVs:
  (a) Stack(12 TabNet + 5 CORN-MLP) + LogReg meta — the "jury" (0.8679 acc)
  (b) TabNet single (tuned)                       — the "judge"  (0.8573 acc)
then combine them at threshold t=0.59:
    P_gated[i] = P_stack[i]  if max(P_stack[i]) >= 0.59 else P_tabnet[i]
which reproduces the row #48 honest OOF acc 0.8718.

Pipeline:
  1. Cls CV with stack(12+5) — saves OOF to outputs/oof/ensemble.npz
     → copy to outputs/oof/cls_stack_12tn_5coral.npz
  2. Cls CV with tabnet single — saves OOF to outputs/oof/tabnet.npz
     → copy to outputs/oof/cls_tabnet_single.npz
  3. Build gated OOF probas (P_stack vs P_tabnet at t=0.59)
     → outputs/oof/cls_sota_gated.npz
  4. Rebuild data/credit_train_cls_aug_v2.csv from gated OOF
  5. Reg 12-TabNet stacking 5-fold CV on the augmented CSV

Why this beats v1:
- v1 used some legacy 0.7633-acc OOF and still got reg +0.0019.
- v2's gated OOF is acc 0.8718 — ~10.85 pts more cls-prior signal in
  the input feature; expect another +0.005-0.010 on reg (target 0.847+).

Total wall-clock: ~12-13 hrs (cls stack 5-6h + cls TabNet 1-2h +
reg stack 5h + negligible glue).

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \\
        nohup python -u scripts/two_stage_v2_with_sota_oof.py \\
        > /tmp/two_stage_v2.log 2>&1 &
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from configs.config import (  # noqa: E402
    Config,
    TaskType,
    pure_tabnet_reg_stacking_components,
    tabnet_plus_coral_components,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = REPO_ROOT / "data" / "credit_train.csv"
TRAIN_AUG_V2_CSV = REPO_ROOT / "data" / "credit_train_cls_aug_v2.csv"

OOF_DIR = REPO_ROOT / "outputs" / "oof"
OOF_AUTO_ENSEMBLE = OOF_DIR / "ensemble.npz"
OOF_AUTO_TABNET = OOF_DIR / "tabnet.npz"
OOF_STACK_STABLE = OOF_DIR / "cls_stack_12tn_5coral.npz"
OOF_TABNET_STABLE = OOF_DIR / "cls_tabnet_single.npz"
OOF_GATED_STABLE = OOF_DIR / "cls_sota_gated.npz"

GATE_THRESHOLD = 0.59  # row #48 chosen via half-split validation
CLS_PROB_COLS = [f"cls_prob_{k}" for k in range(5)]

BASELINE_REG_CATBOOST = 0.8367  # row #34
BASELINE_REG_TABNET_STACK = 0.8406  # row #49 (no augmentation)
PRIOR_TWO_STAGE_V1 = 0.8425  # row #50 (legacy 0.7633 OOF)
TARGET_GATED_ACC = 0.8718  # row #48 OOF


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Step 1: cls Stack(12 TabNet + 5 CORN-MLP)
# ---------------------------------------------------------------------------

def run_cls_stack_cv() -> dict:
    from training.cross_validator import CrossValidator

    log("=" * 60)
    log("STEP 1/5: cls Stack(12 TabNet + 5 CORN-MLP) + LogReg meta — 5-fold CV")
    log("  Expected ~5-6 hrs; OOF auto-saves to outputs/oof/ensemble.npz")
    log("  Target acc ~0.8679 (row #47)")
    log("=" * 60)

    cfg = Config()
    cfg.training.task_type = TaskType.CLASSIFICATION
    cfg.models.clf_model_type = "ensemble"
    cfg.models.ensemble_clf_mode = "stacking"
    cfg.models.ensemble_meta_learner_type = "logreg"
    cfg.models.ensemble_clf_components = tabnet_plus_coral_components(5)
    cfg.training.n_splits = 5
    cfg.paths.train_csv = str(TRAIN_CSV)

    t0 = time.time()
    cv = CrossValidator(cfg)
    res = cv.run()
    elapsed_min = round((time.time() - t0) / 60, 1)
    log(
        f"  done: acc={res['accuracy_mean']:.4f} ± {res['accuracy_std']:.4f} "
        f"({elapsed_min} min)"
    )

    if not OOF_AUTO_ENSEMBLE.exists():
        raise FileNotFoundError(f"expected OOF at {OOF_AUTO_ENSEMBLE}")
    shutil.copy2(OOF_AUTO_ENSEMBLE, OOF_STACK_STABLE)
    log(f"  copied → {OOF_STACK_STABLE}")
    return {
        "step": "cls_stack_cv",
        "accuracy_mean": res["accuracy_mean"],
        "accuracy_std": res["accuracy_std"],
        "accuracy_per_fold": res["accuracy_per_fold"],
        "elapsed_min": elapsed_min,
    }


# ---------------------------------------------------------------------------
# Step 2: cls TabNet single
# ---------------------------------------------------------------------------

def run_cls_tabnet_cv() -> dict:
    from training.cross_validator import CrossValidator

    log("=" * 60)
    log("STEP 2/5: cls TabNet single — 5-fold CV")
    log("  Expected ~1.5-2 hrs; OOF auto-saves to outputs/oof/tabnet.npz")
    log("  Target acc ~0.8573 (row #24)")
    log("=" * 60)

    cfg = Config()
    cfg.training.task_type = TaskType.CLASSIFICATION
    cfg.models.clf_model_type = "tabnet"
    cfg.training.n_splits = 5
    cfg.paths.train_csv = str(TRAIN_CSV)

    t0 = time.time()
    cv = CrossValidator(cfg)
    res = cv.run()
    elapsed_min = round((time.time() - t0) / 60, 1)
    log(
        f"  done: acc={res['accuracy_mean']:.4f} ± {res['accuracy_std']:.4f} "
        f"({elapsed_min} min)"
    )

    if not OOF_AUTO_TABNET.exists():
        raise FileNotFoundError(f"expected OOF at {OOF_AUTO_TABNET}")
    shutil.copy2(OOF_AUTO_TABNET, OOF_TABNET_STABLE)
    log(f"  copied → {OOF_TABNET_STABLE}")
    return {
        "step": "cls_tabnet_cv",
        "accuracy_mean": res["accuracy_mean"],
        "accuracy_std": res["accuracy_std"],
        "accuracy_per_fold": res["accuracy_per_fold"],
        "elapsed_min": elapsed_min,
    }


# ---------------------------------------------------------------------------
# Step 3: gated OOF probas
# ---------------------------------------------------------------------------

def build_gated_oof() -> dict:
    log("=" * 60)
    log(f"STEP 3/5: build gated OOF probas (t={GATE_THRESHOLD})")
    log("=" * 60)

    stack = np.load(OOF_STACK_STABLE)
    tabnet = np.load(OOF_TABNET_STABLE)

    order_s = np.argsort(stack["indices"])
    order_t = np.argsort(tabnet["indices"])

    P_stack = stack["y_proba"][order_s]
    P_tabnet = tabnet["y_proba"][order_t]
    y_true_s = stack["y_true"][order_s]
    y_true_t = tabnet["y_true"][order_t]

    if not np.array_equal(y_true_s, y_true_t):
        # Different fold splits → re-align by indices instead. We sorted by
        # indices already, so y_true should match if both CVs covered all
        # 35k rows. If they differ here something is wrong.
        raise ValueError(
            "y_true mismatch between stack and tabnet OOFs after index sort"
        )
    y_true = y_true_s

    top1_stack = P_stack.max(axis=1)
    mask_low = top1_stack < GATE_THRESHOLD
    P_gated = np.where(mask_low[:, None], P_tabnet, P_stack)
    pred_gated = P_gated.argmax(axis=1)
    acc_stack = float((P_stack.argmax(axis=1) == y_true).mean())
    acc_tabnet = float((P_tabnet.argmax(axis=1) == y_true).mean())
    acc_gated = float((pred_gated == y_true).mean())

    np.savez_compressed(
        OOF_GATED_STABLE,
        y_true=y_true,
        y_pred=pred_gated,
        y_proba=P_gated,
        indices=np.arange(len(y_true)),
    )
    log(f"  stack acc:  {acc_stack:.4f}")
    log(f"  tabnet acc: {acc_tabnet:.4f}")
    log(f"  gated acc:  {acc_gated:.4f}  (target ≈ {TARGET_GATED_ACC:.4f})")
    log(
        f"  low-conf fallback to TabNet: {int(mask_low.sum())}/{len(mask_low)} "
        f"({100 * mask_low.mean():.1f}%)"
    )
    log(f"  saved → {OOF_GATED_STABLE}")
    return {
        "step": "build_gated_oof",
        "stack_accuracy": acc_stack,
        "tabnet_accuracy": acc_tabnet,
        "gated_accuracy": acc_gated,
        "low_conf_count": int(mask_low.sum()),
        "low_conf_pct": float(mask_low.mean()),
    }


# ---------------------------------------------------------------------------
# Step 4: build augmented CSV
# ---------------------------------------------------------------------------

def build_v2_augmented_csv() -> dict:
    log("=" * 60)
    log("STEP 4/5: build v2 augmented CSV from gated OOF")
    log("=" * 60)

    data = np.load(OOF_GATED_STABLE)
    probas = data["y_proba"]
    if probas.shape != (35000, 5):
        raise ValueError(f"unexpected proba shape: {probas.shape}")

    train_df = pd.read_csv(TRAIN_CSV)
    if len(train_df) != probas.shape[0]:
        raise ValueError(
            f"train rows {len(train_df)} vs OOF rows {probas.shape[0]}"
        )
    for k, col in enumerate(CLS_PROB_COLS):
        train_df[col] = probas[:, k]
    train_df.to_csv(TRAIN_AUG_V2_CSV, index=False)
    log(f"  wrote {TRAIN_AUG_V2_CSV} ({len(train_df)} rows + 5 cls_prob_*)")
    return {"step": "build_v2_aug", "csv": str(TRAIN_AUG_V2_CSV)}


# ---------------------------------------------------------------------------
# Step 5: reg CV on augmented data
# ---------------------------------------------------------------------------

def run_reg_two_stage_v2() -> dict:
    from training.cross_validator import CrossValidator

    log("=" * 60)
    log("STEP 5/5: reg 12-TabNet stacking 5-fold CV on v2-augmented CSV")
    log("  Expected ~5 hrs; baselines:")
    log(f"    - row #34 CatBoost reg: {BASELINE_REG_CATBOOST}")
    log(f"    - row #49 TabNet stack reg: {BASELINE_REG_TABNET_STACK}")
    log(f"    - row #50 v1 two-stage:    {PRIOR_TWO_STAGE_V1}")
    log("=" * 60)

    cfg = Config()
    cfg.training.task_type = TaskType.REGRESSION
    cfg.models.reg_model_type = "ensemble"
    cfg.models.ensemble_reg_mode = "stacking"
    cfg.models.ensemble_meta_learner_type = "ridge"
    cfg.models.ensemble_reg_components = pure_tabnet_reg_stacking_components()
    cfg.training.n_splits = 5
    cfg.paths.train_csv = str(TRAIN_AUG_V2_CSV)

    t0 = time.time()
    cv = CrossValidator(cfg)
    res = cv.run()
    elapsed_min = round((time.time() - t0) / 60, 1)
    log(
        f"  done: r2={res['r2_mean']:.4f} ± {res['r2_std']:.4f} "
        f"({elapsed_min} min)"
    )
    return {
        "step": "reg_two_stage_v2",
        "r2_mean": res["r2_mean"],
        "r2_std": res["r2_std"],
        "r2_per_fold": res["r2_per_fold"],
        "elapsed_min": elapsed_min,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def write_summary(steps: list, out_dir: Path) -> None:
    lines = [
        "# Two-Stage v2 SOTA Results",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "Cls baselines: stack(12+5) target acc 0.8679 (row #47); "
        f"TabNet single 0.8573; gated SOTA target {TARGET_GATED_ACC} (row #48).",
        "",
        f"Reg baselines: CatBoost = {BASELINE_REG_CATBOOST} (row #34); "
        f"TabNet stack = {BASELINE_REG_TABNET_STACK} (row #49); "
        f"v1 two-stage = {PRIOR_TWO_STAGE_V1} (row #50).",
        "",
        "## Step results",
    ]
    for s in steps:
        lines.append("")
        lines.append(f"### {s.get('step', '?')}")
        for k, v in s.items():
            if k == "step":
                continue
            lines.append(f"- **{k}**: {v}")
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    log("Two-stage v2 SOTA autorun start")
    out_dir = Path("/tmp/two_stage_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    steps: list = []

    pipeline = [
        ("cls_stack_cv", run_cls_stack_cv),
        ("cls_tabnet_cv", run_cls_tabnet_cv),
        ("build_gated_oof", build_gated_oof),
        ("build_v2_aug", build_v2_augmented_csv),
        ("reg_two_stage_v2", run_reg_two_stage_v2),
    ]
    for name, fn in pipeline:
        try:
            steps.append(fn())
        except Exception as exc:  # noqa: BLE001
            log(f"  STEP {name} ERROR: {exc}")
            import traceback
            traceback.print_exc()
            steps.append({"step": name, "error": str(exc)})
            (out_dir / "results.json").write_text(json.dumps(steps, indent=2, default=str))
            write_summary(steps, out_dir)
            return
        (out_dir / "results.json").write_text(json.dumps(steps, indent=2, default=str))
        write_summary(steps, out_dir)

    log("=" * 60)
    log("FINAL SUMMARY")
    log("=" * 60)
    for s in steps:
        nm = s.get("step", "?")
        rest = {k: v for k, v in s.items() if k != "step"}
        log(f"  {nm}: {json.dumps(rest, default=str)}")


if __name__ == "__main__":
    main()
