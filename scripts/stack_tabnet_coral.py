"""Run Stack(12 TabNet + N CORN-MLP) CV experiments.

Tests whether adding CORN-MLP variants to the SOTA pure-TabNet stacking
ensemble lifts accuracy. Two configs:
  - n_coral=3: matches user's first ask
  - n_coral=5: stronger contribution from CORN-MLP family

Each CV run also persists OOF probas to outputs/oof/<tag>.npz so we can
post-hoc compute Yule's Q vs single TabNet and try gated mixtures.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from configs.config import Config, TaskType
from configs.config import tabnet_plus_coral_components
from training.cross_validator import CrossValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stack_tabnet_coral")


def run(n_coral: int, tag: str):
    cfg = Config()
    cfg.training.task_type = TaskType.CLASSIFICATION
    cfg.training.n_splits = 5
    cfg.models.clf_model_type = "ensemble"
    cfg.models.ensemble_clf_components = tabnet_plus_coral_components(n_coral)
    cfg.models.ensemble_clf_mode = "stacking"
    cfg.models.ensemble_meta_learner_type = "logreg"
    cfg.models.ensemble_stack_method = "holdout"

    n = len(cfg.models.ensemble_clf_components)
    logger.info(f"=== {tag}: n_components={n} (12 TabNet + {n_coral} CORN-MLP) ===")
    t0 = time.time()
    cv = CrossValidator(cfg)
    summary = cv.run()
    elapsed = time.time() - t0
    logger.info(
        f"=== {tag} done: acc={summary['accuracy_mean']:.4f} ± "
        f"{summary['accuracy_std']:.4f}, elapsed {elapsed/60:.1f} min ==="
    )

    # Move OOF file to a tagged location so subsequent runs don't overwrite.
    src = PROJECT / "outputs" / "oof" / "ensemble.npz"
    dst = PROJECT / "outputs" / "oof" / f"{tag}.npz"
    if src.exists():
        src.replace(dst)
        logger.info(f"OOF moved to {dst}")

    # Save summary alongside
    out = {
        "tag": tag,
        "n_coral": n_coral,
        "accuracy_mean": summary["accuracy_mean"],
        "accuracy_std": summary["accuracy_std"],
        "accuracy_per_fold": summary.get("accuracy_per_fold"),
        "elapsed_min": elapsed / 60,
    }
    out_path = PROJECT / "outputs" / "metrics" / f"{tag}_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info(f"Summary saved: {out_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_coral", type=int, required=True, choices=[3, 5])
    parser.add_argument("--tag", type=str, required=True)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    run(n_coral=args.n_coral, tag=args.tag)
