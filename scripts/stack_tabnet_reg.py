"""Run 5-fold CV on 12-TabNet regression stacking (pure_tabnet_reg_stacking_components).

Mirrors the cls SOTA recipe: 8 tuned seed-jittered + 3 wild + 1 baseline,
stacking mode with Ridge meta-learner. Question: does the 12-TabNet cls SOTA
pattern port to regression despite single-TabNet-reg being the weakest
strong model? (See experiments row #43 for why we're skeptical, row #44 for
why the answer was "no for same-family CatBoost"; neural init diversity might
behave differently.)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from configs.config import Config, TaskType, pure_tabnet_reg_stacking_components
from training.cross_validator import CrossValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stack_tabnet_reg")


def main(tag: str):
    cfg = Config()
    cfg.training.task_type = TaskType.REGRESSION
    cfg.training.n_splits = 5
    cfg.models.reg_model_type = "ensemble"
    cfg.models.ensemble_reg_components = pure_tabnet_reg_stacking_components()
    cfg.models.ensemble_reg_mode = "stacking"
    cfg.models.ensemble_meta_learner_type = "ridge"
    cfg.models.ensemble_stack_method = "holdout"

    n = len(cfg.models.ensemble_reg_components)
    logger.info(f"=== {tag}: {n} TabNet reg bases, Ridge meta, 5-fold ===")
    t0 = time.time()
    cv = CrossValidator(cfg)
    summary = cv.run()
    elapsed = time.time() - t0
    logger.info(
        f"=== {tag} done: r2={summary['r2_mean']:.4f} ± {summary['r2_std']:.4f}, "
        f"rmse={summary.get('rmse_mean', float('nan')):.4f}, "
        f"elapsed {elapsed/60:.1f} min ==="
    )

    # Move OOF file to a tagged location
    src = PROJECT / "outputs" / "oof" / "ensemble.npz"
    dst = PROJECT / "outputs" / "oof" / f"{tag}.npz"
    if src.exists():
        src.replace(dst)
        logger.info(f"OOF moved to {dst}")

    out = {
        "tag": tag,
        "task": "regression",
        "r2_mean": summary.get("r2_mean"),
        "r2_std": summary.get("r2_std"),
        "rmse_mean": summary.get("rmse_mean"),
        "r2_per_fold": summary.get("r2_per_fold"),
        "elapsed_min": elapsed / 60,
    }
    out_path = PROJECT / "outputs" / "metrics" / f"{tag}_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info(f"Summary saved: {out_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="stack_12tabnet_reg")
    args = parser.parse_args()
    main(tag=args.tag)
