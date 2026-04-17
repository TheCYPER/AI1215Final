"""TabNet regression autorun: baseline → tune → apply tuned.

Steps (sequential, unattended-friendly):
1. Baseline 3-fold CV with default TabNet reg params (row #37 candidate)
2. Optuna TPE + Hyperband tune, 60 trials, no timeout, internal 3-fold
3. Apply best params → 3-fold CV (row #38 candidate)

Writes progress to /tmp/tabnet_reg_autorun.log and tune artifacts under
outputs/regression_tuning_tabnet.json.

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        nohup python -u scripts/tabnet_reg_autorun.py \\
        > /tmp/tabnet_reg_autorun.log 2>&1 &
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# TabNet + OpenMP on this machine: anything other than 1 thread segfaults.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

torch.set_num_threads(1)

from configs.config import Config, TaskType  # noqa: E402


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def run_cv(config: Config, label: str) -> dict:
    from training.cross_validator import CrossValidator

    log(f"=== CV: {label} ===")
    t0 = time.time()
    cv = CrossValidator(config)
    results = cv.run()
    elapsed = time.time() - t0
    log(
        f"  r2={results['r2_mean']:.4f} ± {results['r2_std']:.4f} "
        f"({elapsed/60:.1f} min)"
    )
    return {
        "label": label,
        "r2_mean": results["r2_mean"],
        "r2_std": results["r2_std"],
        "r2_per_fold": results["r2_per_fold"],
        "elapsed_min": round(elapsed / 60, 1),
    }


def run_tune(config: Config) -> dict:
    from data_cleaning.column_types import infer_column_types
    from feature_engineering.preprocessor import build_preprocessor
    from hyperparameter_tuning.tuner import OptunaTuner
    from modeling import MODEL_REGISTRY
    from training.cross_validator import CrossValidator

    log("=== TUNE: TabNet regression, 60 trials, no timeout ===")

    base_cv = CrossValidator(config)
    X, y = base_cv.load_data()

    num_cols, cat_cols = infer_column_types(
        X,
        targets=config.columns.targets,
        forced_categorical=config.columns.forced_categorical,
        drop_columns=config.columns.drop_columns,
    )

    def preprocessor_builder():
        return build_preprocessor(
            numeric_cols=num_cols,
            categorical_cols=cat_cols,
            freq_encoding_cols=config.features.freq_encoding_cols,
            log_transform_cols=config.features.log_transform_cols,
            enable_credit_features=config.features.enable_credit_features,
            target_encoding_cols=config.features.target_encoding_cols,
            native_categorical=config.features.native_categorical_for_lgbm,
        )

    task_type = config.training.task_type
    base_params = dict(config.get_model_params())
    model_cls = MODEL_REGISTRY["tabnet"]

    def model_builder(params):
        merged = {**base_params, **params}
        model = model_cls(config=merged, task_type=task_type)
        model.build_model()
        return model

    t0 = time.time()
    tuner = OptunaTuner(config, sampler="tpe", pruner="hyperband")
    results = tuner.tune(X, y, preprocessor_builder, model_builder)
    elapsed = time.time() - t0

    artifact = {
        "best_params": results["best_params"],
        "best_score": results["best_score"],
        "sampler": "tpe",
        "pruner": "hyperband",
        "n_trials": config.tuning.n_trials,
        "cv_folds": config.tuning.cv_folds,
        "elapsed_min": round(elapsed / 60, 1),
    }
    out_path = Path("outputs/regression_tuning_tabnet.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    log(f"  best={results['best_score']:.4f} ({elapsed/60:.1f} min)")
    log(f"  saved → {out_path}")
    log(f"  best_params: {json.dumps(results['best_params'], indent=2)}")
    return artifact


def fresh_config(n_splits: int) -> Config:
    cfg = Config()
    cfg.training.task_type = TaskType.REGRESSION
    cfg.models.reg_model_type = "tabnet"
    cfg.training.n_splits = n_splits
    return cfg


def main() -> None:
    log("TabNet regression autorun start")

    # Step 1: baseline 3-fold CV
    cfg = fresh_config(n_splits=3)
    baseline = run_cv(cfg, "tabnet_default_3fold")

    # Step 2: tune (60 trials, no timeout, 3-fold internal)
    cfg = fresh_config(n_splits=3)
    cfg.tuning.n_trials = 60
    cfg.tuning.timeout = None
    cfg.tuning.cv_folds = 3
    tune_result = run_tune(cfg)

    # Step 3: apply best → 3-fold CV
    cfg = fresh_config(n_splits=3)
    tuned_params = dict(cfg.models.tabnet_reg_params)
    tuned_params.update(tune_result["best_params"])
    cfg.models.tabnet_reg_params = tuned_params
    tuned_cv = run_cv(cfg, "tabnet_tuned_3fold")

    # Summary
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"baseline 3-fold: r2={baseline['r2_mean']:.4f} ± {baseline['r2_std']:.4f}")
    log(f"tune best (3-fold internal): {tune_result['best_score']:.4f}")
    log(f"tuned 3-fold: r2={tuned_cv['r2_mean']:.4f} ± {tuned_cv['r2_std']:.4f}")
    log(f"delta tuned vs baseline: {tuned_cv['r2_mean'] - baseline['r2_mean']:+.4f}")


if __name__ == "__main__":
    main()
