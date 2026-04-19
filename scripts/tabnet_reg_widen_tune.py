"""TabNet regression widened-space tune autorun.

Rationale: previous tune (scripts/tabnet_reg_autorun.py) hit 4 search-space
bounds — n_d stuck at lower bound 8, gamma/lambda_sparse/patience stuck at
upper bounds. TPE signal was "go smaller+sparser+more patient" but was
capped. This run widens those bounds and adds `batch_size` as a new axis.

Widened bounds (vs previous tune):
- n_d:          (8, 64)      → (4, 64)     # allow shallower
- n_steps:      (3, 8)       → (3, 10)     # allow more decision steps
- gamma:        (1.0, 2.0)   → (1.0, 3.0)  # allow stronger feature reuse
- lambda_sparse: (1e-5, 1e-2) → (1e-5, 0.05) # allow sparser attention
- patience:     (10, 25)     → (10, 40)    # allow more tolerant ES
- batch_size:   NEW          → (512, 2048) # previously fixed at 1024

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        nohup python -u scripts/tabnet_reg_widen_tune.py \\
        > /tmp/tabnet_reg_widen_tune.log 2>&1 &

Writes:
- /tmp/tabnet_reg_widen_tune.log (progress)
- outputs/regression_tuning_tabnet_widened.json (tune artifact)
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

import torch  # noqa: E402

torch.set_num_threads(1)

from configs.config import Config, TaskType  # noqa: E402


WIDENED_SEARCH_SPACE = {
    "n_d": (4, 64),
    "n_a": (8, 64),
    "n_steps": (3, 10),
    "gamma": (1.0, 3.0),
    "lambda_sparse": (1e-5, 0.05),
    "max_epochs": (50, 150),
    "patience": (10, 40),
    "batch_size": (512, 2048),
}


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

    log("=== TUNE: TabNet reg WIDENED, 60 trials, no timeout ===")
    log(f"  search space: {json.dumps(WIDENED_SEARCH_SPACE, indent=2)}")

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
        "search_space": WIDENED_SEARCH_SPACE,
        "elapsed_min": round(elapsed / 60, 1),
    }
    out_path = Path("outputs/regression_tuning_tabnet_widened.json")
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
    # Override the TabNet reg search space with our widened version.
    cfg.tuning.tabnet_reg_search_space = dict(WIDENED_SEARCH_SPACE)
    return cfg


def main() -> None:
    log("TabNet regression WIDENED tune autorun start")

    # Step 1: tune (60 trials, no timeout, 3-fold internal)
    cfg = fresh_config(n_splits=3)
    cfg.tuning.n_trials = 60
    cfg.tuning.timeout = None
    cfg.tuning.cv_folds = 3
    tune_result = run_tune(cfg)

    # Step 2: apply best → 3-fold CV
    cfg = fresh_config(n_splits=3)
    tuned_params = dict(cfg.models.tabnet_reg_params)
    tuned_params.update(tune_result["best_params"])
    cfg.models.tabnet_reg_params = tuned_params
    tuned_cv = run_cv(cfg, "tabnet_widened_tuned_3fold")

    # Step 3: also 5-fold (real comparison to other reg models)
    cfg = fresh_config(n_splits=5)
    cfg.models.tabnet_reg_params = tuned_params
    tuned_5fold = run_cv(cfg, "tabnet_widened_tuned_5fold")

    # Summary
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"Previous tuned 3-fold (narrow search):    r2 = 0.8338 ± 0.0056")
    log(f"Tune best (widened, 3-fold internal):     {tune_result['best_score']:.4f}")
    log(f"Applied widened tuned 3-fold:             r2={tuned_cv['r2_mean']:.4f} ± {tuned_cv['r2_std']:.4f}")
    log(f"Applied widened tuned 5-fold (comparable to CatBoost default 0.8367):")
    log(f"                                          r2={tuned_5fold['r2_mean']:.4f} ± {tuned_5fold['r2_std']:.4f}")
    log(f"delta widened vs previous tuned (3-fold): {tuned_cv['r2_mean'] - 0.8338:+.4f}")

    # Log which params hit new bounds
    best = tune_result["best_params"]
    bound_report = []
    for param, (lo, hi) in WIDENED_SEARCH_SPACE.items():
        if param in best:
            val = best[param]
            if isinstance(lo, float):
                at_lower = val < lo + 0.05 * (hi - lo)
                at_upper = val > hi - 0.05 * (hi - lo)
            else:
                at_lower = val <= lo + max(1, int(0.05 * (hi - lo)))
                at_upper = val >= hi - max(1, int(0.05 * (hi - lo)))
            flag = "🔻 floor" if at_lower else ("🔺 ceiling" if at_upper else "OK")
            bound_report.append(f"  {param} = {val} (range {lo}-{hi}) {flag}")
    log("Search space boundary report:")
    for line in bound_report:
        log(line)


if __name__ == "__main__":
    main()
