"""Automated regression model sweep.

Tests multiple model types + configs for the InterestRate regression task.
Writes results to /tmp/regression_sweep/results.md.

Usage:
    source .venv/bin/activate
    OMP_NUM_THREADS=1 python -u scripts/regression_sweep.py 2>&1 | tee /tmp/regression_sweep/run.log
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import Config, TaskType

OUT_DIR = Path("/tmp/regression_sweep")


def run_cv(config: Config, name: str) -> dict:
    """Run 5-fold CV and return results."""
    from training.cross_validator import CrossValidator

    ts = time.time()
    cv = CrossValidator(config)
    results = cv.run()
    elapsed = time.time() - ts

    return {
        "name": name,
        "r2_mean": results["r2_mean"],
        "r2_std": results["r2_std"],
        "r2_per_fold": results["r2_per_fold"],
        "elapsed_min": round(elapsed / 60, 1),
    }


def run_tune(config: Config, model_type: str, n_trials: int = 40) -> dict:
    """Run Optuna tune and return best params."""
    from data_cleaning.column_types import infer_column_types
    from feature_engineering.preprocessor import (
        build_preprocessor,
        get_categorical_feature_indices,
    )
    from hyperparameter_tuning.tuner import OptunaTuner
    from modeling import MODEL_REGISTRY

    cv_obj = __import__("training.cross_validator", fromlist=["CrossValidator"])
    CrossValidator = cv_obj.CrossValidator

    temp_cv = CrossValidator(config)
    X, y = temp_cv.load_data()

    num_cols, cat_cols = infer_column_types(
        X, targets=config.columns.targets,
        forced_categorical=config.columns.forced_categorical,
        drop_columns=config.columns.drop_columns,
    )

    def preprocessor_builder():
        return build_preprocessor(
            numeric_cols=num_cols, categorical_cols=cat_cols,
            freq_encoding_cols=config.features.freq_encoding_cols,
            log_transform_cols=config.features.log_transform_cols,
            enable_credit_features=config.features.enable_credit_features,
            target_encoding_cols=config.features.target_encoding_cols,
            native_categorical=config.features.native_categorical_for_lgbm,
        )

    task_type = config.training.task_type
    base_params = dict(config.get_model_params())
    model_cls = MODEL_REGISTRY[model_type]

    def model_builder(params):
        merged = {**base_params, **params}
        model = model_cls(config=merged, task_type=task_type)
        model.build_model()
        return model

    config.tuning.n_trials = n_trials
    config.tuning.timeout = None

    tuner = OptunaTuner(config, sampler="tpe", pruner="hyperband")
    results = tuner.tune(X, y, preprocessor_builder, model_builder)
    tuner.save_results()
    return results


EXPERIMENTS = [
    # 1. Baselines: each model type with default params
    {
        "name": "xgb_default",
        "desc": "XGBoost default params (current baseline)",
        "model": "xgboost",
        "action": "cv",
    },
    {
        "name": "catboost_default",
        "desc": "CatBoost default regression params",
        "model": "catboost",
        "action": "cv",
    },
    {
        "name": "lgbm_default",
        "desc": "LightGBM default regression params",
        "model": "lightgbm",
        "action": "cv",
    },
    {
        "name": "tabnet_default",
        "desc": "TabNet default regression params",
        "model": "tabnet",
        "action": "cv",
    },
    # 2. Tune the top 2 models
    {
        "name": "catboost_tune",
        "desc": "CatBoost Optuna 40 trials",
        "model": "catboost",
        "action": "tune",
        "n_trials": 40,
    },
    {
        "name": "tabnet_tune",
        "desc": "TabNet Optuna 40 trials",
        "model": "tabnet",
        "action": "tune",
        "n_trials": 40,
    },
    # 3. Apply tuned params and re-CV
    {
        "name": "catboost_tuned_cv",
        "desc": "CatBoost with tuned params → 5-fold CV",
        "model": "catboost",
        "action": "apply_tune_cv",
        "tune_ref": "catboost_tune",
    },
    {
        "name": "tabnet_tuned_cv",
        "desc": "TabNet with tuned params → 5-fold CV",
        "model": "tabnet",
        "action": "apply_tune_cv",
        "tune_ref": "tabnet_tune",
    },
]


def main():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Regression sweep started at {datetime.now():%Y-%m-%d %H:%M:%S}")

    all_results = []
    tune_best_params = {}

    for exp in EXPERIMENTS:
        name = exp["name"]
        print(f"\n{'='*60}")
        print(f"[{datetime.now():%H:%M:%S}] {name}: {exp['desc']}")
        print(f"{'='*60}")

        try:
            config = Config()
            config.training.task_type = TaskType.REGRESSION
            config.models.reg_model_type = exp["model"]

            if exp["action"] == "cv":
                result = run_cv(config, name)
                all_results.append(result)
                print(f"  r2={result['r2_mean']:.4f} ± {result['r2_std']:.4f}")

            elif exp["action"] == "tune":
                ts = time.time()
                tune_result = run_tune(config, exp["model"], exp.get("n_trials", 40))
                elapsed = time.time() - ts
                best = tune_result.get("best_params", {})
                tune_best_params[name] = best
                # Save tune results
                with open(OUT_DIR / f"{name}_best.json", "w") as f:
                    json.dump({"best_params": best, "best_score": tune_result.get("best_score")}, f, indent=2)
                print(f"  Best score: {tune_result.get('best_score')}")
                print(f"  Best params: {best}")
                print(f"  Time: {elapsed/60:.1f} min")
                all_results.append({
                    "name": name,
                    "tune_best_score": tune_result.get("best_score"),
                    "elapsed_min": round(elapsed / 60, 1),
                })

            elif exp["action"] == "apply_tune_cv":
                ref = exp["tune_ref"]
                if ref in tune_best_params:
                    best = tune_best_params[ref]
                else:
                    # Try loading from file
                    bp_path = OUT_DIR / f"{ref}_best.json"
                    if bp_path.exists():
                        with open(bp_path) as f:
                            best = json.load(f).get("best_params", {})
                    else:
                        print(f"  SKIP: no tune results for {ref}")
                        continue

                # Apply tuned params
                current = config.get_model_params()
                current.update(best)
                result = run_cv(config, name)
                all_results.append(result)
                print(f"  r2={result['r2_mean']:.4f} ± {result['r2_std']:.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_results.append({"name": name, "error": str(e)})

        _write_results(all_results)

    print(f"\nDone at {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Results: {OUT_DIR / 'results.md'}")


def _write_results(results: list):
    lines = [
        "# Regression Sweep Results",
        f"\nGenerated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"\nBaseline: XGBoost default **r2=0.8244** (row #1)",
        "",
        "| Name | R2 Mean | R2 Std | vs Baseline | Time |",
        "|------|---------|--------|-------------|------|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['name']} | ERROR | — | — | — |")
        elif "r2_mean" in r:
            delta = r["r2_mean"] - 0.8244
            lines.append(
                f"| {r['name']} | **{r['r2_mean']:.4f}** | {r['r2_std']:.4f} | "
                f"{delta:+.4f} | {r['elapsed_min']}m |"
            )
        elif "tune_best_score" in r:
            lines.append(
                f"| {r['name']} | tune best={r.get('tune_best_score', 'N/A')} | — | — | "
                f"{r['elapsed_min']}m |"
            )

    with open(OUT_DIR / "results.md", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
