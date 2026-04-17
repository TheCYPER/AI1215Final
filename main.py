#!/usr/bin/env python3
"""
CreditSense ML Pipeline — CLI entry point.

Usage:
    python main.py --mode eda
    python main.py --mode train --task classification
    python main.py --mode train --task regression
    python main.py --mode cv --task classification
    python main.py --mode cv --task regression
    python main.py --mode tune --task classification
    python main.py --mode submit
    python main.py --mode predict --task classification --model_path outputs/models/classification_pipeline.joblib
"""

import argparse
import sys
from pathlib import Path

from configs import Config, TaskType
from utils.logger import setup_logger

logger = setup_logger("main")


def run_eda(config: Config):
    """Run exploratory data analysis with visualizations."""
    from data_exploration import run_eda as _run_eda
    _run_eda(config)


def run_train(config: Config):
    """Train with single train/val split."""
    from training import Trainer

    trainer = Trainer(config)
    results = trainer.run()
    return results


def run_cv(config: Config):
    """Run k-fold cross-validation."""
    from training import CrossValidator

    cv = CrossValidator(config)
    results = cv.run()
    return results


def run_tune(config: Config, sampler: str = "tpe", pruner: str = "hyperband"):
    """Run Optuna hyperparameter tuning against the configured model type.

    Uses the full training CSV (no train/val pre-split) and fits a fresh
    preprocessor per fold inside the tuner for leakage-free scoring.
    """
    from data_cleaning.column_types import infer_column_types
    from feature_engineering.preprocessor import build_preprocessor
    from hyperparameter_tuning import OptunaTuner
    from modeling import MODEL_REGISTRY
    from training import CrossValidator

    cv = CrossValidator(config)
    X, y = cv.load_data()

    # Infer column types once — they don't depend on any fold.
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

    tuner = OptunaTuner(config, sampler=sampler, pruner=pruner)

    task_type = config.training.task_type
    model_type = config.get_model_type()
    base_params = dict(config.get_model_params())
    model_cls = MODEL_REGISTRY[model_type]

    def model_builder(params):
        merged = {**base_params, **params}
        # MLP proxy: `_mlp_width` scalar → `hidden_layer_sizes` tuple.
        if "_mlp_width" in merged:
            w = int(merged.pop("_mlp_width"))
            merged["hidden_layer_sizes"] = (w, w // 2)
        model = model_cls(config=merged, task_type=task_type)
        if task_type == TaskType.CLASSIFICATION:
            model.build_model(num_classes=config.training.n_classes)
        else:
            model.build_model()
        return model

    results = tuner.tune(X, y, preprocessor_builder, model_builder)
    tuner.save_results()
    return results


def _load_and_predict(artifact, raw_df):
    """Handle both sklearn Pipeline and ensemble dict formats."""
    if isinstance(artifact, dict) and artifact.get("format") == "ensemble_dict":
        preprocessor = artifact["preprocessor"]
        model = artifact["model"]
        X_t = preprocessor.transform(raw_df)
        return model.predict(X_t)
    # Standard sklearn Pipeline
    return artifact.predict(raw_df)


def run_predict(config: Config, model_path: str, output_path: str = None):
    """Make predictions on test set with a single pipeline."""
    import joblib
    import pandas as pd

    artifact = joblib.load(model_path)
    test_df = pd.read_csv(config.paths.test_csv)

    predictions = _load_and_predict(artifact, test_df)
    logger.info(f"Generated {len(predictions)} predictions")

    if output_path is None:
        task = config.training.task_type.value
        output_path = f"{config.paths.predictions_dir}/{task}_predictions.csv"

    pd.DataFrame({"prediction": predictions}).to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")


def run_submit(config: Config, clf_path: str = None, reg_path: str = None):
    """Generate combined Kaggle submission."""
    from submission import SubmissionGenerator

    generator = SubmissionGenerator(config)
    submission = generator.generate(
        clf_pipeline_path=clf_path,
        reg_pipeline_path=reg_path,
    )
    logger.info(f"Submission shape: {submission.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="CreditSense ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["eda", "train", "cv", "tune", "predict", "submit"],
        help="Pipeline mode",
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="classification",
        help="Task type (default: classification)",
    )
    parser.add_argument("--model_path", default=None, help="Path to trained pipeline")
    parser.add_argument("--clf_model", default=None, help="Path to clf pipeline (submit mode)")
    parser.add_argument("--reg_model", default=None, help="Path to reg pipeline (submit mode)")
    parser.add_argument("--output_path", default=None, help="Output path for predictions")
    parser.add_argument(
        "--sampler",
        default="tpe",
        choices=["tpe", "random", "cmaes"],
        help="Optuna sampler for tune mode (default: tpe)",
    )
    parser.add_argument(
        "--pruner",
        default="hyperband",
        choices=["hyperband", "median", "none"],
        help="Optuna pruner for tune mode (default: hyperband)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Override clf_model_type (classification) or reg_model_type "
            "(regression) for this run. Accepts any key from MODEL_REGISTRY. "
            "Lets the automation script switch base models without editing config.py."
        ),
    )
    parser.add_argument(
        "--apply_tune_results",
        default=None,
        help=(
            "Path to JSON file with tune best_params. Merges into the current "
            "model's params dict before run (e.g. used after --mode tune to re-CV "
            "with tuned hyperparameters)."
        ),
    )

    args = parser.parse_args()

    config = Config()
    config.training.task_type = TaskType(args.task)

    if args.model is not None:
        if args.task == "classification":
            config.models.clf_model_type = args.model
        else:
            config.models.reg_model_type = args.model

    if args.apply_tune_results:
        import json
        with open(args.apply_tune_results) as f:
            tune_data = json.load(f)
        best = tune_data.get("best_params", {})
        # MLP proxy: _mlp_width scalar → hidden_layer_sizes tuple.
        if "_mlp_width" in best:
            w = int(best.pop("_mlp_width"))
            best["hidden_layer_sizes"] = (w, w // 2)
        # Fetch the current model's params dict and merge in place.
        current_params = config.get_model_params()
        current_params.update(best)
        logger.info(f"Applied tune best_params: {best}")

    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode.upper()} | Task: {args.task}")
    logger.info("=" * 60)

    try:
        if args.mode == "eda":
            run_eda(config)

        elif args.mode == "train":
            run_train(config)

        elif args.mode == "cv":
            run_cv(config)

        elif args.mode == "tune":
            run_tune(config, sampler=args.sampler, pruner=args.pruner)

        elif args.mode == "predict":
            model_path = args.model_path
            if model_path is None:
                model_path = f"{config.paths.models_dir}/{args.task}_pipeline.joblib"
            if not Path(model_path).exists():
                logger.error(f"Model not found: {model_path}")
                sys.exit(1)
            run_predict(config, model_path, args.output_path)

        elif args.mode == "submit":
            run_submit(config, args.clf_model, args.reg_model)

        logger.info("=" * 60)
        logger.info("Done.")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
