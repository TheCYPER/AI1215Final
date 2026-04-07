#!/usr/bin/env python3
"""
CreditSense ML Pipeline — CLI entry point.

Usage:
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


def run_tune(config: Config):
    """Run hyperparameter tuning with Optuna."""
    from hyperparameter_tuning import OptunaTuner
    from training import Trainer
    from modeling.xgboost_model import XGBoostModel

    trainer = Trainer(config)
    X, y = trainer.load_data()
    X_train, X_val, y_train, y_val = trainer.split_data(X, y)

    trainer.build_preprocessor(X_train)
    trainer.preprocessor_.fit(X_train, y_train)

    tuner = OptunaTuner(config)

    task_type = config.training.task_type

    def model_builder(params):
        model = XGBoostModel(config=params, task_type=task_type)
        if task_type == TaskType.CLASSIFICATION:
            model.build_model(num_classes=config.training.n_classes)
        else:
            model.build_model()
        return model

    results = tuner.tune(X_train, y_train.values, trainer.preprocessor_, model_builder)
    tuner.save_results()
    return results


def run_predict(config: Config, model_path: str, output_path: str = None):
    """Make predictions on test set with a single pipeline."""
    import joblib
    import pandas as pd

    pipeline = joblib.load(model_path)
    test_df = pd.read_csv(config.paths.test_csv)

    predictions = pipeline.predict(test_df)
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
        choices=["train", "cv", "tune", "predict", "submit"],
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

    args = parser.parse_args()

    config = Config()
    config.training.task_type = TaskType(args.task)

    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode.upper()} | Task: {args.task}")
    logger.info("=" * 60)

    try:
        if args.mode == "train":
            run_train(config)

        elif args.mode == "cv":
            run_cv(config)

        elif args.mode == "tune":
            run_tune(config)

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
