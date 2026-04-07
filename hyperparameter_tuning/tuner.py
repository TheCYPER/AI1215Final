"""
Hyperparameter tuning with Optuna.

Task-aware: optimizes accuracy (clf) or R2 (reg).
Search space is defined in config.
"""

import json
import logging
from typing import Any, Callable, Dict, Optional

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

from configs.config import Config, TaskType

logger = logging.getLogger("tuning")


class OptunaTuner:
    """Optuna-based hyperparameter tuner."""

    def __init__(self, config: Config):
        self.config = config
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
        self.study_ = None

    def _objective(self, trial, X, y, preprocessor, model_builder):
        """Optuna objective function — returns CV score."""
        search_space = self.config.get_search_space()

        params = {}
        for name, bounds in search_space.items():
            if isinstance(bounds[0], int) and isinstance(bounds[1], int):
                params[name] = trial.suggest_int(name, bounds[0], bounds[1])
            else:
                params[name] = trial.suggest_float(name, bounds[0], bounds[1])

        model = model_builder(params)

        X_transformed = preprocessor.transform(X)

        task = self.config.training.task_type
        if task == TaskType.CLASSIFICATION:
            scoring = "accuracy"
            cv = StratifiedKFold(
                n_splits=self.config.tuning.cv_folds,
                shuffle=True,
                random_state=42,
            )
        else:
            scoring = "r2"
            cv = KFold(
                n_splits=self.config.tuning.cv_folds,
                shuffle=True,
                random_state=42,
            )

        scores = cross_val_score(model.model_, X_transformed, y, cv=cv, scoring=scoring)
        return scores.mean()

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        preprocessor,
        model_builder: Callable,
    ) -> Dict[str, Any]:
        """
        Run Optuna study.

        Args:
            X: Raw features (before preprocessing).
            y: Target values.
            preprocessor: Fitted preprocessor (used to transform X).
            model_builder: Callable(params) -> BaseModel with model_ attribute.

        Returns:
            Dict with best_params and best_score.
        """
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self._objective(trial, X, y, preprocessor, model_builder),
            n_trials=self.config.tuning.n_trials,
            timeout=self.config.tuning.timeout,
        )

        self.study_ = study
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value

        logger.info(f"Best score: {self.best_score_:.4f}")
        logger.info(f"Best params: {self.best_params_}")

        return {
            "best_params": self.best_params_,
            "best_score": self.best_score_,
        }

    def save_results(self, output_path: Optional[str] = None):
        """Save tuning results to JSON."""
        if output_path is None:
            task = self.config.training.task_type.value
            output_path = f"{self.config.paths.output_dir}/{task}_tuning_results.json"

        results = {
            "best_params": self.best_params_,
            "best_score": self.best_score_,
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Tuning results saved to {output_path}")
