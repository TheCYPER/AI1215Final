"""Optuna-based hyperparameter tuner with pruning support.

Changes vs v1:
- Own K-fold loop with fresh preprocessor per fold (no leakage).
- `categorical_feature` passed to model.fit inside each fold so CatBoost /
  LightGBM use their native cat handling during tuning — critical: without
  this, CatBoost's ordered TS is inactive during tune and the chosen params
  under-represent true CV behaviour.
- Pruner support: reports fold-level score to Optuna, HyperbandPruner kills
  bad trials early.
- Sampler selectable: "tpe" (default), "random", "cmaes" for ablation.
"""

import json
import logging
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from configs.config import Config, TaskType

logger = logging.getLogger("tuning")


class OptunaTuner:
    """Optuna tuner with custom K-fold loop + fold-level pruning."""

    def __init__(self, config: Config, sampler: str = "tpe", pruner: str = "hyperband"):
        self.config = config
        self.sampler_name = sampler
        self.pruner_name = pruner
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
        self.study_ = None

    def _sample_params(self, trial) -> Dict[str, Any]:
        search_space = self.config.get_search_space()
        params: Dict[str, Any] = {}
        for name, bounds in search_space.items():
            lo, hi = bounds
            if isinstance(lo, int) and isinstance(hi, int):
                params[name] = trial.suggest_int(name, lo, hi)
            else:
                params[name] = trial.suggest_float(name, lo, hi)
        return params

    def _objective(self, trial, X, y, preprocessor_builder, model_builder):
        params = self._sample_params(trial)
        task = self.config.training.task_type
        n_splits = self.config.tuning.cv_folds

        if task == TaskType.CLASSIFICATION:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scorer = accuracy_score
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            scorer = r2_score

        scores = []
        for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(X, y)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            # Fresh preprocessor per fold — no leakage.
            pre = preprocessor_builder()
            X_tr_t = pre.fit_transform(X_tr, y_tr)
            X_va_t = pre.transform(X_va)

            from feature_engineering.preprocessor import get_categorical_feature_indices
            cat_idx = get_categorical_feature_indices(pre)

            model = model_builder(params)

            fit_kwargs: Dict[str, Any] = {}
            if (
                task == TaskType.CLASSIFICATION
                and self.config.training.use_class_weights
            ):
                fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_tr)
            if self.config.training.use_early_stopping:
                fit_kwargs["eval_set"] = [(X_va_t, y_va.values)]
            if cat_idx:
                fit_kwargs["categorical_feature"] = cat_idx

            model.fit(X_tr_t, y_tr.values, **fit_kwargs)
            pred = model.predict(X_va_t)
            s = float(scorer(y_va.values, pred))
            scores.append(s)

            # Report intermediate value to enable pruning after each fold.
            trial.report(float(np.mean(scores)), step=fold_idx)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    def _build_sampler(self):
        import optuna
        if self.sampler_name == "random":
            return optuna.samplers.RandomSampler(seed=42)
        if self.sampler_name == "cmaes":
            return optuna.samplers.CmaEsSampler(seed=42)
        # default
        return optuna.samplers.TPESampler(seed=42)

    def _build_pruner(self):
        import optuna
        if self.pruner_name == "hyperband":
            # min_resource = 1 fold, max_resource = n_folds; kills worst trials early.
            return optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=self.config.tuning.cv_folds,
                reduction_factor=3,
            )
        if self.pruner_name == "median":
            return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        return optuna.pruners.NopPruner()

    def tune(
        self,
        X,
        y,
        preprocessor_builder: Callable,
        model_builder: Callable,
    ) -> Dict[str, Any]:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def _progress(study, trial):
            best = study.best_value if study.best_trial else float("nan")
            status = trial.state.name  # COMPLETE / PRUNED / FAIL
            val = trial.value if trial.value is not None else float("nan")
            logger.info(
                f"  trial {trial.number + 1}/{self.config.tuning.n_trials} "
                f"status={status} score={val:.4f} best={best:.4f}"
            )

        logger.info(
            f"Optuna study: sampler={self.sampler_name} pruner={self.pruner_name} "
            f"n_trials={self.config.tuning.n_trials} timeout={self.config.tuning.timeout}s "
            f"cv_folds={self.config.tuning.cv_folds}"
        )
        study = optuna.create_study(
            direction="maximize",
            sampler=self._build_sampler(),
            pruner=self._build_pruner(),
        )
        study.optimize(
            lambda trial: self._objective(trial, X, y, preprocessor_builder, model_builder),
            n_trials=self.config.tuning.n_trials,
            timeout=self.config.tuning.timeout,
            callbacks=[_progress],
            show_progress_bar=False,
        )

        self.study_ = study
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value

        logger.info(f"Best score: {self.best_score_:.4f}")
        logger.info(f"Best params: {self.best_params_}")

        return {
            "best_params": self.best_params_,
            "best_score": self.best_score_,
            "sampler": self.sampler_name,
            "pruner": self.pruner_name,
            "n_completed": len([t for t in study.trials if t.state.name == "COMPLETE"]),
            "n_pruned": len([t for t in study.trials if t.state.name == "PRUNED"]),
        }

    def save_results(self, output_path: Optional[str] = None):
        if output_path is None:
            task = self.config.training.task_type.value
            output_path = (
                f"{self.config.paths.output_dir}/"
                f"{task}_tuning_{self.sampler_name}.json"
            )
        results = {
            "best_params": self.best_params_,
            "best_score": self.best_score_,
            "sampler": self.sampler_name,
            "pruner": self.pruner_name,
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Tuning results saved to {output_path}")
