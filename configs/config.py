"""
Centralized configuration for the CreditSense ML pipeline.

All hyperparameters, column definitions, and paths in one place.
Change model type or params here — the entire pipeline follows.
"""

import random as _random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def random_catboost_components(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate N CatBoost ensemble components with random hyperparameter overrides.

    Diversity levers chosen for actual decorrelation (not just cosmetic):
    - `rsm` (column subsample): each model sees a different feature subset, the
      Random-Forest-style mechanism for breaking error correlation.
    - `subsample` (row subsample): each model sees a different row subset.
    - `random_strength`: CatBoost-internal noise injection during split scoring,
      pushes different models toward different splits.
    - `depth`, `learning_rate`, `l2_leaf_reg`: capacity and regularization spread.
    - `random_seed`: distinct per model so internal randomness differs even with
      identical other params.
    """
    rng = _random.Random(seed)
    components = []
    for i in range(n):
        overrides = {
            "depth": rng.choice([4, 5, 6, 7, 8, 9, 10]),
            "learning_rate": round(rng.uniform(0.02, 0.12), 4),
            "l2_leaf_reg": round(rng.uniform(1.0, 20.0), 2),
            "subsample": round(rng.uniform(0.5, 1.0), 3),
            "rsm": round(rng.uniform(0.5, 1.0), 3),
            "random_strength": round(rng.uniform(0.5, 5.0), 2),
            "random_seed": 100 + i,
        }
        components.append({"type": "catboost", "overrides": overrides})
    return components


# Centers found by our two widened-tune runs; used by cluster_catboost_variants.
_TPE_CENTER = {
    "depth": 6,
    "learning_rate": 0.0407,
    "l2_leaf_reg": 140.23,
    "subsample": 0.9646,
    "rsm": 0.894,
    "random_strength": 5.52,
    "border_count": 60,
    "leaf_estimation_iterations": 29,
}

_RANDOM_CENTER = {
    "depth": 4,
    "learning_rate": 0.0826,
    "l2_leaf_reg": 44.77,
    "subsample": 0.968,
    "rsm": 0.848,
    "random_strength": 8.59,
    "border_count": 53,
    "leaf_estimation_iterations": 13,
}


def cluster_catboost_variants(
    center: Dict[str, Any],
    n: int,
    jitter_pct: float = 0.2,
    seed: int = 42,
    seed_offset: int = 200,
    iter_cap: int = 1000,
) -> List[Dict[str, Any]]:
    """Generate N CatBoost configs clustered around `center` with ±jitter_pct noise.

    Integer params jitter ±round(jitter_pct * center). Float params multiply by
    1 + uniform(-jitter_pct, jitter_pct). iterations is forced to `iter_cap`
    (stacking bases should not vary in training budget — fair comparison).
    """
    # Params that CatBoost constrains to the (0, 1] interval — jitter can push
    # them over 1.0 which triggers a hard error in CatBoost's param validator.
    UNIT_INTERVAL_KEYS = {"subsample", "rsm"}

    rng = _random.Random(seed)
    components = []
    for i in range(n):
        overrides: Dict[str, Any] = {}
        for key, val in center.items():
            if isinstance(val, int):
                jitter = max(1, int(abs(val) * jitter_pct))
                overrides[key] = max(1, val + rng.randint(-jitter, jitter))
            elif isinstance(val, float):
                factor = 1.0 + rng.uniform(-jitter_pct, jitter_pct)
                candidate = val * factor
                if key in UNIT_INTERVAL_KEYS:
                    candidate = min(1.0, max(0.05, candidate))
                elif candidate <= 0:  # never let regularization / lr go non-positive
                    candidate = abs(val * 0.1)
                overrides[key] = round(candidate, 4)
        overrides["iterations"] = iter_cap
        overrides["random_seed"] = seed_offset + i
        components.append({"type": "catboost", "overrides": overrides})
    return components


def middle_catboost_variants(
    n: int,
    seed: int = 300,
    seed_offset: int = 300,
    iter_cap: int = 1000,
) -> List[Dict[str, Any]]:
    """Moderate-region CatBoost configs that avoid both known optima's extremes.

    Depth 5–8, l2_leaf_reg 5–30 (not TPE's 140, not Random's 45 extremes),
    lr 0.04–0.12, random_strength 1–5, rsm 0.7–0.95.
    """
    rng = _random.Random(seed)
    components = []
    for i in range(n):
        overrides = {
            "depth": rng.choice([5, 6, 7, 8]),
            "learning_rate": round(rng.uniform(0.04, 0.12), 4),
            "l2_leaf_reg": round(rng.uniform(5.0, 30.0), 2),
            "subsample": round(rng.uniform(0.7, 0.95), 3),
            "rsm": round(rng.uniform(0.7, 0.95), 3),
            "random_strength": round(rng.uniform(1.0, 5.0), 2),
            "border_count": rng.randint(40, 180),
            "leaf_estimation_iterations": rng.randint(5, 20),
            "iterations": iter_cap,
            "random_seed": seed_offset + i,
        }
        components.append({"type": "catboost", "overrides": overrides})
    return components


def diverse_catboost_components(
    n_per_cluster: int,
    seed_base: int = 42,
    iter_cap: int = 1000,
) -> List[Dict[str, Any]]:
    """TPE-cluster + Random-cluster + middle-cluster, each of size n_per_cluster.

    Used as base models for stacking ensembles that span multiple optima regions
    found during CatBoost tuning runs.
    """
    return (
        cluster_catboost_variants(
            _TPE_CENTER,
            n=n_per_cluster,
            jitter_pct=0.2,
            seed=seed_base,
            seed_offset=200,
            iter_cap=iter_cap,
        )
        + cluster_catboost_variants(
            _RANDOM_CENTER,
            n=n_per_cluster,
            jitter_pct=0.2,
            seed=seed_base + 100,
            seed_offset=200 + n_per_cluster,
            iter_cap=iter_cap,
        )
        + middle_catboost_variants(
            n=n_per_cluster,
            seed=seed_base + 200,
            seed_offset=200 + 2 * n_per_cluster,
            iter_cap=iter_cap,
        )
    )


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass
class PathConfig:
    """File paths configuration."""
    train_csv: str = "data/credit_train.csv"
    test_csv: str = "data/credit_test.csv"
    output_dir: str = "outputs"
    models_dir: str = "outputs/models"
    metrics_dir: str = "outputs/metrics"
    predictions_dir: str = "outputs/predictions"
    logs_dir: str = "outputs/logs"

    def __post_init__(self):
        for attr in [
            "output_dir", "models_dir", "metrics_dir",
            "predictions_dir", "logs_dir",
        ]:
            Path(getattr(self, attr)).mkdir(parents=True, exist_ok=True)


@dataclass
class ColumnConfig:
    """Column definitions — targets, categoricals, columns to drop."""
    classification_target: str = "RiskTier"
    regression_target: str = "InterestRate"
    targets: List[str] = field(default_factory=lambda: ["RiskTier", "InterestRate"])

    # Explicitly categorical despite being numeric dtype
    forced_categorical: List[str] = field(default_factory=lambda: [
        "HasCoApplicant",
        "IncomeVerified",
        "PreviousLoanWithBank",
    ])

    # String-typed categorical columns
    categorical: List[str] = field(default_factory=lambda: [
        "EducationLevel",
        "MaritalStatus",
        "HomeOwnership",
        "State",
        "EmploymentStatus",
        "EmployerType",
        "JobCategory",
        "LoanPurpose",
        "CollateralType",
    ])

    # Columns to drop (add after EDA if needed)
    drop_columns: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Model selection and hyperparameters for both tasks."""
    clf_model_type: str = "ensemble"
    reg_model_type: str = "xgboost"

    # Ensemble composition (only used when *_model_type == "ensemble").
    #
    # Each component is either:
    #   - a string (the model_type name, no overrides), or
    #   - a dict {"type": "<name>", "overrides": {...params to layer on base}}
    #
    # The same model_type can appear multiple times with different overrides
    # — that's the "simple MoE" pattern: N variants of one architecture for
    # decorrelated-error variance reduction.
    #
    # weights=None means uniform.
    ensemble_clf_components: List[Any] = field(
        default_factory=lambda: diverse_catboost_components(n_per_cluster=10)
    )
    ensemble_clf_weights: Optional[List[float]] = None
    # Combination mode: "uniform" / "weighted" / "stacking".
    ensemble_clf_mode: str = "stacking"
    # Meta learner for stacking mode (LogReg default; Ridge for regression).
    ensemble_meta_learner_type: str = "logreg"
    ensemble_reg_components: List[Any] = field(
        default_factory=lambda: ["xgboost", "lightgbm", "catboost"]
    )
    ensemble_reg_weights: Optional[List[float]] = None
    ensemble_reg_mode: str = "uniform"

    # -- XGBoost classification --
    xgb_clf_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "multi:softprob",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
        "reg_alpha": 1.0,
        "random_state": 42,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_jobs": -1,
    })

    # -- XGBoost regression --
    xgb_reg_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "reg:squarederror",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
        "reg_alpha": 1.0,
        "random_state": 42,
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_jobs": -1,
    })

    # -- XGBoost ordinal-regression classifier --
    # Treats ordered class labels as continuous, rounds+clips at predict time.
    # Objective fixed to reg:squarederror internally.
    xgb_ordinal_clf_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "reg:pseudohubererror",
        "huber_slope": 1.0,
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
        "reg_alpha": 1.0,
        "random_state": 42,
        "eval_metric": "mphe",
        "tree_method": "hist",
        "n_jobs": -1,
    })

    # -- CatBoost classification --
    # Params updated 2026-04-15 from Optuna TPE+Hyperband on widened-v2 space
    # (best_score 3-fold 0.8209, 60/60 trials). TPE converged to "moderate depth
    # + extreme regularization + many Newton steps" — a different region from
    # Random's shallow+strong-reg optimum. leaf_estimation_iterations hit upper
    # bound 30, suggesting further widening there may still help marginally.
    cat_clf_params: Dict[str, Any] = field(default_factory=lambda: {
        "iterations": 1781,
        "learning_rate": 0.0407,
        "depth": 6,
        "l2_leaf_reg": 140.23,
        "subsample": 0.9646,
        "rsm": 0.894,
        "random_strength": 5.52,
        "border_count": 60,
        "leaf_estimation_iterations": 29,
        "random_seed": 42,
        "thread_count": -1,
        "early_stopping_rounds": 50,
        "bootstrap_type": "Bernoulli",
    })

    # -- CatBoost regression --
    cat_reg_params: Dict[str, Any] = field(default_factory=lambda: {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "random_seed": 42,
        "thread_count": -1,
        "early_stopping_rounds": 50,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
    })

    # -- LightGBM classification --
    lgbm_clf_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "multiclass",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 7,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,  # required for `subsample` to actually trigger; default 0 means bagging disabled
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    })

    # -- LightGBM regression (placeholder) --
    lgbm_reg_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "regression",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 7,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    })

    # -- MLP (sklearn) classification & regression --
    # `hidden_layer_sizes` expressed as a tuple; the search-space layer picks
    # width + depth indirectly via `hidden_layer_size` scalar (inflated at tune time).
    mlp_clf_params: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_layer_sizes": (256, 128),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,
        "batch_size": 256,
        "learning_rate_init": 1e-3,
        "max_iter": 200,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 15,
        "random_state": 42,
    })
    mlp_reg_params: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_layer_sizes": (256, 128),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,
        "batch_size": 256,
        "learning_rate_init": 1e-3,
        "max_iter": 200,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 15,
        "random_state": 42,
    })

    # -- LogReg + PolynomialFeatures classification & regression --
    logreg_poly_clf_params: Dict[str, Any] = field(default_factory=lambda: {
        "C": 0.1,
        "max_iter": 2000,
        "n_jobs": -1,
        "poly_degree": 2,
        "poly_interaction_only": True,
    })
    logreg_poly_reg_params: Dict[str, Any] = field(default_factory=lambda: {
        "alpha": 1.0,
        "poly_degree": 2,
        "poly_interaction_only": True,
    })

    # -- TabNet (pytorch-tabnet) classification & regression --
    tabnet_clf_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_d": 16,
        "n_a": 16,
        "n_steps": 4,
        "gamma": 1.3,
        "lambda_sparse": 1e-3,
        "max_epochs": 100,
        "patience": 15,
        "batch_size": 1024,
        "virtual_batch_size": 128,
        "seed": 42,
        "verbose": 0,
    })
    tabnet_reg_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_d": 16,
        "n_a": 16,
        "n_steps": 4,
        "gamma": 1.3,
        "lambda_sparse": 1e-3,
        "max_epochs": 100,
        "patience": 15,
        "batch_size": 1024,
        "virtual_batch_size": 128,
        "seed": 42,
        "verbose": 0,
    })


@dataclass
class FeatureEngineeringConfig:
    """Feature engineering settings."""
    # High-cardinality columns -> frequency encoding
    freq_encoding_cols: List[str] = field(default_factory=lambda: ["State"])

    # Mid-cardinality columns that additionally get target encoding
    # (stacked on top of the native-categorical path — both encodings coexist).
    target_encoding_cols: List[str] = field(default_factory=lambda: [
        "JobCategory",
        "EmployerType",
        "LoanPurpose",
    ])

    # Columns to log-transform (skewed financial data)
    log_transform_cols: List[str] = field(default_factory=lambda: [
        "AnnualIncome",
        "RequestedLoanAmount",
        "TotalAssets",
        "SavingsBalance",
        "TotalCreditLimit",
    ])

    # Enable domain credit features
    enable_credit_features: bool = True

    # Use LGBM-native categorical handling (ordinal-encoded, passed via
    # `categorical_feature`) instead of one-hot expansion. Cats in
    # `freq_encoding_cols` still use frequency encoding.
    native_categorical_for_lgbm: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    task_type: TaskType = TaskType.CLASSIFICATION
    n_classes: int = 5
    interest_rate_range: Tuple[float, float] = (4.99, 35.99)

    test_size: float = 0.2
    random_state: int = 42

    # Cross-validation
    n_splits: int = 5
    shuffle: bool = True

    # Class weighting (classification only)
    use_class_weights: bool = True

    # Early stopping
    use_early_stopping: bool = True
    early_stopping_rounds: int = 50

    verbose: bool = True


@dataclass
class HyperparameterTuningConfig:
    """Hyperparameter tuning configuration."""
    n_trials: int = 60
    cv_folds: int = 3  # lighter than final-eval CV (5); pick strong params, not final score
    timeout: Optional[int] = None  # no cap — let all n_trials finish

    # XGBoost search spaces — widened for Phase III tune (matched CatBoost scale).
    clf_search_space: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": (300, 2000),
        "learning_rate": (0.005, 0.3),
        "max_depth": (3, 14),
        "min_child_weight": (1, 30),
        "subsample": (0.4, 1.0),
        "colsample_bytree": (0.3, 1.0),
        "reg_alpha": (0.0, 50.0),
        "reg_lambda": (0.0, 150.0),
        "gamma": (0.0, 10.0),
    })

    reg_search_space: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": (300, 2000),
        "learning_rate": (0.005, 0.3),
        "max_depth": (3, 14),
        "min_child_weight": (1, 30),
        "subsample": (0.4, 1.0),
        "colsample_bytree": (0.3, 1.0),
        "reg_alpha": (0.0, 50.0),
        "reg_lambda": (0.0, 150.0),
        "gamma": (0.0, 10.0),
    })

    # LightGBM search space — widened for Phase III tune.
    lgbm_clf_search_space: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": (300, 2000),
        "learning_rate": (0.005, 0.25),
        "num_leaves": (15, 255),
        "max_depth": (-1, 14),
        "min_child_samples": (3, 150),
        "subsample": (0.4, 1.0),
        "colsample_bytree": (0.3, 1.0),
        "reg_alpha": (0.0, 50.0),
        "reg_lambda": (0.0, 150.0),
        "min_split_gain": (0.0, 5.0),
    })

    lgbm_reg_search_space: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": (300, 2000),
        "learning_rate": (0.005, 0.25),
        "num_leaves": (15, 255),
        "max_depth": (-1, 14),
        "min_child_samples": (3, 150),
        "subsample": (0.4, 1.0),
        "colsample_bytree": (0.3, 1.0),
        "reg_alpha": (0.0, 50.0),
        "reg_lambda": (0.0, 150.0),
        "min_split_gain": (0.0, 5.0),
    })

    # CatBoost search spaces. `iterations` bounded so trials stay fast;
    # final CV uses the full 1000 + early stopping.
    cat_clf_search_space: Dict[str, Any] = field(default_factory=lambda: {
        "iterations": (300, 2500),
        "learning_rate": (0.003, 0.3),
        "depth": (3, 14),
        "l2_leaf_reg": (0.1, 150.0),
        "subsample": (0.5, 1.0),
        "rsm": (0.4, 1.0),
        "random_strength": (0.05, 30.0),
        "border_count": (32, 254),
        "leaf_estimation_iterations": (1, 30),
    })

    cat_reg_search_space: Dict[str, Any] = field(default_factory=lambda: {
        "iterations": (300, 2500),
        "learning_rate": (0.003, 0.3),
        "depth": (3, 14),
        "l2_leaf_reg": (0.1, 150.0),
        "subsample": (0.5, 1.0),
        "rsm": (0.4, 1.0),
        "random_strength": (0.05, 30.0),
        "border_count": (32, 254),
        "leaf_estimation_iterations": (1, 30),
    })

    # MLP search space (sklearn MLPClassifier/Regressor).
    # hidden_layer_size is a single int here — we'll inflate to 1-3 layers in
    # a small helper at fit time since Optuna suggest_int doesn't emit tuples.
    # MLP search space. `_mlp_width` is a scalar that the model_builder in
    # run_tune converts to `hidden_layer_sizes=(w, w//2)` before constructing
    # MLPClassifier. Not a real sklearn param — it's a proxy.
    mlp_clf_search_space: Dict[str, Any] = field(default_factory=lambda: {
        "_mlp_width": (64, 512),
        "alpha": (1e-5, 1e-1),
        "learning_rate_init": (1e-4, 1e-2),
        "batch_size": (64, 512),
        "max_iter": (100, 400),
    })
    mlp_reg_search_space: Dict[str, Any] = field(default_factory=lambda: {
        "_mlp_width": (64, 512),
        "alpha": (1e-5, 1e-1),
        "learning_rate_init": (1e-4, 1e-2),
        "batch_size": (64, 512),
        "max_iter": (100, 400),
    })

    # LogReg + Poly search space.
    logreg_poly_clf_search_space: Dict[str, Any] = field(default_factory=lambda: {
        "C": (0.001, 10.0),
        "poly_degree": (1, 2),  # 1 = no polynomial, 2 = pairwise interactions
    })
    logreg_poly_reg_search_space: Dict[str, Any] = field(default_factory=lambda: {
        "alpha": (0.01, 100.0),
        "poly_degree": (1, 2),
    })

    # TabNet search space.
    tabnet_clf_search_space: Dict[str, Any] = field(default_factory=lambda: {
        "n_d": (8, 64),
        "n_a": (8, 64),
        "n_steps": (3, 8),
        "gamma": (1.0, 2.0),
        "lambda_sparse": (1e-5, 1e-2),
        "max_epochs": (50, 150),
        "patience": (10, 25),
    })
    tabnet_reg_search_space: Dict[str, Any] = field(default_factory=lambda: {
        "n_d": (8, 64),
        "n_a": (8, 64),
        "n_steps": (3, 8),
        "gamma": (1.0, 2.0),
        "lambda_sparse": (1e-5, 1e-2),
        "max_epochs": (50, 150),
        "patience": (10, 25),
    })


@dataclass
class Config:
    """Top-level configuration aggregating all sub-configs."""
    paths: PathConfig = field(default_factory=PathConfig)
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tuning: HyperparameterTuningConfig = field(default_factory=HyperparameterTuningConfig)

    def get_target(self) -> str:
        if self.training.task_type == TaskType.CLASSIFICATION:
            return self.columns.classification_target
        return self.columns.regression_target

    def get_model_type(self) -> str:
        if self.training.task_type == TaskType.CLASSIFICATION:
            return self.models.clf_model_type
        return self.models.reg_model_type

    def get_model_params(self) -> Dict[str, Any]:
        task = self.training.task_type
        model = self.get_model_type()
        if model == "ensemble":
            # Ensemble params are injected by model_factory from component configs;
            # this just returns the top-level ensemble spec.
            if task == TaskType.CLASSIFICATION:
                return {
                    "components": list(self.models.ensemble_clf_components),
                    "weights": self.models.ensemble_clf_weights,
                    "mode": self.models.ensemble_clf_mode,
                    "meta_learner_type": self.models.ensemble_meta_learner_type,
                }
            return {
                "components": list(self.models.ensemble_reg_components),
                "weights": self.models.ensemble_reg_weights,
                "mode": self.models.ensemble_reg_mode,
                "meta_learner_type": self.models.ensemble_meta_learner_type,
            }
        param_map = {
            (TaskType.CLASSIFICATION, "xgboost"): self.models.xgb_clf_params,
            (TaskType.REGRESSION, "xgboost"): self.models.xgb_reg_params,
            (TaskType.CLASSIFICATION, "xgboost_ordinal"): self.models.xgb_ordinal_clf_params,
            (TaskType.CLASSIFICATION, "lightgbm"): self.models.lgbm_clf_params,
            (TaskType.REGRESSION, "lightgbm"): self.models.lgbm_reg_params,
            (TaskType.CLASSIFICATION, "catboost"): self.models.cat_clf_params,
            (TaskType.REGRESSION, "catboost"): self.models.cat_reg_params,
            (TaskType.CLASSIFICATION, "mlp"): self.models.mlp_clf_params,
            (TaskType.REGRESSION, "mlp"): self.models.mlp_reg_params,
            (TaskType.CLASSIFICATION, "logreg_poly"): self.models.logreg_poly_clf_params,
            (TaskType.REGRESSION, "logreg_poly"): self.models.logreg_poly_reg_params,
            (TaskType.CLASSIFICATION, "tabnet"): self.models.tabnet_clf_params,
            (TaskType.REGRESSION, "tabnet"): self.models.tabnet_reg_params,
        }
        key = (task, model)
        if key not in param_map:
            raise ValueError(f"No params for task={task}, model={model}")
        return param_map[key]

    def get_component_params(self, component_name: str) -> Dict[str, Any]:
        """Fetch the params dict for a base model by name (used by EnsembleModel)."""
        task = self.training.task_type
        param_map = {
            (TaskType.CLASSIFICATION, "xgboost"): self.models.xgb_clf_params,
            (TaskType.REGRESSION, "xgboost"): self.models.xgb_reg_params,
            (TaskType.CLASSIFICATION, "lightgbm"): self.models.lgbm_clf_params,
            (TaskType.REGRESSION, "lightgbm"): self.models.lgbm_reg_params,
            (TaskType.CLASSIFICATION, "catboost"): self.models.cat_clf_params,
            (TaskType.REGRESSION, "catboost"): self.models.cat_reg_params,
            (TaskType.CLASSIFICATION, "mlp"): self.models.mlp_clf_params,
            (TaskType.REGRESSION, "mlp"): self.models.mlp_reg_params,
            (TaskType.CLASSIFICATION, "logreg_poly"): self.models.logreg_poly_clf_params,
            (TaskType.REGRESSION, "logreg_poly"): self.models.logreg_poly_reg_params,
            (TaskType.CLASSIFICATION, "tabnet"): self.models.tabnet_clf_params,
            (TaskType.REGRESSION, "tabnet"): self.models.tabnet_reg_params,
        }
        key = (task, component_name)
        if key not in param_map:
            raise ValueError(
                f"No component params for task={task}, component={component_name}"
            )
        return param_map[key]

    def get_search_space(self) -> Dict[str, Any]:
        task = self.training.task_type
        model = self.get_model_type()
        space_map = {
            (TaskType.CLASSIFICATION, "xgboost"): self.tuning.clf_search_space,
            (TaskType.CLASSIFICATION, "xgboost_ordinal"): self.tuning.clf_search_space,
            (TaskType.REGRESSION, "xgboost"): self.tuning.reg_search_space,
            (TaskType.CLASSIFICATION, "lightgbm"): self.tuning.lgbm_clf_search_space,
            (TaskType.REGRESSION, "lightgbm"): self.tuning.lgbm_reg_search_space,
            (TaskType.CLASSIFICATION, "catboost"): self.tuning.cat_clf_search_space,
            (TaskType.REGRESSION, "catboost"): self.tuning.cat_reg_search_space,
            (TaskType.CLASSIFICATION, "mlp"): self.tuning.mlp_clf_search_space,
            (TaskType.REGRESSION, "mlp"): self.tuning.mlp_reg_search_space,
            (TaskType.CLASSIFICATION, "logreg_poly"): self.tuning.logreg_poly_clf_search_space,
            (TaskType.REGRESSION, "logreg_poly"): self.tuning.logreg_poly_reg_search_space,
            (TaskType.CLASSIFICATION, "tabnet"): self.tuning.tabnet_clf_search_space,
            (TaskType.REGRESSION, "tabnet"): self.tuning.tabnet_reg_search_space,
        }
        key = (task, model)
        if key not in space_map:
            raise ValueError(f"No search space for task={task}, model={model}")
        return space_map[key]


def load_config(config_path: Optional[str] = None) -> Config:
    """Load config from file or return defaults."""
    if config_path is None:
        return Config()
    raise NotImplementedError("Config file loading not yet implemented")
