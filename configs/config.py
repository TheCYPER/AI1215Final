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


# Five-cluster centers spanning the full hyperparameter landscape.
# Cluster 2 = TPE optimum, Cluster 4 = Random optimum,
# Clusters 1/3/5 fill the gaps: outer-shallow, transition-middle, outer-deep.
_OUTER_SHALLOW_CENTER = {
    "depth": 3,
    "learning_rate": 0.12,
    "l2_leaf_reg": 10.0,
    "subsample": 0.85,
    "rsm": 0.75,
    "random_strength": 12.0,
    "border_count": 80,
    "leaf_estimation_iterations": 5,
}

_TRANSITION_MIDDLE_CENTER = {
    "depth": 5,
    "learning_rate": 0.065,
    "l2_leaf_reg": 40.0,
    "subsample": 0.92,
    "rsm": 0.85,
    "random_strength": 4.0,
    "border_count": 70,
    "leaf_estimation_iterations": 15,
}

_OUTER_DEEP_CENTER = {
    "depth": 9,
    "learning_rate": 0.02,
    "l2_leaf_reg": 180.0,
    "subsample": 0.98,
    "rsm": 0.92,
    "random_strength": 2.5,
    "border_count": 50,
    "leaf_estimation_iterations": 35,
}


def five_cluster_catboost_components(
    n_per_cluster: int = 3,
    seed_base: int = 42,
    iter_cap: int = 1000,
) -> List[Dict[str, Any]]:
    """5 clusters × n_per_cluster, spanning full hyperparameter landscape.

    Cluster order (low→high depth/reg axis):
      1. outer-shallow  (depth~3, l2~10, lr~0.12)
      2. Random center   (depth~4, l2~45, lr~0.08) — tuning-validated
      3. transition-mid  (depth~5, l2~40, lr~0.065)
      4. TPE center      (depth~6, l2~140, lr~0.04) — tuning-validated
      5. outer-deep      (depth~9, l2~180, lr~0.02)
    """
    clusters = [
        (_OUTER_SHALLOW_CENTER, seed_base, 400),
        (_RANDOM_CENTER, seed_base + 10, 403),
        (_TRANSITION_MIDDLE_CENTER, seed_base + 20, 406),
        (_TPE_CENTER, seed_base + 30, 409),
        (_OUTER_DEEP_CENTER, seed_base + 40, 412),
    ]
    components: List[Dict[str, Any]] = []
    for center, seed, seed_offset in clusters:
        components.extend(
            cluster_catboost_variants(
                center,
                n=n_per_cluster,
                jitter_pct=0.2,
                seed=seed,
                seed_offset=seed_offset,
                iter_cap=iter_cap,
            )
        )
    return components


def pure_tabnet_stacking_components() -> List[Dict[str, Any]]:
    """Config F: 12 TabNet (8 tuned-seed-jittered + 3 wild + 1 baseline).

    Best ensemble config from sweep (row #30): 0.8687 ± 0.0040.
    NN random initialization + architecture variety provides true diversity
    that meta-learner can exploit — unlike same-architecture tree ensembles.
    """
    _TABNET_TUNED = {
        "n_d": 29, "n_a": 22, "n_steps": 6, "gamma": 1.7146,
        "lambda_sparse": 0.007308, "max_epochs": 129, "patience": 20,
        "batch_size": 1024, "virtual_batch_size": 128, "verbose": 0,
    }
    _WILD_CONFIGS = [
        {  # wide + shallow attention
            "n_d": 48, "n_a": 48, "n_steps": 3, "gamma": 1.2,
            "lambda_sparse": 0.001, "max_epochs": 100, "patience": 15,
            "batch_size": 512, "virtual_batch_size": 64, "verbose": 0,
            "seed": 300,
        },
        {  # narrow + deep attention
            "n_d": 16, "n_a": 16, "n_steps": 8, "gamma": 1.8,
            "lambda_sparse": 0.01, "max_epochs": 150, "patience": 20,
            "batch_size": 2048, "virtual_batch_size": 256, "verbose": 0,
            "seed": 301,
        },
        {  # mid + high sparsity
            "n_d": 32, "n_a": 24, "n_steps": 5, "gamma": 1.5,
            "lambda_sparse": 0.05, "max_epochs": 120, "patience": 18,
            "batch_size": 1024, "virtual_batch_size": 128, "verbose": 0,
            "seed": 302,
        },
    ]
    _BASELINE = {
        "n_d": 16, "n_a": 16, "n_steps": 4, "gamma": 1.3,
        "lambda_sparse": 1e-3, "max_epochs": 100, "patience": 15,
        "batch_size": 1024, "virtual_batch_size": 128, "verbose": 0,
        "seed": 999,
    }
    components: List[Dict[str, Any]] = []
    for i in range(8):
        overrides = dict(_TABNET_TUNED)
        overrides["seed"] = 100 + i
        components.append({"type": "tabnet", "overrides": overrides})
    for cfg in _WILD_CONFIGS:
        components.append({"type": "tabnet", "overrides": dict(cfg)})
    components.append({"type": "tabnet", "overrides": dict(_BASELINE)})
    return components


def pure_tabnet_reg_stacking_components() -> List[Dict[str, Any]]:
    """12 TabNet regression bases: 8 widened-tuned-seed-jittered + 3 wild + 1 baseline.

    Mirrors pure_tabnet_stacking_components (cls SOTA row #30) but for
    regression. Tuned base uses row #43 widened-Optuna best_params
    (n_d=32, n_steps=9, gamma=2.41, lambda_sparse=0.044, patience=37),
    which hit 3-fold r2=0.8349 as single model.

    Key uncertainty — cls SOTA path may NOT port to reg:
    - Single TabNet reg is the WEAKEST strong model (5-fold r2=0.8272).
    - Reg meta features are scalar (12-dim) vs cls proba (60-dim) —
      Ridge meta has much less to work with than LogReg meta had on cls.
    - Row #44 already showed same-family reg stacking fails (30 CatBoost
      averaged to single-model baseline).

    If this works despite those headwinds it's because NN random init
    still gives decorrelation trees don't have. If it doesn't, we know
    reg really does need heterogeneous architectures.
    """
    _TABNET_TUNED = {
        "n_d": 32, "n_a": 19, "n_steps": 9, "gamma": 2.41,
        "lambda_sparse": 0.044, "max_epochs": 146, "patience": 37,
        "batch_size": 1587, "virtual_batch_size": 128, "verbose": 0,
    }
    _WILD_CONFIGS = [
        {  # wide + shallow attention
            "n_d": 48, "n_a": 48, "n_steps": 3, "gamma": 1.5,
            "lambda_sparse": 0.001, "max_epochs": 100, "patience": 15,
            "batch_size": 512, "virtual_batch_size": 64, "verbose": 0,
            "seed": 300,
        },
        {  # narrow + deep attention
            "n_d": 16, "n_a": 16, "n_steps": 8, "gamma": 2.0,
            "lambda_sparse": 0.01, "max_epochs": 130, "patience": 20,
            "batch_size": 2048, "virtual_batch_size": 256, "verbose": 0,
            "seed": 301,
        },
        {  # row #42 narrow-tuned (shallower + less sparse)
            "n_d": 8, "n_a": 15, "n_steps": 4, "gamma": 1.95,
            "lambda_sparse": 0.0099, "max_epochs": 122, "patience": 24,
            "batch_size": 1024, "virtual_batch_size": 128, "verbose": 0,
            "seed": 302,
        },
    ]
    _BASELINE = {
        "n_d": 16, "n_a": 16, "n_steps": 4, "gamma": 1.3,
        "lambda_sparse": 1e-3, "max_epochs": 100, "patience": 15,
        "batch_size": 1024, "virtual_batch_size": 128, "verbose": 0,
        "seed": 999,
    }
    components: List[Dict[str, Any]] = []
    for i in range(8):
        overrides = dict(_TABNET_TUNED)
        overrides["seed"] = 100 + i
        components.append({"type": "tabnet", "overrides": overrides})
    for cfg in _WILD_CONFIGS:
        components.append({"type": "tabnet", "overrides": dict(cfg)})
    components.append({"type": "tabnet", "overrides": dict(_BASELINE)})
    return components


def tabnet_plus_coral_components(n_coral: int = 3) -> List[Dict[str, Any]]:
    """Config F + N CORN-MLP variants. Tests whether CORN-MLP (acc 0.8373,
    Q=0.938 vs TabNet) can lift the 0.8687 SOTA when meta-learner sees both
    families' OOF probas.

    CORN-MLP variants vary hidden_sizes / dropout / lr to give some
    decorrelation among themselves.
    """
    components = pure_tabnet_stacking_components()
    _CORAL_VARIANTS = [
        {"hidden_sizes": [256, 128, 64], "dropout": 0.20, "lr": 1e-3, "seed": 200},
        {"hidden_sizes": [512, 256, 128], "dropout": 0.30, "lr": 7e-4, "seed": 201},
        {"hidden_sizes": [192, 96], "dropout": 0.15, "lr": 1.5e-3, "seed": 202},
        {"hidden_sizes": [384, 192, 96], "dropout": 0.25, "lr": 1e-3, "seed": 203},
        {"hidden_sizes": [128, 64, 32], "dropout": 0.10, "lr": 2e-3, "seed": 204},
    ]
    for v in _CORAL_VARIANTS[:n_coral]:
        ov = dict(v)
        ov.setdefault("max_epochs", 80)
        ov.setdefault("patience", 12)
        ov.setdefault("batch_size", 1024)
        ov.setdefault("weight_decay", 1e-4)
        components.append({"type": "coral_mlp", "overrides": ov})
    return components


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
        default_factory=lambda: pure_tabnet_stacking_components()
    )
    ensemble_clf_weights: Optional[List[float]] = None
    # Combination mode: "uniform" / "weighted" / "stacking".
    ensemble_clf_mode: str = "stacking"
    # Meta learner for stacking mode (LogReg default; Ridge for regression).
    ensemble_meta_learner_type: str = "logreg"
    # Stacking method: "holdout" (80/20 split) or "oof" (K-fold OOF, slower but better).
    ensemble_stack_method: str = "holdout"
    # Number of inner folds for OOF stacking.
    ensemble_stack_inner_folds: int = 5
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

    # -- XGBoost ordinal classifier (cumulative logits, CORAL-style) --
    # K-1 binary classifiers; each predicts P(y>k). Objective is fixed to
    # binary:logistic inside the model wrapper — don't set it here.
    xgb_ordinal_clf_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
        "reg_alpha": 1.0,
        "random_state": 42,
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
    # TabNet classification — tuned via Optuna TPE 60 trials (3-fold best 0.8575).
    tabnet_clf_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_d": 29,
        "n_a": 22,
        "n_steps": 6,
        "gamma": 1.7146,
        "lambda_sparse": 0.007308,
        "max_epochs": 129,
        "patience": 20,
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

    # -- FT-Transformer (rtdl_revisiting_models) --
    # n_blocks picks a preset (1..6) that sets d_block/dropouts/etc. 3 blocks
    # is the paper's recommended starting point for mid-sized tabular data.
    ft_transformer_clf_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_blocks": 2,
        "max_epochs": 30,
        "patience": 8,
        "batch_size": 2048,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "seed": 42,
    })
    ft_transformer_reg_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_blocks": 3,
        "max_epochs": 100,
        "patience": 15,
        "batch_size": 1024,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "seed": 42,
    })

    # -- CORN-MLP (ordinal MLP using coral-pytorch) --
    # Neural ordinal classifier; brings ordinal-aware error pattern from a
    # non-tree architecture. ~80 epochs on CPU is ~10 min per fold.
    coral_mlp_clf_params: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_sizes": [256, 128, 64],
        "dropout": 0.2,
        "max_epochs": 80,
        "patience": 12,
        "batch_size": 1024,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "seed": 42,
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
                    "stack_method": self.models.ensemble_stack_method,
                    "stack_inner_folds": self.models.ensemble_stack_inner_folds,
                }
            return {
                "components": list(self.models.ensemble_reg_components),
                "weights": self.models.ensemble_reg_weights,
                "mode": self.models.ensemble_reg_mode,
                "meta_learner_type": self.models.ensemble_meta_learner_type,
                "stack_method": self.models.ensemble_stack_method,
                "stack_inner_folds": self.models.ensemble_stack_inner_folds,
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
            (TaskType.CLASSIFICATION, "ft_transformer"): self.models.ft_transformer_clf_params,
            (TaskType.REGRESSION, "ft_transformer"): self.models.ft_transformer_reg_params,
            (TaskType.CLASSIFICATION, "coral_mlp"): self.models.coral_mlp_clf_params,
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
            (TaskType.CLASSIFICATION, "ft_transformer"): self.models.ft_transformer_clf_params,
            (TaskType.REGRESSION, "ft_transformer"): self.models.ft_transformer_reg_params,
            (TaskType.CLASSIFICATION, "coral_mlp"): self.models.coral_mlp_clf_params,
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
