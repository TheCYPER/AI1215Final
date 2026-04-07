"""
Centralized configuration for the CreditSense ML pipeline.

All hyperparameters, column definitions, and paths in one place.
Change model type or params here — the entire pipeline follows.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
    clf_model_type: str = "xgboost"
    reg_model_type: str = "xgboost"

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

    # -- LightGBM classification (placeholder for teammates) --
    lgbm_clf_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "multiclass",
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


@dataclass
class FeatureEngineeringConfig:
    """Feature engineering settings."""
    # High-cardinality columns -> frequency encoding
    freq_encoding_cols: List[str] = field(default_factory=lambda: ["State"])

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
    n_trials: int = 50
    cv_folds: int = 5
    timeout: Optional[int] = 3600

    clf_search_space: Dict[str, Any] = field(default_factory=lambda: {
        "learning_rate": (0.01, 0.3),
        "max_depth": (3, 10),
        "min_child_weight": (1, 20),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.3, 1.0),
        "reg_alpha": (0, 10),
        "reg_lambda": (0, 20),
    })

    reg_search_space: Dict[str, Any] = field(default_factory=lambda: {
        "learning_rate": (0.01, 0.3),
        "max_depth": (3, 10),
        "min_child_weight": (1, 20),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.3, 1.0),
        "reg_alpha": (0, 10),
        "reg_lambda": (0, 20),
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
        param_map = {
            (TaskType.CLASSIFICATION, "xgboost"): self.models.xgb_clf_params,
            (TaskType.REGRESSION, "xgboost"): self.models.xgb_reg_params,
            (TaskType.CLASSIFICATION, "lightgbm"): self.models.lgbm_clf_params,
            (TaskType.REGRESSION, "lightgbm"): self.models.lgbm_reg_params,
        }
        key = (task, model)
        if key not in param_map:
            raise ValueError(f"No params for task={task}, model={model}")
        return param_map[key]

    def get_search_space(self) -> Dict[str, Any]:
        if self.training.task_type == TaskType.CLASSIFICATION:
            return self.tuning.clf_search_space
        return self.tuning.reg_search_space


def load_config(config_path: Optional[str] = None) -> Config:
    """Load config from file or return defaults."""
    if config_path is None:
        return Config()
    raise NotImplementedError("Config file loading not yet implemented")
