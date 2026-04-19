"""
Model factory — the single point for model selection.

To add a new model:
  1. Create modeling/my_model.py with MyModel(BaseModel)
  2. Add "mymodel": MyModel to MODEL_REGISTRY below
  3. Add params to configs/config.py ModelConfig
"""

from typing import Dict, Type

from configs.config import Config, TaskType
from modeling.base_model import BaseModel
from modeling.catboost_model import CatBoostModel
from modeling.ensemble_model import EnsembleModel
from modeling.lightgbm_model import LightGBMModel
from modeling.logreg_poly_model import LogRegPolyModel
from modeling.mlp_model import MLPModel
from modeling.xgboost_model import XGBoostModel
from modeling.xgboost_ordinal import XGBoostOrdinalModel

# TabNet is heavy (pulls torch); import-guarded so missing dep doesn't break the registry.
try:
    from modeling.tabnet_model import TabNetModel
    _TABNET_AVAILABLE = True
except ImportError:
    _TABNET_AVAILABLE = False

try:
    from modeling.ft_transformer_model import FTTransformerModel
    _FT_AVAILABLE = True
except ImportError:
    _FT_AVAILABLE = False

try:
    from modeling.coral_mlp_model import CoralMLPModel
    _CORAL_AVAILABLE = True
except ImportError:
    _CORAL_AVAILABLE = False

# Register all available models here
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "xgboost": XGBoostModel,
    "xgboost_ordinal": XGBoostOrdinalModel,
    "lightgbm": LightGBMModel,
    "catboost": CatBoostModel,
    "mlp": MLPModel,
    "logreg_poly": LogRegPolyModel,
    "ensemble": EnsembleModel,
}
if _TABNET_AVAILABLE:
    MODEL_REGISTRY["tabnet"] = TabNetModel
if _FT_AVAILABLE:
    MODEL_REGISTRY["ft_transformer"] = FTTransformerModel
if _CORAL_AVAILABLE:
    MODEL_REGISTRY["coral_mlp"] = CoralMLPModel


def model_factory(config: Config) -> BaseModel:
    """Create and build a model based on config.

    For the "ensemble" model_type, recursively builds each component and
    injects the list of base models into the EnsembleModel wrapper.
    """
    task_type = config.training.task_type
    model_type = config.get_model_type()

    if model_type == "ensemble":
        spec = config.get_model_params()  # {"components": [...], "weights": ...}
        base_models: list = []
        for comp in spec["components"]:
            # Normalize: bare string → {"type": <str>, "overrides": {}}
            if isinstance(comp, str):
                comp_type, overrides = comp, {}
            else:
                comp_type = comp["type"]
                overrides = comp.get("overrides", {})
            if comp_type not in MODEL_REGISTRY or comp_type == "ensemble":
                raise ValueError(f"Invalid ensemble component: {comp_type}")
            base_params = config.get_component_params(comp_type)
            merged = {**base_params, **overrides}
            comp_cls = MODEL_REGISTRY[comp_type]
            comp_model = comp_cls(config=merged, task_type=task_type)
            if task_type == TaskType.CLASSIFICATION:
                comp_model.build_model(num_classes=config.training.n_classes)
            else:
                comp_model.build_model()
            base_models.append(comp_model)
        ensemble = EnsembleModel(
            task_type=task_type,
            base_models=base_models,
            weights=spec.get("weights"),
            mode=spec.get("mode", "uniform"),
            meta_learner_type=spec.get("meta_learner_type", "logreg"),
            stack_method=spec.get("stack_method", "holdout"),
            stack_inner_folds=spec.get("stack_inner_folds", 5),
        )
        if task_type == TaskType.CLASSIFICATION:
            ensemble.build_model(num_classes=config.training.n_classes)
        else:
            ensemble.build_model()
        return ensemble

    if model_type not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{model_type}'. Available: {available}"
        )

    params = config.get_model_params()
    model_cls = MODEL_REGISTRY[model_type]
    model = model_cls(config=params, task_type=task_type)
    if task_type == TaskType.CLASSIFICATION:
        model.build_model(num_classes=config.training.n_classes)
    else:
        model.build_model()
    return model


__all__ = ["BaseModel", "model_factory", "MODEL_REGISTRY"]
