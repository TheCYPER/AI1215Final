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
from modeling.xgboost_model import XGBoostModel

# Register all available models here
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "xgboost": XGBoostModel,
    # "lightgbm": LightGBMModel,  # uncomment when implemented
    # "catboost": CatBoostModel,
    # "ensemble": EnsembleModel,
}


def model_factory(config: Config) -> BaseModel:
    """
    Create and build a model based on config.

    Reads task_type and model_type from config, looks up the registry,
    instantiates with the correct params, and calls build_model().
    """
    task_type = config.training.task_type
    model_type = config.get_model_type()
    params = config.get_model_params()

    if model_type not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{model_type}'. Available: {available}"
        )

    model_cls = MODEL_REGISTRY[model_type]
    model = model_cls(config=params, task_type=task_type)

    if task_type == TaskType.CLASSIFICATION:
        model.build_model(num_classes=config.training.n_classes)
    else:
        model.build_model()

    return model


__all__ = ["BaseModel", "model_factory", "MODEL_REGISTRY"]
