"""Learning curve analysis.

Trains TabNet at different data fractions to diagnose whether the
bottleneck is data quantity (bias) or overfitting (variance).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger("analysis.learning_curve")


def run_learning_curve(
    config,
    fractions: List[float] = None,
    n_splits: int = 3,
    out_dir: str = "outputs/analysis",
) -> Dict:
    """Run learning curve with multiple data fractions."""
    from configs.config import TaskType
    from data_cleaning.column_types import infer_column_types
    from feature_engineering.preprocessor import (
        build_preprocessor,
        get_categorical_feature_indices,
    )
    from modeling import model_factory
    from utils.metrics import compute_classification_metrics

    if fractions is None:
        fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load full data
    df = pd.read_csv(config.paths.train_csv)
    target_col = config.get_target()
    drop = [c for c in config.columns.targets if c in df.columns]
    X_full = df.drop(columns=drop)
    y_full = df[target_col]

    results = []

    for frac in fractions:
        logger.info(f"Learning curve: fraction={frac:.0%}")

        if frac < 1.0:
            n_samples = int(len(X_full) * frac)
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_full), n_samples, replace=False)
            idx.sort()
            X = X_full.iloc[idx].reset_index(drop=True)
            y = y_full.iloc[idx].reset_index(drop=True)
        else:
            X = X_full
            y = y_full

        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        train_accs = []
        val_accs = []

        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Build fresh preprocessor
            num_cols, cat_cols = infer_column_types(
                X_train,
                targets=config.columns.targets,
                forced_categorical=config.columns.forced_categorical,
                drop_columns=config.columns.drop_columns,
            )
            preprocessor = build_preprocessor(
                numeric_cols=num_cols,
                categorical_cols=cat_cols,
                freq_encoding_cols=config.features.freq_encoding_cols,
                log_transform_cols=config.features.log_transform_cols,
                enable_credit_features=config.features.enable_credit_features,
                target_encoding_cols=config.features.target_encoding_cols,
                native_categorical=config.features.native_categorical_for_lgbm,
            )
            X_train_t = preprocessor.fit_transform(X_train, y_train)
            X_val_t = preprocessor.transform(X_val)
            cat_indices = get_categorical_feature_indices(preprocessor)

            # Use single TabNet (not ensemble) for speed
            model = model_factory(config)
            fit_kwargs = {}
            if cat_indices:
                fit_kwargs["categorical_feature"] = cat_indices
            fit_kwargs["eval_set"] = [(X_val_t, y_val.values)]
            model.fit(X_train_t, y_train.values, **fit_kwargs)

            train_pred = model.predict(X_train_t)
            val_pred = model.predict(X_val_t)
            train_acc = float((train_pred == y_train.values).mean())
            val_acc = float((val_pred == y_val.values).mean())

            train_accs.append(train_acc)
            val_accs.append(val_acc)
            logger.info(f"  Fold {fold_idx+1}: train={train_acc:.4f} val={val_acc:.4f}")

        results.append({
            "fraction": frac,
            "n_samples": len(X),
            "train_acc_mean": round(float(np.mean(train_accs)), 4),
            "train_acc_std": round(float(np.std(train_accs)), 4),
            "val_acc_mean": round(float(np.mean(val_accs)), 4),
            "val_acc_std": round(float(np.std(val_accs)), 4),
        })

    # Save
    with open(out / "learning_curve.json", "w") as f:
        json.dump(results, f, indent=2)

    _plot_learning_curve(results, out / "learning_curve.png")
    logger.info(f"Learning curve saved to {out}")
    return {"learning_curve": results}


def _plot_learning_curve(results: list, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fracs = [r["n_samples"] for r in results]
    train_means = [r["train_acc_mean"] for r in results]
    train_stds = [r["train_acc_std"] for r in results]
    val_means = [r["val_acc_mean"] for r in results]
    val_stds = [r["val_acc_std"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(fracs,
                     [m - s for m, s in zip(train_means, train_stds)],
                     [m + s for m, s in zip(train_means, train_stds)],
                     alpha=0.2, color="blue")
    ax.plot(fracs, train_means, "o-", color="blue", label="Train")
    ax.fill_between(fracs,
                     [m - s for m, s in zip(val_means, val_stds)],
                     [m + s for m, s in zip(val_means, val_stds)],
                     alpha=0.2, color="orange")
    ax.plot(fracs, val_means, "o-", color="orange", label="Validation")
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve (TabNet)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
