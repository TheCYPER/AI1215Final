"""Error analysis: confusion matrix, per-class metrics, confidence analysis, hard examples."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("analysis.error")


def aggregate_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5,
) -> np.ndarray:
    """Build confusion matrix from OOF predictions."""
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def per_class_report(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
    n_classes: int = 5,
) -> Dict:
    """Per-class precision/recall/F1 + top confusion pairs."""
    from sklearn.metrics import classification_report

    report = classification_report(
        y_true, y_pred, labels=list(range(n_classes)), output_dict=True,
    )
    cm = aggregate_confusion_matrix(y_true, y_pred, n_classes)

    # For each class, find the class it's most often confused with
    confusion_pairs = {}
    for i in range(n_classes):
        row = cm[i].copy()
        row[i] = 0  # ignore correct predictions
        if row.sum() > 0:
            worst_j = int(np.argmax(row))
            confusion_pairs[str(i)] = {
                "most_confused_with": worst_j,
                "count": int(row[worst_j]),
                "total_errors": int(row.sum()),
            }

    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_pairs": confusion_pairs,
    }


def confidence_analysis(
    y_true: np.ndarray, y_proba: np.ndarray,
    bins: int = 10,
) -> Dict:
    """Bin predictions by max softmax probability, check accuracy per bin."""
    max_proba = np.max(y_proba, axis=1)
    y_pred = np.argmax(y_proba, axis=1)
    correct = (y_pred == y_true).astype(int)

    bin_edges = np.linspace(0, 1, bins + 1)
    bin_results = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (max_proba >= lo) & (max_proba < hi)
        if hi == 1.0:
            mask = mask | (max_proba == 1.0)
        n = int(mask.sum())
        if n == 0:
            bin_results.append({
                "bin": f"[{lo:.1f}, {hi:.1f})", "n": 0, "accuracy": None,
            })
            continue
        acc = float(correct[mask].mean())
        bin_results.append({
            "bin": f"[{lo:.1f}, {hi:.1f})",
            "n": n,
            "accuracy": round(acc, 4),
            "mean_confidence": round(float(max_proba[mask].mean()), 4),
        })

    return {
        "bins": bin_results,
        "overall_mean_confidence": round(float(max_proba.mean()), 4),
        "overall_accuracy": round(float(correct.mean()), 4),
    }


def hard_examples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    original_indices: np.ndarray,
    top_n: int = 100,
) -> pd.DataFrame:
    """Find the hardest misclassified samples (highest confidence yet wrong)."""
    wrong_mask = y_pred != y_true
    if wrong_mask.sum() == 0:
        return pd.DataFrame()

    max_proba = np.max(y_proba, axis=1)
    wrong_indices = np.where(wrong_mask)[0]
    wrong_confidences = max_proba[wrong_indices]

    # Sort by confidence descending — high confidence + wrong = hardest
    sorted_idx = np.argsort(-wrong_confidences)[:top_n]
    selected = wrong_indices[sorted_idx]

    rows = []
    for idx in selected:
        rows.append({
            "original_index": int(original_indices[idx]),
            "true_label": int(y_true[idx]),
            "pred_label": int(y_pred[idx]),
            "confidence": round(float(max_proba[idx]), 4),
            "proba": [round(float(p), 4) for p in y_proba[idx]],
        })
    return pd.DataFrame(rows)


def plot_confusion_matrix(
    cm: np.ndarray, out_path: str, title: str = "Confusion Matrix",
):
    """Save confusion matrix as heatmap image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title(f"{title} (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Normalized by row (recall)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="YlOrRd", ax=axes[1])
    axes[1].set_title(f"{title} (row-normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved to {out_path}")


def run_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    original_indices: np.ndarray,
    out_dir: str = "outputs/analysis",
    n_classes: int = 5,
) -> Dict:
    """Run full error analysis suite and save outputs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Confusion matrix
    cm = aggregate_confusion_matrix(y_true, y_pred, n_classes)
    plot_confusion_matrix(cm, str(out / "confusion_matrix.png"))

    # 2. Per-class report
    report = per_class_report(y_true, y_pred, y_proba, n_classes)

    # 3. Confidence analysis
    conf = confidence_analysis(y_true, y_proba)

    # 4. Hard examples
    hard = hard_examples(y_true, y_pred, y_proba, original_indices)
    if not hard.empty:
        hard.to_csv(out / "hard_examples.csv", index=False)

    # Save combined JSON
    results = {
        "confusion_matrix": cm.tolist(),
        "confusion_pairs": report["confusion_pairs"],
        "per_class": report["classification_report"],
        "confidence_analysis": conf,
        "n_hard_examples": len(hard),
    }
    with open(out / "error_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Error analysis saved to {out}")
    return results
