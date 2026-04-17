"""TabNet attention mask analysis.

Trains a single TabNet on full training data, extracts attention masks,
and reports which features the model pays most attention to.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger("analysis.attention")


def analyze_attention(
    model,  # fitted TabNetModel
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    out_dir: str = "outputs/analysis",
) -> Dict:
    """Extract and analyze TabNet attention masks."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Get attention masks
    M_explain, _ = model.explain(X)
    # M_explain shape: (n_samples, n_features)

    # Global feature importance (mean attention across all samples)
    global_importance = M_explain.mean(axis=0)

    # Sort by importance
    sorted_idx = np.argsort(-global_importance)
    ranking = []
    for rank, idx in enumerate(sorted_idx):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        ranking.append({
            "rank": rank + 1,
            "feature": name,
            "importance": round(float(global_importance[idx]), 6),
        })

    # Per-class attention (which features matter for each RiskTier)
    per_class = {}
    for cls in sorted(set(y)):
        cls_mask = y == cls
        if cls_mask.sum() == 0:
            continue
        cls_importance = M_explain[cls_mask].mean(axis=0)
        top5_idx = np.argsort(-cls_importance)[:5]
        per_class[int(cls)] = [
            {
                "feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                "importance": round(float(cls_importance[i]), 6),
            }
            for i in top5_idx
        ]

    # Correct vs incorrect attention patterns
    y_pred = np.argmax(model.predict_proba(X), axis=1)
    correct_mask = y_pred == y
    correct_attn = M_explain[correct_mask].mean(axis=0) if correct_mask.sum() > 0 else global_importance
    wrong_attn = M_explain[~correct_mask].mean(axis=0) if (~correct_mask).sum() > 0 else global_importance
    attn_diff = wrong_attn - correct_attn
    biggest_diff_idx = np.argsort(-np.abs(attn_diff))[:10]
    attention_shift = [
        {
            "feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
            "correct_attn": round(float(correct_attn[i]), 6),
            "wrong_attn": round(float(wrong_attn[i]), 6),
            "diff": round(float(attn_diff[i]), 6),
        }
        for i in biggest_diff_idx
    ]

    results = {
        "global_ranking": ranking[:20],  # top 20
        "per_class_top5": per_class,
        "attention_shift_correct_vs_wrong": attention_shift,
        "n_features": int(M_explain.shape[1]),
        "n_samples": int(M_explain.shape[0]),
    }

    # Save
    with open(out / "attention_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    _plot_importance(ranking[:20], out / "feature_importance.png")
    _plot_attention_shift(attention_shift, out / "attention_shift.png")

    logger.info(f"Attention analysis saved to {out}")
    return results


def _plot_importance(ranking: list, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    names = [r["feature"] for r in ranking]
    vals = [r["importance"] for r in ranking]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(names)), vals, color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Attention")
    ax.set_title("TabNet Global Feature Importance (Top 20)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_attention_shift(shifts: list, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    names = [s["feature"] for s in shifts]
    correct = [s["correct_attn"] for s in shifts]
    wrong = [s["wrong_attn"] for s in shifts]

    x = range(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width/2 for i in x], correct, width, label="Correct pred", color="green", alpha=0.7)
    ax.bar([i + width/2 for i in x], wrong, width, label="Wrong pred", color="red", alpha=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Attention")
    ax.set_title("Attention Shift: Correct vs Wrong Predictions")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
