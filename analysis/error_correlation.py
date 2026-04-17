"""Error correlation analysis across base models.

Answers the question: *which models disagree on errors?* A stacking
ensemble only benefits from adding a model if its mistakes are made on
different samples than the models already in the pool.

Metrics computed for every pair (A, B):
- Cohen's kappa on predictions (high kappa = high agreement = redundant)
- Error overlap = |A_wrong ∩ B_wrong| / |A_wrong ∪ B_wrong|  (Jaccard)
- Yule's Q = (ad - bc) / (ad + bc), where
    a = both correct, b = A wrong B correct, c = A correct B wrong,
    d = both wrong.
  Q is the classic ensemble-diversity measure: +1 = perfect agreement,
  0 = independent, -1 = perfect disagreement. Lower Q → better ensemble value.

Inputs: per-model OOF predictions saved by cross_validator under
`outputs/oof/<model_type>.npz` (keys: y_true, y_pred, y_proba, indices).

Output: `outputs/analysis/error_correlation.md` + Q heatmap PNG.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger("analysis.error_correlation")


def _load_oof(oof_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for path in sorted(glob.glob(os.path.join(oof_dir, "*.npz"))):
        name = os.path.splitext(os.path.basename(path))[0]
        data = np.load(path)
        out[name] = {
            "y_true": data["y_true"],
            "y_pred": data["y_pred"],
            "y_proba": data["y_proba"],
            "indices": data["indices"],
        }
    return out


def _align_to_indices(
    preds: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Re-order every model's predictions to the union of sample indices.

    Different runs may have visited samples in different fold orderings;
    we sort each by its original-dataframe index before comparing.
    """
    if not preds:
        raise ValueError("No OOF files found")
    first = next(iter(preds.values()))
    ref_idx = np.sort(first["indices"])
    aligned: Dict[str, np.ndarray] = {}
    aligned_true: np.ndarray | None = None
    for name, d in preds.items():
        order = np.argsort(d["indices"])
        if not np.array_equal(d["indices"][order], ref_idx):
            raise ValueError(
                f"Model {name} has a different sample set than the reference"
            )
        aligned[name] = d["y_pred"][order]
        if aligned_true is None:
            aligned_true = d["y_true"][order]
    assert aligned_true is not None
    return aligned_true, aligned


def _pair_metrics(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray
) -> Dict[str, float]:
    wrong_a = pred_a != y_true
    wrong_b = pred_b != y_true
    both_wrong = wrong_a & wrong_b
    either_wrong = wrong_a | wrong_b
    overlap = (both_wrong.sum() / max(1, either_wrong.sum()))

    # Cohen's kappa between the two prediction vectors
    from sklearn.metrics import cohen_kappa_score
    kappa = float(cohen_kappa_score(pred_a, pred_b))

    # Yule's Q on correct/wrong agreement
    a = int((~wrong_a & ~wrong_b).sum())
    b = int((wrong_a & ~wrong_b).sum())
    c = int((~wrong_a & wrong_b).sum())
    d = int(both_wrong.sum())
    ad = a * d
    bc = b * c
    q = float((ad - bc) / max(1e-12, (ad + bc)))

    return {
        "kappa": kappa,
        "error_overlap": float(overlap),
        "yule_q": q,
        "both_wrong": d,
        "a_wrong_only": b,
        "b_wrong_only": c,
        "both_correct": a,
    }


def run_error_correlation(
    oof_dir: str = "outputs/oof",
    out_dir: str = "outputs/analysis",
):
    os.makedirs(out_dir, exist_ok=True)
    preds = _load_oof(oof_dir)
    if len(preds) < 2:
        raise RuntimeError(
            f"Need >=2 models in {oof_dir}; found {list(preds.keys())}"
        )
    logger.info(f"Loaded OOF for {len(preds)} models: {list(preds.keys())}")

    y_true, aligned = _align_to_indices(preds)
    names = sorted(aligned.keys())

    # Per-model accuracy
    per_model_acc = {n: float((aligned[n] == y_true).mean()) for n in names}

    # Pairwise metrics
    pair_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for i, a in enumerate(names):
        pair_results[a] = {}
        for j, b in enumerate(names):
            if a == b:
                continue
            pair_results[a][b] = _pair_metrics(y_true, aligned[a], aligned[b])

    # Markdown report
    report_lines = ["# Error correlation across base models", ""]
    report_lines.append("## Per-model OOF accuracy")
    report_lines.append("")
    report_lines.append("| Model | Accuracy |")
    report_lines.append("|-------|----------|")
    for n in sorted(per_model_acc, key=lambda k: -per_model_acc[k]):
        report_lines.append(f"| {n} | {per_model_acc[n]:.4f} |")
    report_lines.append("")

    report_lines.append("## Yule's Q matrix (lower = better ensemble diversity)")
    report_lines.append("")
    header = "| | " + " | ".join(names) + " |"
    sep = "|" + "---|" * (len(names) + 1)
    report_lines.append(header)
    report_lines.append(sep)
    for a in names:
        row = [a]
        for b in names:
            if a == b:
                row.append("—")
            else:
                row.append(f"{pair_results[a][b]['yule_q']:.3f}")
        report_lines.append("| " + " | ".join(row) + " |")
    report_lines.append("")

    report_lines.append("## Error-overlap Jaccard (lower = more complementary)")
    report_lines.append("")
    report_lines.append(header)
    report_lines.append(sep)
    for a in names:
        row = [a]
        for b in names:
            if a == b:
                row.append("—")
            else:
                row.append(f"{pair_results[a][b]['error_overlap']:.3f}")
        report_lines.append("| " + " | ".join(row) + " |")
    report_lines.append("")

    # Recommend most-complementary-to-each model pairs
    report_lines.append("## Most complementary partner per model (lowest Yule's Q)")
    report_lines.append("")
    report_lines.append("| Model | Best partner | Yule's Q | Error overlap |")
    report_lines.append("|-------|--------------|----------|---------------|")
    for a in names:
        others = [(b, pair_results[a][b]) for b in names if b != a]
        best = min(others, key=lambda kv: kv[1]["yule_q"])
        report_lines.append(
            f"| {a} | {best[0]} | {best[1]['yule_q']:.3f} | "
            f"{best[1]['error_overlap']:.3f} |"
        )
    report_lines.append("")

    md_path = os.path.join(out_dir, "error_correlation.md")
    with open(md_path, "w") as f:
        f.write("\n".join(report_lines))
    logger.info(f"Report saved: {md_path}")

    # JSON dump for programmatic access
    json_path = os.path.join(out_dir, "error_correlation.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "per_model_acc": per_model_acc,
                "pairs": pair_results,
            },
            f,
            indent=2,
        )
    logger.info(f"Raw metrics saved: {json_path}")

    # Heatmap of Yule's Q
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(names)
        mat = np.zeros((n, n), dtype=float)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i == j:
                    mat[i, j] = np.nan
                else:
                    mat[i, j] = pair_results[a][b]["yule_q"]
        fig, ax = plt.subplots(figsize=(1.2 * n + 2, 1.1 * n + 1))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
        for i in range(n):
            for j in range(n):
                if i != j:
                    ax.text(
                        j, i, f"{mat[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if abs(mat[i, j]) > 0.5 else "black",
                        fontsize=8,
                    )
        plt.colorbar(im, label="Yule's Q")
        plt.title("Error diversity (Yule's Q) — lower is better for ensembling")
        plt.tight_layout()
        png_path = os.path.join(out_dir, "error_correlation_heatmap.png")
        plt.savefig(png_path, dpi=100)
        plt.close()
        logger.info(f"Heatmap saved: {png_path}")
    except ImportError:
        logger.warning("matplotlib not available — skipping heatmap")

    return pair_results, per_model_acc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_error_correlation()
