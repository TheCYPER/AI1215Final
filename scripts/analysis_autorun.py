"""Automated analysis suite: error analysis + attention analysis + learning curve.

Runs all three analyses sequentially and writes a combined summary.

Usage:
    source .venv/bin/activate
    python -u scripts/analysis_autorun.py 2>&1 | tee /tmp/analysis_run.log
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import Config, TaskType

OUT_DIR = "outputs/analysis"


def step1_error_analysis(config: Config) -> dict:
    """Run 5-fold CV with OOF prediction collection + error analysis."""
    print(f"\n{'='*60}")
    print(f"[{datetime.now():%H:%M:%S}] Step 1: Error Analysis (5-fold CV)")
    print(f"{'='*60}")
    ts = time.time()

    from training.cross_validator import CrossValidator
    cv = CrossValidator(config)
    results = cv.run()

    elapsed = time.time() - ts
    print(f"[{datetime.now():%H:%M:%S}] Step 1 done in {elapsed/60:.1f} min")
    print(f"  CV accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
    return results


def step3_attention_analysis(config: Config) -> dict:
    """Train single TabNet and analyze attention masks."""
    print(f"\n{'='*60}")
    print(f"[{datetime.now():%H:%M:%S}] Step 3: TabNet Attention Analysis")
    print(f"{'='*60}")
    ts = time.time()

    import numpy as np
    import pandas as pd
    from data_cleaning.column_types import infer_column_types
    from feature_engineering.preprocessor import (
        build_preprocessor,
        get_categorical_feature_indices,
        get_feature_names,
    )
    from modeling.tabnet_model import TabNetModel
    from analysis.attention_analyzer import analyze_attention

    # Load data
    df = pd.read_csv(config.paths.train_csv)
    target_col = config.get_target()
    drop = [c for c in config.columns.targets if c in df.columns]
    X = df.drop(columns=drop)
    y = df[target_col]

    # Preprocess
    num_cols, cat_cols = infer_column_types(
        X, targets=config.columns.targets,
        forced_categorical=config.columns.forced_categorical,
        drop_columns=config.columns.drop_columns,
    )
    preprocessor = build_preprocessor(
        numeric_cols=num_cols, categorical_cols=cat_cols,
        freq_encoding_cols=config.features.freq_encoding_cols,
        log_transform_cols=config.features.log_transform_cols,
        enable_credit_features=config.features.enable_credit_features,
        target_encoding_cols=config.features.target_encoding_cols,
        native_categorical=config.features.native_categorical_for_lgbm,
    )
    X_t = preprocessor.fit_transform(X, y)
    cat_indices = get_categorical_feature_indices(preprocessor)
    feature_names = get_feature_names(preprocessor)

    # Train single TabNet
    tabnet_params = dict(config.models.tabnet_clf_params)
    model = TabNetModel(config=tabnet_params, task_type=TaskType.CLASSIFICATION)
    model.build_model(num_classes=config.training.n_classes)
    fit_kwargs = {}
    if cat_indices:
        fit_kwargs["categorical_feature"] = cat_indices
    model.fit(X_t, y.values, **fit_kwargs)

    # Analyze
    results = analyze_attention(
        model=model, X=X_t, y=y.values,
        feature_names=feature_names, out_dir=OUT_DIR,
    )

    elapsed = time.time() - ts
    print(f"[{datetime.now():%H:%M:%S}] Step 3 done in {elapsed/60:.1f} min")
    top3 = results["global_ranking"][:3]
    print(f"  Top 3 features: {[r['feature'] for r in top3]}")
    return results


def step5_learning_curve(config: Config) -> dict:
    """Run learning curve with TabNet at multiple data fractions."""
    print(f"\n{'='*60}")
    print(f"[{datetime.now():%H:%M:%S}] Step 5: Learning Curve")
    print(f"{'='*60}")
    ts = time.time()

    from analysis.learning_curve import run_learning_curve

    # Use single TabNet (not ensemble) for learning curve
    config.models.clf_model_type = "tabnet"
    results = run_learning_curve(config, out_dir=OUT_DIR)

    elapsed = time.time() - ts
    print(f"[{datetime.now():%H:%M:%S}] Step 5 done in {elapsed/60:.1f} min")
    for r in results["learning_curve"]:
        print(f"  {r['fraction']:.0%} ({r['n_samples']}): train={r['train_acc_mean']:.4f} val={r['val_acc_mean']:.4f}")
    return results


def write_summary(step1_res, step3_res, step5_res):
    """Write combined markdown summary."""
    out = Path(OUT_DIR)
    lines = [
        "# Analysis Summary",
        f"\nGenerated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
    ]

    # Error analysis
    lines.append("## 1. Error Analysis")
    if step1_res:
        lines.append(f"- CV accuracy: {step1_res.get('accuracy_mean', 'N/A')}")
        lines.append("- See: confusion_matrix.png, error_analysis.json, hard_examples.csv")
    else:
        lines.append("- SKIPPED or FAILED")

    lines.append("")

    # Attention analysis
    lines.append("## 3. TabNet Attention")
    if step3_res:
        lines.append("### Top 10 Features by Attention")
        lines.append("| Rank | Feature | Importance |")
        lines.append("|------|---------|------------|")
        for r in step3_res.get("global_ranking", [])[:10]:
            lines.append(f"| {r['rank']} | {r['feature']} | {r['importance']:.4f} |")
        lines.append("")
        lines.append("### Attention Shift (correct vs wrong predictions)")
        for s in step3_res.get("attention_shift_correct_vs_wrong", [])[:5]:
            lines.append(f"- **{s['feature']}**: correct={s['correct_attn']:.4f} wrong={s['wrong_attn']:.4f} (Δ={s['diff']:+.4f})")
        lines.append("- See: feature_importance.png, attention_shift.png")
    else:
        lines.append("- SKIPPED or FAILED")

    lines.append("")

    # Learning curve
    lines.append("## 5. Learning Curve")
    if step5_res:
        lines.append("| Fraction | Samples | Train Acc | Val Acc | Gap |")
        lines.append("|----------|---------|-----------|---------|-----|")
        for r in step5_res.get("learning_curve", []):
            gap = r["train_acc_mean"] - r["val_acc_mean"]
            lines.append(
                f"| {r['fraction']:.0%} | {r['n_samples']} | "
                f"{r['train_acc_mean']:.4f} | {r['val_acc_mean']:.4f} | {gap:.4f} |"
            )
        lines.append("- See: learning_curve.png")
    else:
        lines.append("- SKIPPED or FAILED")

    with open(out / "summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary written to {out / 'summary.md'}")


def main():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = Config()
    config.training.task_type = TaskType.CLASSIFICATION

    print(f"Analysis suite started at {datetime.now():%Y-%m-%d %H:%M:%S}")

    step1_res = None
    step3_res = None
    step5_res = None

    try:
        step1_res = step1_error_analysis(config)
    except Exception as e:
        print(f"[ERROR] Step 1 failed: {e}")
        import traceback; traceback.print_exc()

    try:
        step3_res = step3_attention_analysis(config)
    except Exception as e:
        print(f"[ERROR] Step 3 failed: {e}")
        import traceback; traceback.print_exc()

    try:
        # Reset to single TabNet for learning curve
        config_lc = Config()
        config_lc.training.task_type = TaskType.CLASSIFICATION
        config_lc.models.clf_model_type = "tabnet"
        step5_res = step5_learning_curve(config_lc)
    except Exception as e:
        print(f"[ERROR] Step 5 failed: {e}")
        import traceback; traceback.print_exc()

    write_summary(step1_res, step3_res, step5_res)
    print(f"\nAll done at {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
