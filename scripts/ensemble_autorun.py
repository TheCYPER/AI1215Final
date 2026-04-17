"""Automated ensemble configuration sweep.

Runs multiple heterogeneous stacking configs back-to-back, each with 5-fold CV.
Writes results to /tmp/ensemble_sweep/results.md for human review.

Usage:
    source .venv/bin/activate
    python -u scripts/ensemble_autorun.py 2>&1 | tee /tmp/ensemble_sweep/run.log
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    Config,
    TaskType,
    _TPE_CENTER,
    _RANDOM_CENTER,
    cluster_catboost_variants,
    middle_catboost_variants,
)

# ---------------------------------------------------------------------------
# TabNet overrides: tuned best params (row #24) with seed variations
# ---------------------------------------------------------------------------
_TABNET_TUNED_CENTER = {
    "n_d": 29,
    "n_a": 22,
    "n_steps": 6,
    "gamma": 1.7146,
    "lambda_sparse": 0.007308,
    "max_epochs": 129,
    "patience": 20,
    "batch_size": 1024,
    "virtual_batch_size": 128,
    "verbose": 0,
}

# "Wild" TabNet configs — deliberately different from tuned center
_TABNET_WILD_CONFIGS = [
    {  # wider + shallower attention
        "n_d": 48, "n_a": 48, "n_steps": 3, "gamma": 1.2,
        "lambda_sparse": 0.001, "max_epochs": 100, "patience": 15,
        "batch_size": 512, "virtual_batch_size": 64, "verbose": 0,
    },
    {  # narrow + deep attention
        "n_d": 16, "n_a": 16, "n_steps": 8, "gamma": 1.8,
        "lambda_sparse": 0.01, "max_epochs": 150, "patience": 20,
        "batch_size": 2048, "virtual_batch_size": 256, "verbose": 0,
    },
    {  # mid but high sparsity
        "n_d": 32, "n_a": 24, "n_steps": 5, "gamma": 1.5,
        "lambda_sparse": 0.05, "max_epochs": 120, "patience": 18,
        "batch_size": 1024, "virtual_batch_size": 128, "verbose": 0,
    },
]

# MLP tuned best (from Phase III tune)
_MLP_TUNED_OVERRIDES = {
    "hidden_layer_sizes": (158, 79),
    "alpha": 0.0898,
    "learning_rate_init": 0.00855,
    "batch_size": 66,
    "max_iter": 400,
}


def tabnet_tuned_variants(n: int, seed_start: int = 100) -> list:
    """N TabNet variants with jittered seed around tuned center."""
    variants = []
    for i in range(n):
        overrides = dict(_TABNET_TUNED_CENTER)
        overrides["seed"] = seed_start + i
        variants.append({"type": "tabnet", "overrides": overrides})
    return variants


def tabnet_wild_variants() -> list:
    """3 deliberately different TabNet configs."""
    variants = []
    for i, cfg in enumerate(_TABNET_WILD_CONFIGS):
        overrides = dict(cfg)
        overrides["seed"] = 300 + i
        variants.append({"type": "tabnet", "overrides": overrides})
    return variants


def catboost_components(n_tpe: int, n_random: int, n_middle: int,
                        iter_cap: int = 1000) -> list:
    """CatBoost variants from the 3 known optima regions."""
    comps = []
    if n_tpe > 0:
        comps.extend(cluster_catboost_variants(
            _TPE_CENTER, n=n_tpe, jitter_pct=0.2,
            seed=42, seed_offset=200, iter_cap=iter_cap,
        ))
    if n_random > 0:
        comps.extend(cluster_catboost_variants(
            _RANDOM_CENTER, n=n_random, jitter_pct=0.2,
            seed=142, seed_offset=200 + n_tpe, iter_cap=iter_cap,
        ))
    if n_middle > 0:
        comps.extend(middle_catboost_variants(
            n=n_middle, seed=300, seed_offset=200 + n_tpe + n_random,
            iter_cap=iter_cap,
        ))
    return comps


def mlp_variants(n: int) -> list:
    """MLP variants: first is tuned, rest get seed jitter."""
    variants = []
    for i in range(n):
        overrides = dict(_MLP_TUNED_OVERRIDES)
        overrides["random_state"] = 42 + i
        variants.append({"type": "mlp", "overrides": overrides})
    return variants


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------
CONFIGS = {
    "A_user_spec": {
        "desc": "User spec: 8 TabNet (5 tuned + 3 wild) + 7 CatBoost (4T+2R+1M) + 2 MLP = 17 bases",
        "components": (
            tabnet_tuned_variants(5, seed_start=100)
            + tabnet_wild_variants()
            + catboost_components(n_tpe=4, n_random=2, n_middle=1)
            + mlp_variants(2)
        ),
    },
    "B_tabnet_heavy": {
        "desc": "TabNet-heavy: 10 TabNet (7 tuned + 3 wild) + 3 CatBoost (2T+1R) + 1 MLP = 14 bases",
        "components": (
            tabnet_tuned_variants(7, seed_start=100)
            + tabnet_wild_variants()
            + catboost_components(n_tpe=2, n_random=1, n_middle=0)
            + mlp_variants(1)
        ),
    },
    "C_balanced_slim": {
        "desc": "Balanced slim: 4 TabNet (3 tuned + 1 wild) + 4 CatBoost (2T+1R+1M) + 2 MLP = 10 bases",
        "components": (
            tabnet_tuned_variants(3, seed_start=100)
            + [tabnet_wild_variants()[0]]  # just the wide-shallow one
            + catboost_components(n_tpe=2, n_random=1, n_middle=1)
            + mlp_variants(2)
        ),
    },
    "D_catboost_anchor": {
        "desc": "CatBoost anchor: 3 TabNet (2 tuned + 1 wild) + 12 CatBoost (6T+4R+2M) + 1 MLP = 16 bases",
        "components": (
            tabnet_tuned_variants(2, seed_start=100)
            + [tabnet_wild_variants()[1]]  # narrow-deep
            + catboost_components(n_tpe=6, n_random=4, n_middle=2)
            + mlp_variants(1)
        ),
    },
    "E_kitchen_sink": {
        "desc": "Kitchen sink: 5 TabNet + 5 CatBoost + 2 MLP + 1 LGBM + 1 XGB + 1 LogRegPoly = 15 bases",
        "components": (
            tabnet_tuned_variants(4, seed_start=100)
            + [tabnet_wild_variants()[2]]  # mid+high-sparsity
            + catboost_components(n_tpe=3, n_random=1, n_middle=1)
            + mlp_variants(2)
            + [{"type": "lightgbm", "overrides": {}}]
            + [{"type": "xgboost", "overrides": {}}]
            + [{"type": "logreg_poly", "overrides": {}}]
        ),
    },
    "F_pure_tabnet": {
        "desc": "Pure TabNet ensemble: 12 TabNet (8 tuned + 3 wild + 1 baseline-params) = 12 bases",
        "components": (
            tabnet_tuned_variants(8, seed_start=100)
            + tabnet_wild_variants()
            + [{"type": "tabnet", "overrides": {
                "n_d": 16, "n_a": 16, "n_steps": 4, "gamma": 1.3,
                "lambda_sparse": 1e-3, "max_epochs": 100, "patience": 15,
                "batch_size": 1024, "virtual_batch_size": 128,
                "seed": 999, "verbose": 0,
            }}]
        ),
    },
    "G_duo_strong": {
        "desc": "Duo strong: 6 TabNet (5 tuned + 1 wild) + 6 CatBoost (3T+2R+1M) = 12 bases, no weak models",
        "components": (
            tabnet_tuned_variants(5, seed_start=100)
            + [tabnet_wild_variants()[0]]
            + catboost_components(n_tpe=3, n_random=2, n_middle=1)
        ),
    },
}


def run_one_config(name: str, components: list, desc: str, out_dir: Path) -> dict:
    """Run a single ensemble config through 5-fold CV and return results."""
    from training.cross_validator import CrossValidator

    ts_start = time.time()
    print(f"\n{'='*70}")
    print(f"[{datetime.now():%H:%M:%S}] Config {name}: {desc}")
    print(f"  Components: {len(components)} bases")
    # count by type
    type_counts = {}
    for c in components:
        t = c["type"] if isinstance(c, dict) else c
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  Breakdown: {type_counts}")
    print(f"{'='*70}")

    config = Config()
    config.training.task_type = TaskType.CLASSIFICATION
    config.models.clf_model_type = "ensemble"
    config.models.ensemble_clf_mode = "stacking"
    config.models.ensemble_meta_learner_type = "logreg"
    config.models.ensemble_clf_components = components

    cv = CrossValidator(config)
    results = cv.run()

    elapsed = time.time() - ts_start
    acc_mean = results["accuracy_mean"]
    acc_std = results["accuracy_std"]
    folds = results["accuracy_per_fold"]

    print(f"\n[{datetime.now():%H:%M:%S}] Config {name} DONE in {elapsed/60:.1f} min")
    print(f"  Result: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  Folds: {[f'{f:.4f}' for f in folds]}")

    # Save per-config JSON
    result_data = {
        "name": name,
        "desc": desc,
        "n_bases": len(components),
        "type_counts": type_counts,
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "accuracy_per_fold": folds,
        "elapsed_min": round(elapsed / 60, 1),
    }
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump(result_data, f, indent=2)

    return result_data


def main():
    out_dir = Path("/tmp/ensemble_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ensemble sweep started at {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Output dir: {out_dir}")
    print(f"Configs to run: {list(CONFIGS.keys())}")

    all_results = []

    for name, spec in CONFIGS.items():
        try:
            result = run_one_config(
                name, spec["components"], spec["desc"], out_dir,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Config {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "name": name,
                "desc": spec["desc"],
                "error": str(e),
            })

        # Write intermediate results after each config
        _write_results_md(all_results, out_dir)

    print(f"\n{'='*70}")
    print(f"ALL DONE at {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Results: {out_dir / 'results.md'}")


def _write_results_md(results: list, out_dir: Path):
    """Write/overwrite results.md with current accumulated results."""
    lines = [
        "# Ensemble Sweep Results",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "Baselines: TabNet tuned **0.8573 ± 0.0049** (row #24) | "
        "30-CatBoost stacking **0.8331 ± 0.0016** (row #15)",
        "",
        "| Config | Description | #Bases | Breakdown | Mean Acc | Std | "
        "vs TabNet | vs 30-Stack | Time |",
        "|--------|-------------|--------|-----------|----------|-----|"
        "----------|-------------|------|",
    ]

    for r in results:
        if "error" in r:
            lines.append(
                f"| {r['name']} | {r['desc'][:50]}… | — | — | "
                f"**ERROR** | — | — | — | — |"
            )
            continue
        delta_tabnet = r["accuracy_mean"] - 0.8573
        delta_stack = r["accuracy_mean"] - 0.8331
        lines.append(
            f"| {r['name']} | {r['desc'][:60]}… | {r['n_bases']} | "
            f"{r['type_counts']} | **{r['accuracy_mean']:.4f}** | "
            f"{r['accuracy_std']:.4f} | {delta_tabnet:+.4f} | "
            f"{delta_stack:+.4f} | {r['elapsed_min']}m |"
        )

    lines.extend([
        "",
        "## Per-fold details",
        "",
    ])
    for r in results:
        if "error" in r:
            continue
        folds_str = " | ".join(f"{f:.4f}" for f in r["accuracy_per_fold"])
        lines.append(f"**{r['name']}**: {folds_str}")

    lines.extend([
        "",
        "## Config Details",
        "",
    ])
    for r in results:
        if "error" in r:
            lines.append(f"### {r['name']} — ERROR: {r['error']}")
            continue
        lines.append(f"### {r['name']}")
        lines.append(f"- {r['desc']}")
        lines.append(f"- {r['n_bases']} bases: {r['type_counts']}")
        lines.append(f"- Result: **{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}**")
        lines.append("")

    with open(out_dir / "results.md", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
