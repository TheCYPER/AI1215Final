"""Regression ensemble autorun: 4 stacking configs (R-A, R-B, R-C, R-D).

Compares ensemble strategies for the InterestRate regression task. All four
configs use stacking mode (80/20 holdout, Ridge meta) and 5-fold outer CV
so they are directly comparable to the single-model 5-fold scores already
in experiments.md.

Configs:
- R-A: 30 diverse CatBoost (10 around regression TPE-tuned center +
       10 around a "shallow" center + 10 middle-region random).
       Clones the cls row #15 design that gave +0.0068.
- R-B: 8 CatBoost + 4 XGB + 4 LGBM. Tree-only heterogeneous, no TabNet.
       Tests whether mixing tree families works when single-model spread
       is tight (0.829-0.837).
- R-C: 10 CatBoost + 2 XGB + 2 LGBM + 2 TabNet. Adds 2 TabNet to test
       whether NN diversity helps OR drags (TabNet is the weakest single
       model at 0.8272 5-fold per row #39).
- R-D: 15 CatBoost (5 TPE + 5 shallow + 5 middle) + 3 XGB + 3 LGBM.
       Maximum tree spread, no TabNet. Tests "more is better" hypothesis.

Baseline for comparison: CatBoost default 5-fold = 0.8367 ± 0.0148 (row #34).

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        nohup python -u scripts/regression_ensemble_autorun.py \\
        > /tmp/regression_ensemble_autorun.log 2>&1 &

Writes:
- /tmp/regression_ensemble_autorun.log (progress)
- /tmp/regression_ensemble/results.json (machine-readable)
- /tmp/regression_ensemble/results.md (human-readable summary)
"""

from __future__ import annotations

import json
import os
import random as _random
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

torch.set_num_threads(1)

from configs.config import (  # noqa: E402
    Config,
    TaskType,
    cluster_catboost_variants,
    middle_catboost_variants,
)


# ---------------------------------------------------------------------------
# Regression-specific cluster centers (different from cls _TPE_CENTER!)
# ---------------------------------------------------------------------------

# From row #35 (CatBoost reg TPE 40-trial widened tune best, applied score 0.8345):
REG_TPE_CENTER = {
    "depth": 6,
    "learning_rate": 0.055,
    "l2_leaf_reg": 51.9,
    "subsample": 0.71,
    "rsm": 0.76,
    "random_strength": 6.6,
    "border_count": 182,
    "leaf_estimation_iterations": 11,
}

# Hand-picked "shallow + lighter reg" diversity center. Mirrors the cls
# pattern where Random search found a depth=4 / lower-reg optimum that
# decorrelated with TPE's depth=6 region.
REG_SHALLOW_CENTER = {
    "depth": 4,
    "learning_rate": 0.08,
    "l2_leaf_reg": 25.0,
    "subsample": 0.85,
    "rsm": 0.85,
    "random_strength": 3.0,
    "border_count": 128,
    "leaf_estimation_iterations": 8,
}

# TabNet reg widened-tuned base (row #39 best). Each TabNet variant differs
# only by `seed` so they're cheap diversity (decorrelation from random init,
# not architecture).
TABNET_TUNED_BASE = {
    "n_d": 32,
    "n_a": 19,
    "n_steps": 9,
    "gamma": 2.41,
    "lambda_sparse": 0.044,
    "max_epochs": 146,
    "patience": 37,
    "batch_size": 1587,
}

# Default training budget for ensemble bases. Slightly above TPE-tuned best
# (775) to allow shallow variants more room without runtime explosion.
ITER_CAP = 800


# ---------------------------------------------------------------------------
# Variant generators
# ---------------------------------------------------------------------------

def reg_diverse_catboost(n_per_cluster: int, seed_base: int = 42, iter_cap: int = ITER_CAP) -> list:
    """30 CatBoosts: TPE-cluster + shallow-cluster + middle-region."""
    return (
        cluster_catboost_variants(
            REG_TPE_CENTER, n=n_per_cluster, jitter_pct=0.2,
            seed=seed_base, seed_offset=200, iter_cap=iter_cap,
        )
        + cluster_catboost_variants(
            REG_SHALLOW_CENTER, n=n_per_cluster, jitter_pct=0.2,
            seed=seed_base + 100, seed_offset=200 + n_per_cluster, iter_cap=iter_cap,
        )
        + middle_catboost_variants(
            n=n_per_cluster, seed=seed_base + 200,
            seed_offset=200 + 2 * n_per_cluster, iter_cap=iter_cap,
        )
    )


def xgb_variants(n: int, seed_base: int = 500) -> list:
    rng = _random.Random(seed_base)
    out = []
    for i in range(n):
        out.append({"type": "xgboost", "overrides": {
            "max_depth": rng.choice([4, 5, 6, 7, 8]),
            "learning_rate": round(rng.uniform(0.04, 0.10), 4),
            "subsample": round(rng.uniform(0.7, 1.0), 3),
            "colsample_bytree": round(rng.uniform(0.7, 1.0), 3),
            "reg_lambda": round(rng.uniform(1.0, 10.0), 2),
            "n_estimators": 700,
            "random_state": seed_base + i,
        }})
    return out


def lgbm_variants(n: int, seed_base: int = 600) -> list:
    rng = _random.Random(seed_base)
    out = []
    for i in range(n):
        out.append({"type": "lightgbm", "overrides": {
            "num_leaves": rng.choice([31, 63, 127]),
            "max_depth": rng.choice([-1, 6, 8, 10]),
            "learning_rate": round(rng.uniform(0.04, 0.10), 4),
            "feature_fraction": round(rng.uniform(0.7, 1.0), 3),
            "bagging_fraction": round(rng.uniform(0.7, 1.0), 3),
            "bagging_freq": 5,
            "lambda_l2": round(rng.uniform(0.1, 5.0), 2),
            "n_estimators": 700,
            "random_state": seed_base + i,
        }})
    return out


def tabnet_variants(n: int, seed_base: int = 700) -> list:
    """Use widened-tuned TabNet base + seed perturbation only."""
    return [{"type": "tabnet", "overrides": {
        **TABNET_TUNED_BASE,
        "seed": seed_base + i,
    }} for i in range(n)]


# ---------------------------------------------------------------------------
# Configs to run
# ---------------------------------------------------------------------------

CONFIGS = [
    {
        "name": "R-A_30_diverse_catboost",
        "desc": "30 CatBoost (10 TPE + 10 shallow + 10 middle) stacking",
        "components": reg_diverse_catboost(n_per_cluster=10, seed_base=42),
    },
    {
        # Originally R-B had +4 LGBM but LightGBM segfaults after heavy CatBoost
        # fit on macOS (libomp/libgomp clash, see feedback memo). Dropped LGBM —
        # it was the weakest tree (row #33: 0.8294) and primary culprit.
        "name": "R-B2_heterogeneous_CB_XGB",
        "desc": "8 CatBoost + 8 XGB stacking (LGBM dropped due to segfault)",
        "components": (
            cluster_catboost_variants(
                REG_TPE_CENTER, n=4, jitter_pct=0.2,
                seed=11, seed_offset=200, iter_cap=ITER_CAP,
            )
            + cluster_catboost_variants(
                REG_SHALLOW_CENTER, n=4, jitter_pct=0.2,
                seed=22, seed_offset=204, iter_cap=ITER_CAP,
            )
            + xgb_variants(8, seed_base=500)
        ),
    },
    {
        "name": "R-C2_CB_dominant_with_tabnet",
        "desc": "10 CatBoost + 4 XGB + 2 TabNet stacking (LGBM dropped)",
        "components": (
            cluster_catboost_variants(
                REG_TPE_CENTER, n=5, jitter_pct=0.2,
                seed=33, seed_offset=200, iter_cap=ITER_CAP,
            )
            + cluster_catboost_variants(
                REG_SHALLOW_CENTER, n=5, jitter_pct=0.2,
                seed=44, seed_offset=205, iter_cap=ITER_CAP,
            )
            + xgb_variants(4, seed_base=510)
            + tabnet_variants(2, seed_base=710)
        ),
    },
    {
        "name": "R-D2_max_spread_15CB_6XGB",
        "desc": "5 TPE-CB + 5 shallow-CB + 5 middle-CB + 6 XGB (LGBM dropped)",
        "components": (
            reg_diverse_catboost(n_per_cluster=5, seed_base=99)
            + xgb_variants(6, seed_base=520)
        ),
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def run_config(spec: dict, n_splits: int = 5) -> dict:
    from training.cross_validator import CrossValidator

    cfg = Config()
    cfg.training.task_type = TaskType.REGRESSION
    cfg.models.reg_model_type = "ensemble"
    cfg.training.n_splits = n_splits
    cfg.models.ensemble_reg_mode = "stacking"
    cfg.models.ensemble_meta_learner_type = "ridge"
    cfg.models.ensemble_reg_components = spec["components"]

    log(f"=== {spec['name']} ===")
    log(f"  desc: {spec['desc']}")
    log(f"  components: {len(spec['components'])}")

    t0 = time.time()
    cv = CrossValidator(cfg)
    results = cv.run()
    elapsed = time.time() - t0
    log(
        f"  r2={results['r2_mean']:.4f} ± {results['r2_std']:.4f} "
        f"({elapsed/60:.1f} min)"
    )
    return {
        "name": spec["name"],
        "desc": spec["desc"],
        "n_components": len(spec["components"]),
        "r2_mean": results["r2_mean"],
        "r2_std": results["r2_std"],
        "r2_per_fold": results["r2_per_fold"],
        "elapsed_min": round(elapsed / 60, 1),
    }


def write_md_summary(results: list, out_dir: Path) -> None:
    lines = [
        "# Regression Ensemble Results",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "Baseline: CatBoost default 5-fold **r2=0.8367 ± 0.0148** (row #34).",
        "",
        "| Name | Bases | r2 mean | r2 std | Δ vs CatBoost default | Time |",
        "|------|-------|---------|--------|----------------------|------|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['name']} | — | ERROR | — | — | — |")
        else:
            d = r["r2_mean"] - 0.8367
            lines.append(
                f"| {r['name']} | {r['n_components']} | "
                f"**{r['r2_mean']:.4f}** | {r['r2_std']:.4f} | "
                f"{d:+.4f} | {r['elapsed_min']}m |"
            )
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")


def main(only: str | None = None) -> None:
    """Run all configs, or only the one whose name matches `only`.

    Per-config isolation: a wrapper shell calls this script once per config
    (--only NAME). Each invocation is a fresh Python process, so joblib
    workers and native ML libs (XGB/LGBM/CatBoost/TabNet) start clean.
    """
    log(f"Regression ensemble autorun start (only={only})")
    out_dir = Path("/tmp/regression_ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = [s for s in CONFIGS if (only is None or s["name"] == only)]
    if not selected:
        log(f"  no config matched --only {only!r}; available: "
            + ", ".join(c["name"] for c in CONFIGS))
        return

    # Append to existing results so the markdown stays a running tally.
    results_path = out_dir / "results.json"
    all_results: list = []
    if results_path.exists():
        try:
            all_results = json.loads(results_path.read_text())
        except Exception:  # noqa: BLE001
            all_results = []

    for spec in selected:
        # Skip if we already have a successful result for this name.
        if any(r.get("name") == spec["name"] and "error" not in r for r in all_results):
            log(f"  SKIP {spec['name']} (already completed)")
            continue
        # Drop any stale error entry for this name before re-running.
        all_results = [r for r in all_results if r.get("name") != spec["name"]]

        try:
            r = run_config(spec, n_splits=5)
            all_results.append(r)
        except Exception as exc:  # noqa: BLE001
            log(f"  ERROR in {spec['name']}: {exc}")
            import traceback
            traceback.print_exc()
            all_results.append({"name": spec["name"], "error": str(exc)})

        results_path.write_text(json.dumps(all_results, indent=2))
        write_md_summary(all_results, out_dir)

    log("=" * 60)
    log("RUN SUMMARY (this invocation)")
    log("=" * 60)
    for r in all_results:
        if r.get("name") not in {s["name"] for s in selected}:
            continue
        if "error" in r:
            log(f"  {r['name']}: ERROR ({r['error']})")
        else:
            d = r["r2_mean"] - 0.8367
            log(
                f"  {r['name']}: r2={r['r2_mean']:.4f} ± {r['r2_std']:.4f} "
                f"({d:+.4f} vs CatBoost default 5-fold) [{r['elapsed_min']}m]"
            )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None,
                   help="Run only this config name; default runs all in this process.")
    args = p.parse_args()
    main(only=args.only)
