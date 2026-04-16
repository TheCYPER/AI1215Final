"""Smoke test for new Phase III model wrappers.

Verifies that MLP, LogRegPoly and TabNet (if available) each:
- register in MODEL_REGISTRY
- build via model_factory
- fit on tiny synthetic data
- predict / predict_proba with correct shapes
"""

import sys
import warnings
import numpy as np

# Silence the noisy-but-benign optimization warnings.
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from configs.config import Config, TaskType
from modeling import MODEL_REGISTRY, model_factory


def main() -> int:
    print(f"Registry keys: {sorted(MODEL_REGISTRY.keys())}", flush=True)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 20)).astype(np.float32)
    y = rng.integers(0, 5, size=500)
    # Simulate 3 ordinal-encoded cat cols at [17, 18, 19].
    X[:, 17] = rng.integers(0, 5, size=500).astype(np.float32)
    X[:, 18] = rng.integers(0, 4, size=500).astype(np.float32)
    X[:, 19] = rng.integers(0, 6, size=500).astype(np.float32)

    targets = ["mlp", "logreg_poly"]
    if "tabnet" in MODEL_REGISTRY:
        targets.append("tabnet")

    failures = []
    for mtype in targets:
        cfg = Config()
        cfg.training.task_type = TaskType.CLASSIFICATION
        cfg.models.clf_model_type = mtype
        if mtype == "tabnet":
            cfg.models.tabnet_clf_params["max_epochs"] = 3
            cfg.models.tabnet_clf_params["patience"] = 2
        if mtype == "mlp":
            cfg.models.mlp_clf_params["max_iter"] = 30

        try:
            m = model_factory(cfg)
            if mtype == "tabnet":
                m.fit(X, y, categorical_feature=[17, 18, 19])
            else:
                m.fit(X, y)
            proba = m.predict_proba(X[:10])
            pred = m.predict(X[:10])
            assert proba.shape == (10, 5), f"bad proba shape: {proba.shape}"
            assert pred.shape == (10,), f"bad pred shape: {pred.shape}"
            print(
                f"{mtype:13} OK — proba{proba.shape} row0_sum={proba[0].sum():.3f} pred[:5]={pred[:5].tolist()}",
                flush=True,
            )
        except Exception as e:
            failures.append((mtype, repr(e)))
            print(f"{mtype:13} FAIL — {e!r}", flush=True)

    if failures:
        print(f"\n{len(failures)} failures", flush=True)
        return 1
    print("\nALL PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
