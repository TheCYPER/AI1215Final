#!/usr/bin/env bash
# Run each regression ensemble config in its own Python process.
# Avoids joblib worker leakage across configs (caused R-B SIGKILL on 2026-04-18).
#
# Usage:
#   nohup bash scripts/regression_ensemble_runner.sh \
#     > /tmp/regression_ensemble_runner.log 2>&1 &

set -u

cd "$(dirname "$0")/.."
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# macOS-specific: XGBoost and LightGBM both link libomp; without these vars
# the second one's libomp causes Segfault 11 during ensemble construction.
export KMP_DUPLICATE_LIB_OK=TRUE
export KMP_INIT_AT_FORK=FALSE

LOG_DIR=/tmp/regression_ensemble
mkdir -p "$LOG_DIR"

CONFIGS=(
  "R-A_30_diverse_catboost"           # already done; --only will SKIP if found
  "R-B2_heterogeneous_CB_XGB"         # was R-B; LGBM dropped (libomp segfault)
  "R-C2_CB_dominant_with_tabnet"      # was R-C; LGBM dropped
  "R-D2_max_spread_15CB_6XGB"         # was R-D; LGBM dropped
)

echo "[$(date +'%F %T')] runner start; running ${#CONFIGS[@]} configs sequentially in fresh processes"

for cfg in "${CONFIGS[@]}"; do
  PER_LOG="$LOG_DIR/${cfg}.log"
  echo "[$(date +'%F %T')] >>> launching ${cfg} (per-config log: ${PER_LOG})"
  python -u scripts/regression_ensemble_autorun.py --only "${cfg}" \
    > "${PER_LOG}" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[$(date +'%F %T')] !!! ${cfg} exited with code ${rc}; continuing to next config"
  else
    echo "[$(date +'%F %T')] <<< ${cfg} done OK"
  fi
done

echo "[$(date +'%F %T')] runner done"
echo "Final results: $LOG_DIR/results.md"
cat "$LOG_DIR/results.md"
