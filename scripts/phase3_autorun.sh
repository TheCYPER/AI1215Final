#!/bin/bash
# Phase III overnight automation.
#
# For each base model family (MLP, LogRegPoly, TabNet, LightGBM widened,
# XGBoost widened):
#   1. Run 5-fold CV with default params.   → baseline accuracy
#   2. Run Optuna tune with HyperbandPruner. → best_params JSON
#   3. Re-run 5-fold CV with tuned params.   → tuned accuracy
# Each step's logs go to a timestamped file under /tmp/phase3/. A running
# summary is appended to experiments.md via log_experiment.py.
#
# Designed to survive individual model failures: `set +e` + per-step trap.
# Run from worktree root:
#   bash scripts/phase3_autorun.sh
#
# Monitor progress:
#   tail -f /tmp/phase3/SUMMARY.log
set -u

WORKTREE="/Users/percy/MLFinal2026-phase3"
cd "$WORKTREE"
source .venv/bin/activate

LOG_DIR="/tmp/phase3"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/SUMMARY.log"

# Single-thread BLAS / MKL avoids fork-unsafe crashes (segfault on TabNet, etc.)
# Main repo's CV may still be running on the other cores; our runs take what's left.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH="$WORKTREE"

log() {
    local msg="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" | tee -a "$SUMMARY"
}

# Extract CV accuracy + std from a cv run log file. Returns empty if not found.
extract_cv_mean_std() {
    local logf="$1"
    grep -E "CV accuracy:" "$logf" | tail -1 | sed -E 's/.*CV accuracy: ([0-9.]+) \+\/- ([0-9.]+).*/\1 \2/'
}

# Run a CV with a given model and optional tune-results file.
# Args: model_name, label, [tune_json_path]
run_cv() {
    local model="$1"; local label="$2"; local tune_json="${3:-}"
    local logf="$LOG_DIR/${label}_cv.log"
    log "=== CV start: $label (model=$model) ==="
    local extra=""
    if [ -n "$tune_json" ] && [ -f "$tune_json" ]; then
        extra="--apply_tune_results $tune_json"
        log "  applying tune results from $tune_json"
    fi
    python -u main.py --mode cv --task classification --model "$model" $extra > "$logf" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        log "  CV FAILED rc=$rc (see $logf)"
        return 1
    fi
    local ms="$(extract_cv_mean_std "$logf")"
    log "  CV DONE: $label  acc=$ms"
    echo "$ms"
    return 0
}

# Run an Optuna tune for a model. Returns 0 on success; emits per-label JSON.
# Args: model_name, label, sampler, pruner
run_tune() {
    local model="$1"; local label="$2"; local sampler="${3:-tpe}"; local pruner="${4:-hyperband}"
    local logf="$LOG_DIR/${label}_tune.log"
    log "=== TUNE start: $label (sampler=$sampler pruner=$pruner) ==="
    python -u main.py --mode tune --task classification --model "$model" \
        --sampler "$sampler" --pruner "$pruner" > "$logf" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        log "  TUNE FAILED rc=$rc (see $logf)"
        return 1
    fi
    # Tuner writes to outputs/classification_tuning_${sampler}.json — copy to a
    # per-label filename so the next tune can overwrite without losing history.
    local src="$WORKTREE/outputs/classification_tuning_${sampler}.json"
    local dst="$LOG_DIR/${label}_best.json"
    if [ -f "$src" ]; then
        cp "$src" "$dst"
    fi
    local best="$(grep -E 'Best score:' "$logf" | tail -1)"
    log "  TUNE DONE: $label  $best  (best_params → $dst)"
    return 0
}

# Path of the per-label JSON where a tune stashed its best_params.
tune_json_path() {
    local label="$1"
    echo "$LOG_DIR/${label}_best.json"
}

# Override tuning knobs for Phase III (per-run overrides via env is cleaner,
# but we don't have plumbing for that, so we trust config defaults: 60 trials,
# no timeout. For cheaper models we can rely on HyperbandPruner to kill bad trials).
log "==================== PHASE III AUTORUN START ===================="
log "Worktree: $WORKTREE"
log "SOTA baseline (main): row #14 TPE-widened CatBoost 0.8263 ± 0.0023"

# ---------- LogRegPoly ----------
# Fast: no tune, just CV. (LogReg is nearly deterministic given poly_degree.)
run_cv "logreg_poly" "logreg_poly_baseline" || true

# ---------- MLP ----------
run_cv "mlp" "mlp_baseline" || true
run_tune "mlp" "mlp" "tpe" "hyperband" || true
run_cv "mlp" "mlp_tuned" "$(tune_json_path mlp)" || true

# ---------- LightGBM widened tune ----------
# LGBM baseline at default was row #4 (0.7953). Widened search space should
# push that further. After tune, re-CV.
run_tune "lightgbm" "lightgbm_widened" "tpe" "hyperband" || true
run_cv "lightgbm" "lightgbm_widened_tuned" "$(tune_json_path lightgbm_widened)" || true

# ---------- XGBoost widened tune ----------
run_tune "xgboost" "xgboost_widened" "tpe" "hyperband" || true
run_cv "xgboost" "xgboost_widened_tuned" "$(tune_json_path xgboost_widened)" || true

# ---------- TabNet ----------
# Baseline only: tuning TabNet is expensive (~3-5 min per trial × 60 = 5h).
# If baseline looks promising (>0.78) we can tune in a later run.
run_cv "tabnet" "tabnet_baseline" || true

log "==================== PHASE III AUTORUN DONE ===================="

# Emit a markdown summary table, easy to paste into experiments.md.
SUMMARY_MD="$LOG_DIR/results.md"
{
    echo "# Phase III autorun results"
    echo
    echo "| step | mean acc | std |"
    echo "|------|----------|-----|"
    for f in "$LOG_DIR"/*_cv.log; do
        [ -f "$f" ] || continue
        label=$(basename "$f" _cv.log)
        ms=$(extract_cv_mean_std "$f")
        [ -z "$ms" ] && ms="— —"
        # shellcheck disable=SC2086
        read -r mean std <<< "$ms"
        echo "| $label | ${mean:-—} | ${std:-—} |"
    done
    echo
    echo "Tune JSONs in $LOG_DIR/*_best.json"
} > "$SUMMARY_MD"

log "Artifacts:"
log "  $LOG_DIR/*.log             (per-step log files)"
log "  $LOG_DIR/results.md        (markdown summary to paste into experiments.md)"
log "  $LOG_DIR/*_best.json       (tune best_params per model)"
log "  $WORKTREE/outputs/         (metrics + models)"
