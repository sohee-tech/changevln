#!/usr/bin/env bash
# =============================================================================
# run_sub500_eval.sh — 500-ep benchmark evaluation (91 reused + 409 new)
#
# Arm 순서 (sequential, crash-safe):
#   1. sub500_base_clean  (409 ep, ~1.8h)
#   2. sub500_base_obs    (409 ep, ~8.2h)
#   3. sub500_oap_clean   (409 ep, ~1.8h)
#   4. sub500_oap_obs     (409 ep, ~8.5h)
#   Total: ~20.3h (each arm skips if result file exists)
#
# VRAM: RTX 5090 32GB — sequential only (NaVILA 8B ~20-22GB per instance)
# =============================================================================
set -e
cd "$(dirname "$0")"

CONDA_ENV="vlnce-stable"
LOG_DIR="eval_out/sub500_logs"
RESULT_SUB="navila-llama3-8b-8f/VLN-CE-v1/val_unseen/val_unseen_1-0.json"

mkdir -p "$LOG_DIR"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_DIR/master.log"
}

run_arm() {
    local label="$1"
    local yaml="$2"
    local outdir="$3"
    local arm_log="$LOG_DIR/${label}.log"
    local result="$outdir/$RESULT_SUB"

    if [ -f "$result" ]; then
        log "SKIP $label — result already exists: $result"
        return 0
    fi

    log "START $label  yaml=$yaml"
    local t0=$(date +%s)

    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"

    CUDA_VISIBLE_DEVICES=0 python run.py \
        --run-type eval \
        --exp-config "$yaml" \
        2>&1 | tee "$arm_log"

    local t1=$(date +%s)
    local elapsed=$(( t1 - t0 ))
    local elapsed_min=$(( elapsed / 60 ))

    if [ ! -f "$result" ]; then
        log "FAIL $label — result file missing after run! (${elapsed_min}min)"
        log "  Log: $arm_log"
        exit 1
    fi

    log "DONE $label (${elapsed_min}min)  result=$result"
}

log "=========================================="
log "sub500 evaluation START (4 arms, 409 ep each)"
log "Log dir: $LOG_DIR"
log "=========================================="

# Arm 1: Base + clean
run_arm "sub500_base_clean" \
    "vlnce_baselines/config/r2r_baselines/navila_sub500_base_clean.yaml" \
    "eval_out/sub500_base_clean_traj_v2"

# Arm 2: Base + obstacle
run_arm "sub500_base_obs" \
    "vlnce_baselines/config/r2r_baselines/navila_sub500_base_obs.yaml" \
    "eval_out/sub500_base_traj_v2"

# Arm 3: OAP + clean
run_arm "sub500_oap_clean" \
    "vlnce_baselines/config/r2r_baselines/navila_sub500_oap_clean.yaml" \
    "eval_out/sub500_oap_clean_traj_v2"

# Arm 4: OAP + obstacle
run_arm "sub500_oap_obs" \
    "vlnce_baselines/config/r2r_baselines/navila_sub500_oap_obs.yaml" \
    "eval_out/sub500_oap_traj_v2"

log "=========================================="
log "ALL 4 ARMS COMPLETE"
log "  sub500_base_traj_v2:       eval_out/sub500_base_traj_v2/$RESULT_SUB"
log "  sub500_base_clean_traj_v2: eval_out/sub500_base_clean_traj_v2/$RESULT_SUB"
log "  sub500_oap_traj_v2:        eval_out/sub500_oap_traj_v2/$RESULT_SUB"
log "  sub500_oap_clean_traj_v2:  eval_out/sub500_oap_clean_traj_v2/$RESULT_SUB"
log "  Merge with obs91 results after this run completes."
log "=========================================="
