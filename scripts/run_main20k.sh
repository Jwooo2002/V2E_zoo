#!/usr/bin/env bash
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate cdm-mamba-kd-cu121

cd ~/PycharmProjects/cdm-mamba-kd

export CSDM_DATA_PATH=/home/jiwoong/PycharmProjects/cdm-mamba-kd/data/real_small_corpus.txt
export CSDM_STORAGE_MIN_FREE_GB=${CSDM_STORAGE_MIN_FREE_GB:-20}
export CSDM_ARTIFACT_HEALTH_CACHE_SAMPLE_SIZE=${CSDM_ARTIFACT_HEALTH_CACHE_SAMPLE_SIZE:-64}

echo "[main20k] CSDM_DATA_PATH=$CSDM_DATA_PATH"
echo "[main20k] CSDM_STORAGE_MIN_FREE_GB=$CSDM_STORAGE_MIN_FREE_GB"
echo "[main20k] CSDM_ARTIFACT_HEALTH_CACHE_SAMPLE_SIZE=$CSDM_ARTIFACT_HEALTH_CACHE_SAMPLE_SIZE"
echo "[main20k] started at $(date)"

echo "[main20k] launching KD on GPU 0"
CUDA_VISIBLE_DEVICES=0 python scripts/run_registered_experiment.py \
  --experiment configs/experiments/train_real_small_kd.yaml \
  --base-output-dir /mnt/sda2/csdm_main20k_kd \
  --override max_steps=20000 \
  --override batch_size=2 \
  --override gradient_accumulation_steps=16 \
  --override storage_min_free_gb="$CSDM_STORAGE_MIN_FREE_GB" \
  --with-perturbation \
  --with-report \
  --artifact-health-check \
  --artifact-health-cache-sample-size "$CSDM_ARTIFACT_HEALTH_CACHE_SAMPLE_SIZE" \
  --no-timeout &
PID_KD=$!

echo "[main20k] launching CSDM w=0.1 on GPU 1"
CUDA_VISIBLE_DEVICES=1 python scripts/run_registered_experiment.py \
  --experiment configs/experiments/train_real_small_csdm.yaml \
  --base-output-dir /mnt/sda2/csdm_main20k_csdm_w01 \
  --override max_steps=20000 \
  --override batch_size=2 \
  --override gradient_accumulation_steps=16 \
  --override csdm_weight=0.1 \
  --override storage_min_free_gb="$CSDM_STORAGE_MIN_FREE_GB" \
  --with-perturbation \
  --with-report \
  --artifact-health-check \
  --artifact-health-cache-sample-size "$CSDM_ARTIFACT_HEALTH_CACHE_SAMPLE_SIZE" \
  --no-timeout &
PID_CSDM=$!

echo "[main20k] waiting for KD PID=$PID_KD and CSDM PID=$PID_CSDM"
set +e
wait "$PID_KD"
RC_KD=$?
echo "[main20k] KD finished with rc=$RC_KD at $(date)"

wait "$PID_CSDM"
RC_CSDM=$?
echo "[main20k] CSDM w=0.1 finished with rc=$RC_CSDM at $(date)"
set -e

if [[ "$RC_KD" -ne 0 || "$RC_CSDM" -ne 0 ]]; then
  echo "[main20k] stopping before top-k because KD rc=$RC_KD and CSDM rc=$RC_CSDM"
  exit 1
fi

echo "[main20k] launching CSDM+top-k w=0.03 on GPU 0"
set +e
CUDA_VISIBLE_DEVICES=0 python scripts/run_registered_experiment.py \
  --experiment configs/experiments/train_real_small_csdm_topk.yaml \
  --base-output-dir /mnt/sda2/csdm_main20k_csdm_topk_w003 \
  --override max_steps=20000 \
  --override batch_size=2 \
  --override gradient_accumulation_steps=16 \
  --override csdm_weight=0.03 \
  --override storage_min_free_gb="$CSDM_STORAGE_MIN_FREE_GB" \
  --with-perturbation \
  --with-report \
  --artifact-health-check \
  --artifact-health-cache-sample-size "$CSDM_ARTIFACT_HEALTH_CACHE_SAMPLE_SIZE" \
  --no-timeout
RC_TOPK=$?
set -e

if [[ "$RC_TOPK" -ne 0 ]]; then
  echo "[main20k] CSDM+top-k w=0.03 failed with rc=$RC_TOPK at $(date)"
  exit "$RC_TOPK"
fi

echo "[main20k] all done at $(date)"
