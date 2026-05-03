#!/bin/bash
set -euo pipefail

export USER="${USER:-rrao27}"
export LOGNAME="${LOGNAME:-rrao27}"
export WANDB_USERNAME="${WANDB_USERNAME:-rrao27}"
export HOME="${PWD}"
export HF_HOME="${PWD}/hf_cache"
export TRANSFORMERS_CACHE="${PWD}/hf_cache"
export HF_DATASETS_CACHE="${PWD}/hf_cache"

cleanup_and_package() {
    EXIT_CODE=$?
    echo
    echo "===== Packaging output ====="
    if [ -d open_set_detector ]; then
        tar -czf open_set_detector.tar.gz open_set_detector
    else
        mkdir -p open_set_detector
        echo "training failed before output directory was created" > open_set_detector/FAILED.txt
        tar -czf open_set_detector.tar.gz open_set_detector
    fi
    echo "Final files:"
    ls -lh || true
    ls -lh open_set_detector || true
    exit ${EXIT_CODE}
}
trap cleanup_and_package EXIT

echo "===== Job environment ====="
date
hostname
pwd
id || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"
echo "WANDB_PROJECT=${WANDB_PROJECT:-not_set}"
echo "WANDB_MODE=${WANDB_MODE:-online}"

echo
echo "===== GPU info ====="
nvidia-smi || true

echo
echo "===== Python info ====="
python --version
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda device count:', torch.cuda.device_count())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
PY

echo
echo "===== Installing Python dependencies ====="
python -m pip install --user --upgrade pip
python -m pip install --user transformers datasets scikit-learn numpy pandas tqdm joblib wandb accelerate

echo
echo "===== Starting RAID open-set training ====="
python train_open_set_chtc.py \
  --dataset_id liamdugan/raid \
  --dataset_config "" \
  --dataset_split train \
  --num_samples 50000 \
  --batch_size 32 \
  --max_length 512 \
  --amp_dtype fp16 \
  --output_dir open_set_detector \
  --use_wandb \
  --wandb_project "${WANDB_PROJECT:-open-set-llm-detector}" \
  --wandb_entity "${WANDB_ENTITY:-}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --wandb_run_name "raid-chtc-${CLUSTER:-local}-${PROCESS:-0}" \
  --wandb_log_every 5
