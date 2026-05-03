#!/bin/bash
set -euo pipefail

export USER="${USER:-rrao27}"
export LOGNAME="${LOGNAME:-rrao27}"
export WANDB_USERNAME="${WANDB_USERNAME:-rrao27}"
export HOME="${PWD}"
export PYTHONUNBUFFERED=1
export HF_HOME="${PWD}/hf_cache"
export TRANSFORMERS_CACHE="${PWD}/hf_cache"
export HF_DATASETS_CACHE="${PWD}/hf_cache"

# Optional: keep Hugging Face/W&B credentials passed by condor getenv.
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-open-set-detector}"

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
python -m pip install --upgrade pip
python -m pip install "transformers>=4.40" "datasets>=2.19" "scikit-learn" "numpy" "pandas" "tqdm" "joblib" "wandb"

echo
echo "===== Starting training ====="
set +e
python train_open_set_chtc.py \
  --dataset_id liamdugan/raid \
  --dataset_splits train,extra \
  --target_models human,chatgpt,gpt4,gpt3,gpt2,llama-chat,cohere,cohere-chat,mpt,mpt-chat,mistral,mistral-chat \
  --known_models llama-chat,mpt,mpt-chat,chatgpt,gpt3,gpt4,cohere \
  --wild_unknown_models gpt2,cohere-chat \
  --val_unknown_models mistral \
  --test_unknown_models mistral-chat \
  --samples_per_label 20000 \
  --balance_domains \
  --target_domains abstracts,books,news,poetry,recipes,reddit,reviews,wiki \
  --samples_per_label_per_domain 2500 \
  --num_samples -1 \
  --include_adversarial \
  --shuffle_buffer 100000 \
  --batch_size 32 \
  --max_length 512 \
  --amp_dtype fp16 \
  --energy_gate_strategy max_known_false_unknown \
  --energy_gate_max_known_false_unknown 0.18 \
  --output_dir open_set_detector \
  --use_wandb \
  --wandb_project "${WANDB_PROJECT:-open-set-detector}" \
  --wandb_entity "${WANDB_ENTITY:-}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --wandb_run_name "openworld-raid-${CLUSTER:-local}-${PROCESS:-0}" \
  --wandb_log_every 5
TRAIN_EXIT_CODE=$?
set -e

echo
echo "===== Packaging output ====="
if [ -d open_set_detector ]; then
  tar -czf open_set_detector.tar.gz open_set_detector
else
  mkdir -p open_set_detector
  echo "training failed before output directory was created" > open_set_detector/FAILED.txt
  tar -czf open_set_detector.tar.gz open_set_detector
fi
ls -lh
exit ${TRAIN_EXIT_CODE}
