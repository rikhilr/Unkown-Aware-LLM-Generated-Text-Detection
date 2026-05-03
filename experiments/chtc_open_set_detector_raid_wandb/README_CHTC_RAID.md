# CHTC RAID W&B open-set detector

## Setup

```bash
unzip chtc_open_set_detector_raid_wandb.zip
cd chtc_open_set_detector_raid_wandb
mkdir -p logs
chmod +x *.sh
```

For W&B:

```bash
export WANDB_API_KEY="your_key"
export WANDB_PROJECT="open-set-llm-detector"
export WANDB_MODE="online"
# optional
export WANDB_ENTITY="your_entity"
```

For Hugging Face rate limits:

```bash
export HF_TOKEN="your_hf_token"
```

## Inspect RAID columns first

```bash
condor_submit inspect_raid.sub
```

Then check:

```bash
cat logs/inspect_<cluster>_0.out
```

## Train on GPU

```bash
condor_submit train_gpu.sub
```

The job uses:

- `request_gpus = 1`
- `+WantGPULab = true`
- `gpus_minimum_memory = 16000`
- CUDA through PyTorch
- W&B online or offline logging

Outputs:

```text
open_set_detector.tar.gz
```

Unpack it:

```bash
tar -xzf open_set_detector.tar.gz
```

## Predict

```bash
python predict_open_set.py --artifact_dir open_set_detector --text "paste text here"
```
