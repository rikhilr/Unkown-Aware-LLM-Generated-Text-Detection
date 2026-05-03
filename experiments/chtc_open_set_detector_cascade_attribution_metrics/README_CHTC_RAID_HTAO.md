# CHTC RAID + HTAO-style OOD detector

This project trains a GPU embedding pipeline on RAID and then fits:

1. a baseline human-vs-LLM classifier,
2. a known-generator classifier,
3. HTAO-inspired OOD scores:
   - global DeepSVDD for human-as-outlier detection,
   - class-conditional DeepSVDD for unknown generator rejection,
   - energy score from known-generator logits,
   - an ensemble rejector using energy + DeepSVDD features.

## Run

```bash
mkdir -p logs
chmod +x *.sh
export WANDB_API_KEY="..."
export WANDB_PROJECT="open-set-llm-detector"
export WANDB_MODE="online"
condor_submit train_gpu.sub
```

## Using more or less RAID

`run_train.sh` currently uses `train,extra`, includes adversarial rows, and caps each RAID label at `--samples_per_label 20000`.
Increase this to use more of RAID:

```bash
--samples_per_label 50000
```

Using literally every RAID row through a transformer encoder can take many GPU-hours/days and requires much more disk, so the script supports it but does not default to it:

```bash
--samples_per_label -1 --num_samples -1
```

## W&B

The script logs embedding progress every `--wandb_log_every` batches with a monotonic global step, so charts should update live during embedding extraction instead of only showing final points.

Key charts:

- `embedding/*/percent`
- `embedding/*/examples_per_sec`
- `gpu/memory_allocated_gb`
- `open_set_val/htao_energy/*`
- `open_set_val/htao_deepsvdd/*`
- `open_set_val/ensemble_rejector/*`
- `open_set_test/*`
