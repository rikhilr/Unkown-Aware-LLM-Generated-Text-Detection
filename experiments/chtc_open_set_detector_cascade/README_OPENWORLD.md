# Open-world RAID detector for CHTC

This version implements a two-stage detector:

1. Stage 1 triage: `human`, `llm`, or `unknown`.
2. Stage 2 attribution: if stage 1 predicts `llm`, classify the known LLM family.

It uses:

- RAID train + extra splits
- group splitting by `source_id`
- HTAO-style global DeepSVDD for human-as-OOD
- energy and class-conditional DeepSVDD scores
- wild-OOD training inspired by “Feed Two Birds with One Scone”
- a stage-1 meta-classifier over OOD features
- W&B live progress logging

Run:

```bash
export WANDB_API_KEY="..."
export WANDB_PROJECT="open-set-detector"
export WANDB_MODE="online"
condor_submit train_gpu.sub
```

Default split:

- known LLMs: llama-chat, mpt, mpt-chat, chatgpt, gpt3, gpt4, cohere
- wild unknowns used to train the rejector: gpt2, cohere-chat
- validation unknown: mistral
- test unknown: mistral-chat

Use multiple jobs rotating held-out models to estimate open-world robustness.

## 2026-05 cascade update

The primary inference path is now:

```text
input text
  -> human-vs-LLM classifier
  -> if human: return human
  -> HTAO-style energy unknown gate
  -> if energy > threshold: return unknown
  -> known-LLM classifier
```

This replaces the previous 3-way `human / llm / unknown` triage classifier as the main decision layer. The triage classifier can still be run as an ablation by adding:

```bash
--run_triage_ablation
```

to `run_train.sh`, but it is off by default because it reduced final known-model attribution accuracy in the previous run.

New W&B/log sections to inspect:

```text
cascade_wild_val/final_gated
cascade_val_heldout/final_gated
cascade_test_heldout/final_gated
cascade_test_heldout/gate/known_false_unknown_rate
cascade_test_heldout/gate/unknown_detection_rate
cascade_test_heldout/gate/known_correct_final_rate
cascade_test_heldout/gate/human_recall
```

`predict_open_set.py` now uses this cascade directly and no longer requires `stage1_triage_clf.joblib`.
