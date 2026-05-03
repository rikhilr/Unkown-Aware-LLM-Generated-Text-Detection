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
