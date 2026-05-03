# Open-world RAID detector for CHTC

This version uses the cascade that performed best in the recent experiments:

```text
input text
  -> human-vs-LLM classifier
  -> if human: return human
  -> HTAO-style energy unknown gate
  -> if energy > threshold: return unknown
  -> known-LLM classifier
```

It uses:

- RAID train + extra splits
- group splitting by `source_id`
- human-vs-LLM binary classifier
- known-LLM family classifier trained only on known LLMs
- HTAO-style energy scoring for the unknown gate
- class-conditional DeepSVDD / ensemble rejector as logged baselines
- wild-OOD validation inspired by “Feed Two Birds with One Scone”
- W&B live progress logging

## Important cascade update

The primary energy gate is now tuned to avoid interfering with known-model attribution. Instead of using the balanced OOD threshold by default, it chooses the best unknown-detection threshold subject to this validation constraint:

```text
known_false_unknown_rate <= --energy_gate_max_known_false_unknown
```

The default is:

```bash
--energy_gate_strategy max_known_false_unknown
--energy_gate_max_known_false_unknown 0.10
```

Raise the cap to `0.15` or `0.20` if you want more aggressive unknown detection. Lower it if you want to protect known-model classification more.

## Run

```bash
export WANDB_API_KEY="..."
export WANDB_PROJECT="open-set-detector"
export WANDB_MODE="online"
condor_submit train_gpu.sub
```

Default split:

- known LLMs: llama-chat, mpt, mpt-chat, chatgpt, gpt3, gpt4, cohere
- wild unknowns used to tune the unknown gate: gpt2, cohere-chat
- validation unknown: mistral
- test unknown: mistral-chat

Use multiple jobs rotating held-out models to estimate open-world robustness.

## Metrics to inspect

Use:

```bash
grep -E "primary_energy_gate|cascade|attribution_after_gate|known_false_unknown|unknown_detection|known_attr_accuracy|known_pass_rate|Known-LLM|accuracy|macro avg|weighted avg" logs/train_<cluster_id>_0.out
```

The most important sections are:

```text
primary_energy_gate/*
cascade_test_heldout/final_gated
cascade_test_heldout/gate/known_false_unknown_rate
cascade_test_heldout/gate/unknown_detection_rate
cascade_test_heldout/gate/known_pass_rate
cascade_test_heldout/gate/known_attr_accuracy_given_passed_gate
cascade_test_heldout/gate/raw_known_model_accuracy_before_gate
cascade_test_heldout/attribution_after_gate
```

Interpretation:

- `known_false_unknown_rate`: known LLMs wrongly rejected as unknown.
- `unknown_detection_rate`: held-out unknown LLMs correctly rejected.
- `known_pass_rate`: known LLMs that pass both gates and reach attribution.
- `known_attr_accuracy_given_passed_gate`: attribution accuracy only among known LLMs that pass the gate.
- `raw_known_model_accuracy_before_gate`: known attribution accuracy before applying the human/unknown gates.

The old 3-way `human / llm / unknown` triage classifier is off by default because it reduced final attribution accuracy. Run it only as an ablation with:

```bash
--run_triage_ablation
```

`predict_open_set.py` uses the cascade directly and no longer requires `stage1_triage_clf.joblib`.

## Domain-balanced RAID collection

The latest version balances the training/evaluation sample across RAID domains instead of allowing the streaming order to fill all quotas with `abstracts`.

The run script now uses:

```bash
--balance_domains \
--target_domains abstracts,books,news,poetry,recipes,reddit,reviews,wiki \
--samples_per_label_per_domain 2500
```

This means each model/domain pair gets up to 2,500 examples. With 12 target labels and 8 domains, the target total is:

```text
12 labels * 8 domains * 2,500 = 240,000 examples
```

This keeps the total size comparable to the old 20,000-per-label run, but spreads the data across all RAID domains.

To make a smaller smoke test, change `--samples_per_label_per_domain 2500` to `250`.

To make a larger final run, change it to `5000`, which gives 40,000 examples per label and 480,000 examples total.
