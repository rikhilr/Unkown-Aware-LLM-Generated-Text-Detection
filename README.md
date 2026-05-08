# Unknown-Aware LLM-Generated Text Detection

A cascade classifier that detects AI-generated text while handling **unknown LLM sources** — text from models the system was never trained on. Rather than forcing every input into a known category, the system can say "this is LLM-generated, but from an unknown model."

**Demo:** https://huggingface.co/spaces/rikhilrao/llm-detector-demo

## How it works

The system uses a three-stage cascade:

1. **Human vs. LLM gate** — A logistic regression classifier over [StyleDistance](https://huggingface.co/StyleDistance/styledistance) embeddings decides if the text is human-written or machine-generated.
2. **Unknown LLM gate (HTAO energy score)** — If the text is classified as LLM-generated, an energy-based out-of-distribution score determines whether it comes from a *known* or *unknown* model family. A high energy score means the text is likely from an unseen model.
3. **Known LLM attribution** — If the energy score is below the threshold, a multi-class logistic regression identifies which known LLM produced it (e.g., ChatGPT, GPT-4, LLaMA, Cohere, MPT).

The OOD detector uses a combination of:
- **Energy scores** from the known-model classifier's logits (HTAO-style)
- **Normalized DeepSVDD distances** to per-class centroids in embedding space
- A **learned ensemble rejector** trained on wild unknown model examples

## Datasets

- **[RAID](https://huggingface.co/datasets/liamdugan/raid)** — Primary training dataset. Multi-domain, multi-model benchmark with adversarial attack variants.
- **[MAGE / DeepFake](https://huggingface.co/datasets/yaful/MAGE)** — Secondary dataset used for EDA.

Known models during training: `chatgpt`, `gpt3`, `gpt4`, `llama-chat`, `cohere`, `mpt`, `mpt-chat`

Unknown models held out for evaluation: `mistral`, `mistral-chat`

## Repository structure

```
experiments/                         # Training and prediction scripts
  chtc_open_set_detector_cascade/    # Primary cascade model
  chtc_open_set_detector_cascade_attribution_metrics/
  chtc_open_set_detector_domain_balanced/
  chtc_open_set_detector_openworld/
  chtc_open_set_detector_raid_*wandb/
final_model/                         # Saved model artifacts (RAID-trained)
abstract_only_cascade_model/         # Variant trained on abstracts only
RAID/                                # RAID dataset EDA
DeepFake/                            # MAGE/DeepFake dataset EDA
HOUND/                               # HOUND baseline results
report.pdf                           # Final report
```

## Training

```bash
cd experiments/chtc_open_set_detector_cascade
python train_open_set_chtc.py \
  --model_id StyleDistance/styledistance \
  --dataset_id liamdugan/raid \
  --dataset_splits train,extra \
  --known_models llama-chat,mpt,mpt-chat,chatgpt,gpt3,gpt4,cohere \
  --wild_unknown_models gpt2,cohere-chat \
  --val_unknown_models mistral \
  --test_unknown_models mistral-chat \
  --samples_per_label 20000 \
  --include_adversarial \
  --output_dir open_set_detector
```

Training requires a CUDA GPU. The script streams the dataset, extracts embeddings, and saves `.joblib` artifacts plus `metadata.json`.

## Inference

```bash
python experiments/chtc_open_set_detector_cascade/predict_open_set.py \
  --artifact_dir final_model \
  --text "Your text here"
```

Output JSON:
```json
{
  "final_prediction": "chatgpt",          // "human", "unknown", or a known model name
  "human_llm_prediction": "llm",
  "energy_score": -7.2,
  "energy_says_unknown": false,
  "known_llm_probabilities": { ... }
}
```

## Notes

There was equal collaboration between all team members. We worked in person and split into pairs to perform EDA on each of the two datasets, which is why only two people appear in the commit history.
