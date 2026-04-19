# Modeled after DeTeCtive paper

import argparse
import logging
import os
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from pytorch_metric_learning.losses import SupConLoss
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    DebertaV2ForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from labels import FAMILY_MAP

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
EMBEDDING_DIM = 768


def compute_prototypes(model, dataset, le, device, batch_size=64):
    """
    After training, compute per-class centroid embeddings from [CLS] tokens.
    Used at inference instead of argmax over logits.
    """
    model.eval()
    sums = defaultdict(lambda: torch.zeros(EMBEDDING_DIM))
    counts = defaultdict(int)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda x: {k: torch.tensor([d[k] for d in x]) for k in x[0]},
    )

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, output_hidden_states=True)
            # [CLS] token from last hidden state, L2-normalized
            embs = F.normalize(out.hidden_states[-1][:, 0, :].float(), dim=-1).cpu()
            for emb, label in zip(embs, labels):
                sums[label.item()] += emb
                counts[label.item()] += 1

    prototypes = {le.classes_[k]: F.normalize(sums[k] / counts[k], dim=0) for k in sums}
    return prototypes


def train(run_name: str, use_wandb: bool = False):

    RESULTS_DIR = Path("results") / run_name
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(42)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = local_rank == 0

    print(f"[Rank {local_rank}/{world_size}] Starting")

    use_wandb = use_wandb and is_main
    if use_wandb:
        wandb.init(project="HOUND", name=run_name)

    # -----------------------
    # Dataset
    # -----------------------
    dataset = load_dataset("Shengkun/Raid_split", split="train")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # -----------------------
    # Label encoding
    # -----------------------
    le = LabelEncoder()
    le.fit(dataset["model"])

    id2label = {i: c for i, c in enumerate(le.classes_)}
    label2id = {c: i for i, c in enumerate(le.classes_)}

    family_le = LabelEncoder()
    all_families = [FAMILY_MAP.get(m, "Unknown") for m in dataset["model"]]
    family_le.fit(all_families)

    def encode_labels(batch):
        model_ids = le.transform(batch["model"]).tolist()
        family_names = [FAMILY_MAP.get(m, "Unknown") for m in batch["model"]]
        family_ids = family_le.transform(family_names).tolist()
        return {
            "labels": model_ids,
            "family_labels": family_ids,  # This fuels the PhantomHunter head
        }

    dataset = dataset.map(encode_labels, batched=True)

    # -----------------------
    # Class weights
    # -----------------------
    class_weights = torch.tensor(
        compute_class_weight("balanced", classes=le.classes_, y=dataset["model"]),
        dtype=torch.float32,
    )

    # -----------------------
    # Tokenization
    # -----------------------
    def tokenize(batch):
        return tokenizer(
            batch["generation"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    train_dataset = dataset["train"].shuffle(seed=42)
    eval_dataset = dataset["test"]

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # -----------------------
    # Model — no torch_dtype here for DeBERTa-v3; bf16 set in TrainingArguments
    # -----------------------
    model = DebertaV2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(le.classes_),
        id2label=id2label,
        label2id=label2id,
    )

    # -----------------------
    # Supervised Contrastive + Weighted CE trainer
    # -----------------------
    supcon_loss_fn = SupConLoss(temperature=0.07)

    class SupConWeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")  # Specific Model (e.g., Llama-2-70b)

            # New: Get the Family Label (e.g., Llama)
            # You'll need to pass this in your dataset mapping
            family_labels = inputs.get("family_labels")

            outputs = model(**inputs, output_hidden_states=True)
            embeddings = F.normalize(outputs.hidden_states[-1][:, 0, :].float(), dim=-1)

            # 1. PhantomHunter Style: Contrastive loss on FAMILIES
            con_loss = supcon_loss_fn(embeddings, family_labels)

            # 2. Attribution: Weighted Cross-Entropy on specific MODELS
            w = class_weights.to(outputs.logits.device)
            ce_loss = F.cross_entropy(outputs.logits.float(), labels, weight=w)

            # Joint optimization (50/50 balance)
            loss = 0.5 * con_loss + 0.5 * ce_loss
            return (loss, outputs) if return_outputs else loss

    # -----------------------
    # Metrics
    # -----------------------
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds)}

    # -----------------------
    # Training args
    # -----------------------
    training_args = TrainingArguments(
        output_dir=str(RESULTS_DIR),
        num_train_epochs=4,
        learning_rate=2e-5,
        warmup_ratio=0.06,
        weight_decay=0.01,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        bf16=True,
        ddp_find_unused_parameters=False,
        report_to="wandb" if use_wandb else "none",
    )

    trainer = SupConWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    last_checkpoint = get_last_checkpoint(str(RESULTS_DIR))
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # -----------------------
    # Save model + prototypes
    # -----------------------
    if is_main:
        final_dir = RESULTS_DIR / "final_model"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        joblib.dump(le, final_dir / "label_encoder.pkl")

        print("Computing prototypes on training set...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prototypes = compute_prototypes(trainer.model, train_dataset, le, device)
        joblib.dump(prototypes, final_dir / "prototypes.pkl")
        print(f"Saved {len(prototypes)} class prototypes.")

    if use_wandb:
        wandb.finish()


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("run_name")
    parser.add_argument("-w", "--use_wandb", action="store_true")

    args = parser.parse_args()
    train(args.run_name, args.use_wandb)


if __name__ == "__main__":
    main()
