import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import joblib

from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from transformers import (
    RobertaForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


def train(run_name: str, use_wandb: bool = False):

    RESULTS_DIR = Path("results") / run_name
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(42)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    print(f"[Rank {local_rank}/{world_size}] Starting")

    is_main = local_rank == 0
    use_wandb = use_wandb and is_main

    if use_wandb:
        wandb.init(project="HOUND", name=run_name)

    # -----------------------
    # Dataset
    # -----------------------
    dataset = load_dataset("Shengkun/Raid_split", split="train")

    model_name = "FacebookAI/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # -----------------------
    # Label encoding
    # -----------------------
    le = LabelEncoder()
    le.fit(dataset["model"])

    joblib.dump(le, RESULTS_DIR / "label_encoder.pkl")

    def encode_labels(batch):
        return {"labels": le.transform(batch["model"]).tolist()}

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
            max_length=256,
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    train_dataset = dataset["train"].shuffle(seed=42)
    eval_dataset = dataset["test"]  # no need to shuffle eval

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # -----------------------
    # Model
    # -----------------------
    id2label = {i: c for i, c in enumerate(le.classes_)}
    label2id = {c: i for i, c in enumerate(le.classes_)}

    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(le.classes_),
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch.bfloat16,
    )

    # -----------------------
    # Weighted loss trainer
    # -----------------------
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            w = class_weights.to(outputs.logits.device)
            loss = F.cross_entropy(outputs.logits.float(), labels, weight=w)
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
        eval_strategy="epoch",
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

    trainer = WeightedTrainer(
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
    # Save
    # -----------------------
    if is_main:
        final_dir = RESULTS_DIR / "final_model"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        joblib.dump(le, final_dir / "label_encoder.pkl")

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