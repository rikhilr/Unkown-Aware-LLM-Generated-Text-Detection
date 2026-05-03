#!/usr/bin/env python3
import argparse
import json
import os
import random
import tarfile
from collections import Counter
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

try:
    import wandb
except Exception:
    wandb = None


HUMAN_LABELS = {
    "human", "real", "original", "gold", "none", "no_model", "null", "0", "false", "h"
}

TEXT_KEYS = [
    "generation", "text", "content", "document", "article", "response", "output", "abstract"
]

MODEL_KEYS = [
    "model", "generator", "model_name", "llm", "llm_name", "source", "src"
]

BINARY_KEYS = [
    "label", "class", "is_human", "is_machine", "human", "source"
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_label(x) -> str:
    return str(x).strip().lower()


def is_missing_value(x) -> bool:
    if x is None:
        return True
    s = normalize_label(x)
    return s in {"", "nan", "none", "null"}


def is_human_label(label: str) -> bool:
    return normalize_label(label) in HUMAN_LABELS


def first_key(sample: dict, keys: List[str]) -> Optional[str]:
    for key in keys:
        if key in sample and not is_missing_value(sample[key]):
            return key
    return None


def extract_text_and_label(sample: dict) -> Tuple[str, str]:
    """
    Works for RAID and is also robust to common dataset schemas.

    RAID text rows commonly have a text/generation field and a model field where
    human rows have model='human' and machine rows have the generator model name.
    """
    text_key = first_key(sample, TEXT_KEYS)
    if text_key is None:
        raise KeyError(f"Could not find text field. Keys={list(sample.keys())}")
    text = str(sample[text_key])

    model_key = first_key(sample, MODEL_KEYS)
    binary_key = first_key(sample, BINARY_KEYS)

    model_value = normalize_label(sample[model_key]) if model_key else ""
    binary_value = normalize_label(sample[binary_key]) if binary_key else ""

    if model_value and is_human_label(model_value):
        return text, "human"

    if binary_value and is_human_label(binary_value):
        return text, "human"

    if model_value:
        return text, model_value

    if binary_value in {"machine", "ai", "llm", "generated", "1", "true"}:
        return text, "unknown_generator"

    if binary_value:
        return text, binary_value

    raise KeyError(f"Could not infer label. Keys={list(sample.keys())}")


def load_text_dataset(args) -> Tuple[List[str], np.ndarray]:
    print(f"Loading dataset: {args.dataset_id}", flush=True)

    kwargs = {
        "path": args.dataset_id,
        "split": args.dataset_split,
        "streaming": True,
    }
    if args.dataset_config:
        kwargs["name"] = args.dataset_config

    ds = load_dataset(**kwargs)

    if args.shuffle_stream:
        ds = ds.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)

    texts, labels = [], []
    skipped = 0
    first_printed = False

    pbar = tqdm(total=args.num_samples, desc="Loading/normalizing rows")
    for sample in ds:
        if len(texts) >= args.num_samples:
            break

        if not first_printed:
            print("\nFirst sample keys:", list(sample.keys()), flush=True)
            print("First sample preview:", {k: str(v)[:250] for k, v in sample.items()}, flush=True)
            first_printed = True

        try:
            text, label = extract_text_and_label(sample)
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"Skipping unparsable sample: {e}", flush=True)
            continue

        text = text.strip()
        if not text:
            continue

        texts.append(text)
        labels.append(normalize_label(label))
        pbar.update(1)
    pbar.close()

    labels = np.array(labels)
    if len(labels) == 0:
        raise RuntimeError("No usable samples loaded. Inspect dataset field names.")

    print(f"Loaded usable samples: {len(texts)}", flush=True)
    print(f"Skipped samples: {skipped}", flush=True)
    return texts, labels


def print_label_counts(labels: np.ndarray) -> None:
    print("\nLabel counts:", flush=True)
    counts = Counter(labels.tolist())
    for label, count in counts.most_common(80):
        print(f"  {label}: {count}", flush=True)


def parse_label_set(s: str) -> Optional[set]:
    if not s:
        return None
    return {normalize_label(x) for x in s.split(",") if x.strip()}


def choose_model_splits(labels: np.ndarray, args) -> Tuple[set, set, set]:
    manual_known = parse_label_set(args.known_models)
    manual_val_unknown = parse_label_set(args.val_unknown_models)
    manual_test_unknown = parse_label_set(args.test_unknown_models)

    if manual_known and manual_val_unknown and manual_test_unknown:
        return manual_known, manual_val_unknown, manual_test_unknown

    counts = Counter(labels.tolist())
    llm_labels = [lab for lab in counts if not is_human_label(lab)]
    llm_labels = [lab for lab in llm_labels if counts[lab] >= args.min_samples_per_llm]
    llm_labels = sorted(llm_labels, key=lambda x: counts[x], reverse=True)

    if len(llm_labels) < 3:
        raise RuntimeError(
            f"Need at least 3 non-human LLM labels with >= {args.min_samples_per_llm} samples. "
            f"Found: {llm_labels}"
        )

    num_known = max(2, int(args.known_fraction * len(llm_labels)))
    num_val_unknown = max(1, int(args.val_unknown_fraction * len(llm_labels)))

    known = set(llm_labels[:num_known])
    val_unknown = set(llm_labels[num_known:num_known + num_val_unknown])
    test_unknown = set(llm_labels[num_known + num_val_unknown:])

    if not test_unknown:
        moved = sorted(known)[-1]
        known.remove(moved)
        test_unknown.add(moved)

    return known, val_unknown, test_unknown


def split_data(texts: List[str], labels: np.ndarray, known: set, val_unknown: set, test_unknown: set, seed: int):
    rng = np.random.default_rng(seed)
    splits = {
        "llm_train": ([], []),
        "llm_val_known": ([], []),
        "llm_test_known": ([], []),
        "llm_val_unknown": ([], []),
        "llm_test_unknown": ([], []),
        "human_train": ([], []),
        "human_val": ([], []),
        "human_test": ([], []),
    }

    for text, label in zip(texts, labels):
        label = normalize_label(label)
        r = rng.random()

        if is_human_label(label):
            if r < 0.70:
                key = "human_train"
            elif r < 0.85:
                key = "human_val"
            else:
                key = "human_test"
            splits[key][0].append(text)
            splits[key][1].append("human")
            continue

        if label in known:
            if r < 0.70:
                key = "llm_train"
            elif r < 0.85:
                key = "llm_val_known"
            else:
                key = "llm_test_known"
        elif label in val_unknown:
            key = "llm_val_unknown"
        elif label in test_unknown:
            key = "llm_test_unknown"
        else:
            continue

        splits[key][0].append(text)
        splits[key][1].append(label)

    return {k: (v[0], np.array(v[1])) for k, v in splits.items()}


class WB:
    def __init__(self, enabled: bool):
        self.enabled = enabled and wandb is not None

    def log(self, metrics: dict):
        if self.enabled:
            wandb.log(metrics)

    def finish(self):
        if self.enabled:
            wandb.finish()


def init_wandb(args, known, val_unknown, test_unknown) -> WB:
    if not args.use_wandb:
        return WB(False)
    if wandb is None:
        print("wandb package is not installed, continuing without W&B.", flush=True)
        return WB(False)

    mode = args.wandb_mode or os.environ.get("WANDB_MODE", "online")
    project = args.wandb_project or os.environ.get("WANDB_PROJECT", "open-set-llm-detector")
    entity = args.wandb_entity or os.environ.get("WANDB_ENTITY", None) or None

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            mode=mode,
            name=args.wandb_run_name,
            dir=args.output_dir,
            config={
                **vars(args),
                "known_models": sorted(list(known)),
                "val_unknown_models": sorted(list(val_unknown)),
                "test_unknown_models": sorted(list(test_unknown)),
            },
        )
        print(f"W&B initialized in {mode} mode. Run dir: {run.dir}", flush=True)
        return WB(True)
    except Exception as e:
        print(f"W&B online init failed: {e}", flush=True)
        print("Falling back to W&B offline mode.", flush=True)
        run = wandb.init(
            project=project,
            entity=entity,
            mode="offline",
            name=args.wandb_run_name,
            dir=args.output_dir,
            config=vars(args),
        )
        print(f"W&B initialized in offline mode. Run dir: {run.dir}", flush=True)
        return WB(True)


def print_cuda_diagnostics() -> None:
    print("\n===== CHTC / CUDA diagnostics =====", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not_set')}", flush=True)
    print(f"torch version={torch.__version__}", flush=True)
    print(f"torch cuda available={torch.cuda.is_available()}", flush=True)
    print(f"torch cuda device count={torch.cuda.device_count()}", flush=True)
    if torch.cuda.is_available():
        print(f"current device={torch.cuda.current_device()}", flush=True)
        print(f"device name={torch.cuda.get_device_name(0)}", flush=True)
    os.system("nvidia-smi || true")


def autocast_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return torch.amp.autocast(device_type="cpu", enabled=False)
    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def extract_embeddings(texts: List[str], model, tokenizer, device, args, wb: WB, split_name: str) -> np.ndarray:
    if len(texts) == 0:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)

    model.eval()
    all_embeddings = []
    total_batches = (len(texts) + args.batch_size - 1) // args.batch_size

    pbar = tqdm(range(0, len(texts), args.batch_size), desc=f"Embedding {split_name}")
    for batch_idx, start in enumerate(pbar):
        batch = texts[start:start + args.batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(device)

        with torch.no_grad():
            with autocast_context(device, args.amp_dtype):
                outputs = model(**encoded, output_hidden_states=True)
                hidden = outputs.hidden_states[args.hidden_layer]
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        all_embeddings.append(pooled.detach().cpu().numpy().astype(np.float32))

        if batch_idx % args.wandb_log_every == 0:
            metrics = {
                f"embedding/{split_name}/batch": batch_idx,
                f"embedding/{split_name}/percent": 100.0 * (batch_idx + 1) / total_batches,
            }
            if device.type == "cuda":
                metrics.update({
                    "gpu/memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "gpu/max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                })
            wb.log(metrics)

    return np.vstack(all_embeddings)


def report_to_wandb(prefix: str, y_true, y_pred, wb: WB) -> None:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics = {}
    for k in ["accuracy", "macro avg", "weighted avg"]:
        if k == "accuracy":
            metrics[f"{prefix}/accuracy"] = report.get("accuracy", 0.0)
        elif k in report:
            for metric_name, value in report[k].items():
                metrics[f"{prefix}/{k.replace(' ', '_')}/{metric_name}"] = value
    wb.log(metrics)


def compute_centroids(X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    centroids = {}
    for label in sorted(set(y.tolist())):
        idx = y == label
        centroids[label] = X[idx].mean(axis=0)
    return centroids


def nearest_centroid_predict(X: np.ndarray, centroids: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    labels = list(centroids.keys())
    C = np.vstack([centroids[k] for k in labels])
    D = cosine_distances(X, C)
    nearest = D.argmin(axis=1)
    return np.array([labels[i] for i in nearest]), D.min(axis=1)


def choose_unknown_threshold(known_dist: np.ndarray, unknown_dist: np.ndarray) -> Tuple[float, float]:
    all_d = np.concatenate([known_dist, unknown_dist])
    candidates = np.linspace(float(all_d.min()), float(all_d.max()), 400)
    best_t, best_score = float(candidates[0]), -1.0
    for t in candidates:
        known_kept = np.mean(known_dist <= t)
        unknown_rejected = np.mean(unknown_dist > t)
        score = 0.5 * known_kept + 0.5 * unknown_rejected
        if score > best_score:
            best_t, best_score = float(t), float(score)
    return best_t, best_score


def evaluate_open_set(prefix: str, X_known, y_known, X_unknown, y_unknown, clf, centroids, threshold, wb: WB):
    print(f"\n===== {prefix} open-set evaluation =====", flush=True)

    y_closed = clf.predict(X_known)
    print("Known closed-set classification:", flush=True)
    print(classification_report(y_known, y_closed, zero_division=0), flush=True)
    report_to_wandb(f"{prefix}/known_closed", y_known, y_closed, wb)

    _, known_dist = nearest_centroid_predict(X_known, centroids)
    nearest_unknown, unknown_dist = nearest_centroid_predict(X_unknown, centroids)

    known_rejected = known_dist > threshold
    unknown_rejected = unknown_dist > threshold

    metrics = {
        f"{prefix}/known_false_unknown_rate": float(np.mean(known_rejected)),
        f"{prefix}/unknown_detection_rate": float(np.mean(unknown_rejected)),
        f"{prefix}/known_correct_not_rejected": float(np.mean((y_closed == y_known) & (~known_rejected))),
    }

    try:
        labels = np.concatenate([np.zeros_like(known_dist), np.ones_like(unknown_dist)])
        scores = np.concatenate([known_dist, unknown_dist])
        metrics[f"{prefix}/distance_auroc"] = float(roc_auc_score(labels, scores))
    except Exception:
        pass

    wb.log(metrics)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}", flush=True)

    print("\nSample unknown nearest-known predictions:", flush=True)
    for true_label, near, dist, rejected in zip(y_unknown[:20], nearest_unknown[:20], unknown_dist[:20], unknown_rejected[:20]):
        print({
            "true_unknown_model": str(true_label),
            "nearest_known_model": str(near),
            "distance": float(dist),
            "rejected_as_unknown": bool(rejected),
        }, flush=True)

    return metrics


def save_json(path: str, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, default="liamdugan/raid")
    parser.add_argument("--dataset_config", type=str, default="")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--model_id", type=str, default="StyleDistance/styledistance")
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--hidden_layer", type=int, default=-2)
    parser.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16", "none"])
    parser.add_argument("--output_dir", type=str, default="open_set_detector")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_stream", action="store_true", default=True)
    parser.add_argument("--shuffle_buffer", type=int, default=10000)
    parser.add_argument("--min_samples_per_llm", type=int, default=100)
    parser.add_argument("--known_fraction", type=float, default=0.60)
    parser.add_argument("--val_unknown_fraction", type=float, default=0.20)
    parser.add_argument("--known_models", type=str, default="")
    parser.add_argument("--val_unknown_models", type=str, default="")
    parser.add_argument("--test_unknown_models", type=str, default="")
    parser.add_argument("--min_words_for_attribution", type=int, default=80)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_log_every", type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print_cuda_diagnostics()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("No CUDA GPU visible. Check HTCondor request_gpus and container image.")

    texts, labels = load_text_dataset(args)
    print_label_counts(labels)

    known, val_unknown, test_unknown = choose_model_splits(labels, args)
    print("\nUsing split:", flush=True)
    print(f"Known models: {sorted(list(known))}", flush=True)
    print(f"Validation unknown models: {sorted(list(val_unknown))}", flush=True)
    print(f"Test unknown models: {sorted(list(test_unknown))}", flush=True)

    wb = init_wandb(args, known, val_unknown, test_unknown)

    splits = split_data(texts, labels, known, val_unknown, test_unknown, args.seed)
    print("\nSplit sizes:", flush=True)
    for name, (tx, y) in splits.items():
        print(f"  {name}: {len(tx)}", flush=True)
        wb.log({f"split_size/{name}": len(tx)})

    if len(splits["human_train"][0]) == 0:
        raise RuntimeError("No human training samples loaded. RAID should include human rows; try increasing --num_samples or --shuffle_buffer.")
    if len(splits["llm_train"][0]) == 0:
        raise RuntimeError("No known LLM training samples loaded. Check label parsing and split settings.")
    if len(splits["llm_val_unknown"][0]) == 0 or len(splits["llm_test_unknown"][0]) == 0:
        raise RuntimeError("No unknown validation/test LLM samples. Need at least a few LLM labels.")

    print("\nLoading encoder/tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    encoder = AutoModel.from_pretrained(args.model_id).to(device)
    encoder.eval()

    print("\nExtracting embeddings on GPU...", flush=True)
    X_llm_train = extract_embeddings(splits["llm_train"][0], encoder, tokenizer, device, args, wb, "llm_train")
    y_llm_train = splits["llm_train"][1]

    X_llm_val_known = extract_embeddings(splits["llm_val_known"][0], encoder, tokenizer, device, args, wb, "llm_val_known")
    y_llm_val_known = splits["llm_val_known"][1]

    X_llm_test_known = extract_embeddings(splits["llm_test_known"][0], encoder, tokenizer, device, args, wb, "llm_test_known")
    y_llm_test_known = splits["llm_test_known"][1]

    X_llm_val_unknown = extract_embeddings(splits["llm_val_unknown"][0], encoder, tokenizer, device, args, wb, "llm_val_unknown")
    y_llm_val_unknown = splits["llm_val_unknown"][1]

    X_llm_test_unknown = extract_embeddings(splits["llm_test_unknown"][0], encoder, tokenizer, device, args, wb, "llm_test_unknown")
    y_llm_test_unknown = splits["llm_test_unknown"][1]

    X_human_train = extract_embeddings(splits["human_train"][0], encoder, tokenizer, device, args, wb, "human_train")
    X_human_val = extract_embeddings(splits["human_val"][0], encoder, tokenizer, device, args, wb, "human_val")
    X_human_test = extract_embeddings(splits["human_test"][0], encoder, tokenizer, device, args, wb, "human_test")

    print("\nTraining human-vs-LLM classifier on CPU...", flush=True)
    X_binary_train = np.vstack([X_human_train, X_llm_train])
    y_binary_train = np.array(["human"] * len(X_human_train) + ["llm"] * len(X_llm_train))

    X_binary_val = np.vstack([X_human_val, X_llm_val_known])
    y_binary_val = np.array(["human"] * len(X_human_val) + ["llm"] * len(X_llm_val_known))

    X_binary_test_known = np.vstack([X_human_test, X_llm_test_known])
    y_binary_test_known = np.array(["human"] * len(X_human_test) + ["llm"] * len(X_llm_test_known))

    human_llm_clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=1),
    )
    human_llm_clf.fit(X_binary_train, y_binary_train)

    print("\nHuman-vs-LLM validation on known LLMs:", flush=True)
    pred_val = human_llm_clf.predict(X_binary_val)
    print(classification_report(y_binary_val, pred_val, zero_division=0), flush=True)
    report_to_wandb("human_llm_val_known", y_binary_val, pred_val, wb)

    print("\nHuman-vs-LLM test on known LLMs:", flush=True)
    pred_test = human_llm_clf.predict(X_binary_test_known)
    print(classification_report(y_binary_test_known, pred_test, zero_division=0), flush=True)
    report_to_wandb("human_llm_test_known", y_binary_test_known, pred_test, wb)

    print("\nHuman-vs-LLM test on unknown LLMs:", flush=True)
    pred_unknown_binary = human_llm_clf.predict(X_llm_test_unknown)
    y_unknown_binary = np.array(["llm"] * len(X_llm_test_unknown))
    print(classification_report(y_unknown_binary, pred_unknown_binary, zero_division=0), flush=True)
    report_to_wandb("human_llm_test_unknown_llms", y_unknown_binary, pred_unknown_binary, wb)

    print("\nTraining known-LLM classifier on CPU...", flush=True)
    known_llm_clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", multi_class="auto", n_jobs=1),
    )
    known_llm_clf.fit(X_llm_train, y_llm_train)

    print("\nKnown-LLM validation classification:", flush=True)
    known_val_pred = known_llm_clf.predict(X_llm_val_known)
    print(classification_report(y_llm_val_known, known_val_pred, zero_division=0), flush=True)
    report_to_wandb("known_llm_val", y_llm_val_known, known_val_pred, wb)

    centroids = compute_centroids(X_llm_train, y_llm_train)
    _, val_known_dist = nearest_centroid_predict(X_llm_val_known, centroids)
    _, val_unknown_dist = nearest_centroid_predict(X_llm_val_unknown, centroids)
    threshold, threshold_score = choose_unknown_threshold(val_known_dist, val_unknown_dist)

    print(f"\nChosen unknown threshold: {threshold:.6f}", flush=True)
    print(f"Validation threshold score: {threshold_score:.4f}", flush=True)
    wb.log({"unknown_threshold": threshold, "threshold_score": threshold_score})

    val_metrics = evaluate_open_set("open_set_val", X_llm_val_known, y_llm_val_known, X_llm_val_unknown, y_llm_val_unknown, known_llm_clf, centroids, threshold, wb)
    test_metrics = evaluate_open_set("open_set_test", X_llm_test_known, y_llm_test_known, X_llm_test_unknown, y_llm_test_unknown, known_llm_clf, centroids, threshold, wb)

    print("\nSaving artifacts...", flush=True)
    joblib.dump(human_llm_clf, os.path.join(args.output_dir, "human_llm_clf.joblib"))
    joblib.dump(known_llm_clf, os.path.join(args.output_dir, "known_model_clf.joblib"))
    joblib.dump(centroids, os.path.join(args.output_dir, "centroids.joblib"))

    metadata = {
        "dataset_id": args.dataset_id,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "model_id": args.model_id,
        "known_models": sorted(list(known)),
        "val_unknown_models": sorted(list(val_unknown)),
        "test_unknown_models": sorted(list(test_unknown)),
        "unknown_threshold": threshold,
        "min_words_for_attribution": args.min_words_for_attribution,
        "max_length": args.max_length,
        "hidden_layer": args.hidden_layer,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(os.path.join(args.output_dir, "metadata.json"), metadata)

    if wb.enabled:
        try:
            artifact = wandb.Artifact("open_set_detector", type="model")
            artifact.add_dir(args.output_dir)
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"Could not upload W&B artifact: {e}", flush=True)

    print("Done.", flush=True)
    wb.finish()


if __name__ == "__main__":
    main()
