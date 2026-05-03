#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from itertools import chain

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


DEFAULT_RAID_LABELS = [
    "human",
    "chatgpt",
    "gpt4",
    "gpt3",
    "gpt2",
    "llama-chat",
    "cohere",
    "cohere-chat",
    "mpt",
    "mpt-chat",
    "mistral",
    "mistral-chat",
]

# RAID covers these eight domains. Keeping all eight in the training sample
# prevents the streamed dataset order from filling the quota with abstracts only.
DEFAULT_RAID_DOMAINS = [
    "abstracts",
    "books",
    "news",
    "poetry",
    "recipes",
    "reddit",
    "reviews",
    "wiki",
]

HUMAN_LABELS = {"human", "real", "original", "gold", "none_human"}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_print(msg):
    print(f"{now()} {msg}", flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm(x):
    return str(x).strip().lower()


def is_human_label(label):
    return norm(label) in HUMAN_LABELS


def parse_csv_list(x):
    if x is None:
        return []
    x = str(x).strip()
    if not x:
        return []
    return [v.strip().lower() for v in x.split(",") if v.strip()]


class WB:
    def __init__(self, enabled=False, project=None, entity=None, mode="online", run_name=None, config=None):
        self.enabled = bool(enabled and wandb is not None)
        self.step = 0
        if self.enabled:
            try:
                wandb.init(
                    project=project or "open-set-llm-detector",
                    entity=entity or None,
                    mode=mode or "online",
                    name=run_name or None,
                    config=config or {},
                    dir=(config or {}).get("output_dir", "."),
                )
                wandb.define_metric("global_step")
                wandb.define_metric("*")
                log_print(f"W&B initialized in {mode} mode. Run dir: {wandb.run.dir}")
            except Exception as e:
                log_print(f"W&B init failed, disabling W&B: {e}")
                self.enabled = False

    def log(self, metrics):
        if not self.enabled:
            return
        self.step += 1
        payload = {"global_step": self.step}
        payload.update(metrics)
        wandb.log(payload, step=self.step)

    def finish(self):
        if self.enabled:
            wandb.finish()

    def artifact_dir(self, path, name="open_set_detector"):
        if not self.enabled:
            return
        try:
            art = wandb.Artifact(name=name, type="model")
            art.add_dir(path)
            wandb.log_artifact(art)
        except Exception as e:
            log_print(f"W&B artifact upload failed: {e}")


def maybe_env_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token


def raid_row_to_sample(row):
    text = str(row.get("generation", row.get("text", row.get("content", ""))))
    model = norm(row.get("model", row.get("source", row.get("label", ""))))
    if is_human_label(model):
        model = "human"
    return {
        "text": text,
        "model": model,
        "source_id": str(row.get("source_id", row.get("id", ""))),
        "domain": norm(row.get("domain", "unknown")),
        "attack": norm(row.get("attack", "unknown")),
        "decoding": norm(row.get("decoding", "unknown")),
        "title": str(row.get("title", "")),
    }


def load_raid_stream(args):
    splits = parse_csv_list(args.dataset_splits)
    if not splits:
        splits = [args.dataset_split]
    streams = []
    for split in splits:
        kwargs = {"path": args.dataset_id, "split": split, "streaming": True}
        if args.dataset_config:
            kwargs["name"] = args.dataset_config
        log_print(f"Opening HF dataset {args.dataset_id}, split={split}, streaming=True")
        ds = load_dataset(**kwargs)
        if args.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed + len(streams))
        streams.append(ds)
    return chain(*streams)


def collect_samples(args, wb):
    maybe_env_token()
    target_labels = parse_csv_list(args.target_models) or DEFAULT_RAID_LABELS
    target_labels = [norm(x) for x in target_labels]
    target_set = set(target_labels)
    max_total = None if args.num_samples < 0 else args.num_samples

    target_domains = parse_csv_list(getattr(args, "target_domains", ""))
    if getattr(args, "balance_domains", False) and not target_domains:
        target_domains = DEFAULT_RAID_DOMAINS
    target_domain_set = set(target_domains)

    # Old behavior: cap only by label. New behavior: cap by (label, domain).
    domain_balance_enabled = bool(getattr(args, "balance_domains", False))
    label_quotas_enabled = args.samples_per_label > 0
    if domain_balance_enabled:
        if not target_domains:
            raise ValueError("--balance_domains requires --target_domains or the built-in DEFAULT_RAID_DOMAINS.")
        if args.samples_per_label_per_domain > 0:
            per_domain_quota = args.samples_per_label_per_domain
        elif args.samples_per_label > 0:
            per_domain_quota = int(math.ceil(args.samples_per_label / max(1, len(target_domains))))
        else:
            raise ValueError("With --balance_domains, set --samples_per_label_per_domain or --samples_per_label.")
        target_pairs = {(label, domain) for label in target_labels for domain in target_domains}
    else:
        per_domain_quota = None
        target_pairs = set()

    counts = Counter()
    pair_counts = Counter()
    domain_counts = Counter()
    attack_counts = Counter()
    rows = []
    seen = 0
    kept = 0
    t0 = time.time()

    log_print("Collecting RAID rows with stratified quotas.")
    log_print(f"Target labels: {target_labels}")
    if domain_balance_enabled:
        log_print(f"Domain balancing enabled. Target domains: {target_domains}")
        log_print(f"samples_per_label_per_domain={per_domain_quota}; expected_total={len(target_pairs) * per_domain_quota}")
    else:
        log_print("Domain balancing disabled. Using label-only quotas.")
    log_print(f"samples_per_label={args.samples_per_label}; num_samples={args.num_samples}")

    def all_label_quotas_done():
        return label_quotas_enabled and all(counts[l] >= args.samples_per_label for l in target_labels)

    def all_pair_quotas_done():
        return domain_balance_enabled and all(pair_counts[p] >= per_domain_quota for p in target_pairs)

    for raw in load_raid_stream(args):
        seen += 1
        s = raid_row_to_sample(raw)
        label = s["model"]
        domain = s["domain"]
        if not s["text"].strip():
            continue
        if target_set and label not in target_set:
            continue
        if domain_balance_enabled and domain not in target_domain_set:
            continue
        if not args.include_adversarial and s["attack"] not in {"none", "unknown", ""}:
            continue

        if domain_balance_enabled:
            pair = (label, domain)
            if pair_counts[pair] >= per_domain_quota:
                if all_pair_quotas_done():
                    break
                continue
        elif label_quotas_enabled and counts[label] >= args.samples_per_label:
            if all_label_quotas_done():
                break
            continue

        rows.append(s)
        counts[label] += 1
        pair_counts[(label, domain)] += 1
        domain_counts[domain] += 1
        attack_counts[s["attack"]] += 1
        kept += 1

        if kept % max(1, args.wandb_log_every * 100) == 0:
            wb.log({
                "data_collection/kept": kept,
                "data_collection/seen": seen,
                "data_collection/examples_per_sec": kept / max(1e-9, time.time() - t0),
                "data_collection/unique_labels": len(counts),
                "data_collection/unique_domains": len(domain_counts),
            })
            log_print(f"Collected {kept} rows after scanning {seen}. Label counts: {dict(counts)}")
            if domain_balance_enabled:
                log_print(f"  Domain counts so far: {dict(domain_counts)}")

        if max_total is not None and kept >= max_total:
            break
        if domain_balance_enabled:
            if all_pair_quotas_done():
                break
        elif all_label_quotas_done():
            break

    log_print("Final label counts:")
    for k, v in counts.most_common():
        log_print(f"  {k}: {v}")
    log_print("Final domain counts:")
    for k, v in domain_counts.most_common():
        log_print(f"  {k}: {v}")
    if domain_balance_enabled:
        log_print("Final label/domain coverage sample:")
        for (label, domain), v in sorted(pair_counts.items())[:50]:
            log_print(f"  {label}/{domain}: {v}")
        missing_or_short = [(label, domain, pair_counts[(label, domain)]) for label, domain in sorted(target_pairs) if pair_counts[(label, domain)] < per_domain_quota]
        if missing_or_short:
            log_print("WARNING: Some label/domain quotas were not filled. First 50 shortfalls:")
            for label, domain, v in missing_or_short[:50]:
                log_print(f"  {label}/{domain}: {v}/{per_domain_quota}")
    log_print("Attack counts sample: " + str(attack_counts.most_common(15)))
    wb.log({
        "data_collection/final_rows": len(rows),
        "data_collection/final_unique_labels": len(counts),
        "data_collection/final_unique_domains": len(domain_counts),
    })
    return rows

def choose_splits_by_source(samples, args):
    labels = sorted(set(s["model"] for s in samples))
    llm_labels = [l for l in labels if not is_human_label(l)]
    known = set(parse_csv_list(args.known_models))
    wild_unknown = set(parse_csv_list(getattr(args, "wild_unknown_models", "")))
    val_unknown = set(parse_csv_list(args.val_unknown_models))
    test_unknown = set(parse_csv_list(args.test_unknown_models))

    if not known:
        # Use around 60% of LLM labels as known, then split the rest into val/test unknown.
        counts = Counter(s["model"] for s in samples if not is_human_label(s["model"]))
        llm_sorted = [x for x, _ in counts.most_common()]
        n_known = max(3, int(0.60 * len(llm_sorted)))
        n_val = max(1, int(0.20 * len(llm_sorted)))
        known = set(llm_sorted[:n_known])
        remaining = llm_sorted[n_known:]
        wild_unknown = set(remaining[:max(1, min(2, len(remaining)))])
        val_unknown = set(remaining[len(wild_unknown):len(wild_unknown)+n_val])
        test_unknown = set(remaining[len(wild_unknown)+n_val:])
        if not test_unknown and len(llm_sorted) > n_known:
            test_unknown = {llm_sorted[-1]}
            known.discard(llm_sorted[-1])

    overlap = (known & wild_unknown) | (known & val_unknown) | (known & test_unknown) | (wild_unknown & val_unknown) | (wild_unknown & test_unknown) | (val_unknown & test_unknown)
    if overlap:
        raise ValueError(f"Model split sets overlap. Each model must appear in only one set. Overlap: {sorted(overlap)}")

    log_print("Using model split:")
    log_print(f"  Known models: {sorted(known)}")
    log_print(f"  Wild unknown models used for stage-1 OOD training: {sorted(wild_unknown)}")
    log_print(f"  Validation unknown models: {sorted(val_unknown)}")
    log_print(f"  Test unknown models: {sorted(test_unknown)}")

    # Group by source_id to reduce leakage across train/val/test.
    human = [s for s in samples if is_human_label(s["model"])]
    known_llm = [s for s in samples if s["model"] in known]
    wild_unk = [s for s in samples if s["model"] in wild_unknown]
    val_unk = [s for s in samples if s["model"] in val_unknown]
    test_unk = [s for s in samples if s["model"] in test_unknown]

    def group_split(rows, train_frac=0.70, val_frac=0.15):
        groups = defaultdict(list)
        for r in rows:
            sid = r["source_id"] or f"row-{id(r)}"
            groups[sid].append(r)
        keys = list(groups.keys())
        random.shuffle(keys)
        n = len(keys)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)
        train_keys = set(keys[:n_train])
        val_keys = set(keys[n_train:n_train+n_val])
        test_keys = set(keys[n_train+n_val:])
        out = {"train": [], "val": [], "test": []}
        for k, grp in groups.items():
            if k in train_keys: out["train"].extend(grp)
            elif k in val_keys: out["val"].extend(grp)
            else: out["test"].extend(grp)
        return out

    human_split = group_split(human)
    known_split = group_split(known_llm)
    wild_split = group_split(wild_unk)

    splits = {
        "llm_train": known_split["train"],
        "llm_val_known": known_split["val"],
        "llm_test_known": known_split["test"],
        "llm_wild_unknown_train": wild_split["train"],
        "llm_wild_unknown_val": wild_split["val"],
        "llm_wild_unknown_test": wild_split["test"],
        "llm_val_unknown": val_unk,
        "llm_test_unknown": test_unk,
        "human_train": human_split["train"],
        "human_val": human_split["val"],
        "human_test": human_split["test"],
    }
    log_print("Split sizes:")
    for k, v in splits.items():
        log_print(f"  {k}: {len(v)}")
    return splits, known, wild_unknown, val_unknown, test_unknown


def texts_labels(rows):
    return [r["text"] for r in rows], np.array([r["model"] for r in rows])


def extract_embeddings(texts, model, tokenizer, device, args, wb, stage):
    model.eval()
    if len(texts) == 0:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)
    out = []
    t0 = time.time()
    num_batches = math.ceil(len(texts) / args.batch_size)
    pbar = tqdm(range(0, len(texts), args.batch_size), desc=f"Embedding {stage}")
    for bi, i in enumerate(pbar, start=1):
        batch = texts[i:i+args.batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length).to(device)
        with torch.no_grad():
            if device.type == "cuda" and args.amp_dtype != "none":
                dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
                with torch.autocast(device_type="cuda", dtype=dtype):
                    outputs = model(**inputs, output_hidden_states=True)
            else:
                outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[args.embedding_layer]
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        out.append(pooled.cpu().numpy().astype(np.float32))
        if bi % args.wandb_log_every == 0 or bi == num_batches:
            done = min(i + args.batch_size, len(texts))
            elapsed = time.time() - t0
            metrics = {
                f"embedding/{stage}/examples_done": done,
                f"embedding/{stage}/percent": 100.0 * done / len(texts),
                f"embedding/{stage}/batches_done": bi,
                f"embedding/{stage}/examples_per_sec": done / max(1e-9, elapsed),
                "embedding/current_stage_index": bi,
                "time/elapsed_sec": time.time() - args.start_time,
            }
            if device.type == "cuda":
                metrics.update({
                    "gpu/memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "gpu/max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "gpu/memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                })
            wb.log(metrics)
    return np.vstack(out)


def classification_metrics_to_wandb(prefix, report_dict):
    metrics = {}
    for label, vals in report_dict.items():
        if isinstance(vals, dict):
            clean = str(label).replace(" ", "_")
            for k, v in vals.items():
                if isinstance(v, (int, float)):
                    metrics[f"{prefix}/{clean}/{k}"] = float(v)
        elif isinstance(vals, (int, float)):
            metrics[f"{prefix}/{label}"] = float(vals)
    return metrics


def fit_logreg(X, y):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))


def compute_centroids(X, y):
    centers = {}
    for label in sorted(set(y)):
        centers[str(label)] = X[np.array(y) == label].mean(axis=0)
    return centers


def nearest_centroid_scores(X, centers):
    names = list(centers.keys())
    C = np.vstack([centers[n] for n in names])
    d = cosine_distances(X, C)
    idx = d.argmin(axis=1)
    nearest = np.array([names[i] for i in idx])
    min_d = d[np.arange(len(X)), idx]
    # second nearest distance for margin/ratio
    part = np.partition(d, kth=1, axis=1)
    second = part[:, 1] if d.shape[1] > 1 else min_d
    return nearest, min_d, second, d


def class_radii(X, y, centers, quantile=0.95):
    radii = {}
    for label, c in centers.items():
        mask = np.array(y) == label
        if mask.sum() == 0:
            radii[label] = 1.0
            continue
        d = cosine_distances(X[mask], c.reshape(1, -1)).ravel()
        radii[label] = float(np.quantile(d, quantile) + 1e-8)
    return radii


def normalized_deepsvdd_score(X, centers, radii):
    nearest, min_d, second, all_d = nearest_centroid_scores(X, centers)
    normed = np.array([min_d[i] / max(radii.get(nearest[i], 1.0), 1e-8) for i in range(len(X))])
    return nearest, min_d, second, normed


def decision_logits(pipe, X):
    scaler = pipe.named_steps.get("standardscaler")
    lr = pipe.named_steps.get("logisticregression")
    Xs = scaler.transform(X) if scaler is not None else X
    z = lr.decision_function(Xs)
    if z.ndim == 1:
        z = np.stack([-z, z], axis=1)
    return z


def softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def logsumexp_np(x, axis=1):
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def energy_score(pipe, X, T=1.0):
    logits = decision_logits(pipe, X)
    # Liu et al. energy: E(x) = -T logsumexp(f(x)/T). OOD generally has higher energy.
    return -T * logsumexp_np(logits / T, axis=1)


def ood_feature_matrix(X, known_model_clf, centers, radii, energy_temperature):
    nearest, min_d, second_d, norm_d = normalized_deepsvdd_score(X, centers, radii)
    logits = decision_logits(known_model_clf, X)
    probs = softmax_np(logits)
    max_prob = probs.max(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2] if probs.shape[1] > 1 else max_prob
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    energy = -energy_temperature * logsumexp_np(logits / energy_temperature, axis=1)
    feats = np.column_stack([min_d, second_d, second_d - min_d, norm_d, energy, max_prob, entropy, margin])
    return feats


def choose_threshold(known_scores, unknown_scores, higher_is_unknown=True):
    ks = np.asarray(known_scores)
    us = np.asarray(unknown_scores)
    vals = np.unique(np.quantile(np.concatenate([ks, us]), np.linspace(0, 1, 400)))
    best = (None, -1.0)
    for t in vals:
        if higher_is_unknown:
            known_kept = np.mean(ks <= t)
            unknown_rej = np.mean(us > t)
        else:
            known_kept = np.mean(ks >= t)
            unknown_rej = np.mean(us < t)
        score = 0.5 * known_kept + 0.5 * unknown_rej
        if score > best[1]:
            best = (float(t), float(score))
    return best


def choose_threshold_with_known_fpr_cap(known_scores, unknown_scores, max_known_false_unknown=0.10, higher_is_unknown=True):
    """Choose an OOD threshold that protects known-class attribution.

    For the primary cascade we care about not rejecting too many known LLMs.
    This returns the threshold with the highest unknown-detection rate subject to
    known_false_unknown <= max_known_false_unknown, measured on the wild-val set.
    """
    ks = np.asarray(known_scores)
    us = np.asarray(unknown_scores)
    vals = np.unique(np.quantile(np.concatenate([ks, us]), np.linspace(0, 1, 1000)))
    best = None
    best_unknown = -1.0
    best_known_false = None

    for t in vals:
        if higher_is_unknown:
            known_false = float(np.mean(ks > t))
            unknown_detect = float(np.mean(us > t))
        else:
            known_false = float(np.mean(ks < t))
            unknown_detect = float(np.mean(us < t))

        if known_false <= max_known_false_unknown and unknown_detect > best_unknown:
            best = float(t)
            best_unknown = unknown_detect
            best_known_false = known_false

    # Fallback: use the most conservative threshold if no candidate satisfies the cap.
    if best is None:
        best = float(np.max(vals) if higher_is_unknown else np.min(vals))
        if higher_is_unknown:
            best_known_false = float(np.mean(ks > best))
            best_unknown = float(np.mean(us > best))
        else:
            best_known_false = float(np.mean(ks < best))
            best_unknown = float(np.mean(us < best))

    return best, {
        "known_false_unknown_rate": best_known_false,
        "unknown_detection_rate": best_unknown,
        "max_known_false_unknown": float(max_known_false_unknown),
    }


def choose_primary_energy_threshold(X_known, X_unknown, clf, centers, radii, args, wb):
    known_feats = ood_feature_matrix(X_known, clf, centers, radii, args.energy_temperature)
    unk_feats = ood_feature_matrix(X_unknown, clf, centers, radii, args.energy_temperature)
    known_energy = known_feats[:, 4]
    unk_energy = unk_feats[:, 4]

    balanced_thresh, balanced_score = choose_threshold(known_energy, unk_energy, higher_is_unknown=True)
    capped_thresh, capped_metrics = choose_threshold_with_known_fpr_cap(
        known_energy,
        unk_energy,
        max_known_false_unknown=args.energy_gate_max_known_false_unknown,
        higher_is_unknown=True,
    )

    if args.energy_gate_strategy == "balanced":
        chosen = balanced_thresh
        strategy_metrics = {
            "strategy": "balanced",
            "balanced_score": float(balanced_score),
            "known_false_unknown_rate": float(np.mean(known_energy > chosen)),
            "unknown_detection_rate": float(np.mean(unk_energy > chosen)),
        }
    else:
        chosen = capped_thresh
        strategy_metrics = {
            "strategy": "max_known_false_unknown",
            **capped_metrics,
        }

    metrics = {
        "primary_energy_gate/balanced_threshold": float(balanced_thresh),
        "primary_energy_gate/balanced_score": float(balanced_score),
        "primary_energy_gate/capped_threshold": float(capped_thresh),
        "primary_energy_gate/capped_known_false_unknown_rate": float(capped_metrics["known_false_unknown_rate"]),
        "primary_energy_gate/capped_unknown_detection_rate": float(capped_metrics["unknown_detection_rate"]),
        "primary_energy_gate/chosen_threshold": float(chosen),
        "primary_energy_gate/chosen_known_false_unknown_rate_on_wild_val": float(strategy_metrics["known_false_unknown_rate"]),
        "primary_energy_gate/chosen_unknown_detection_rate_on_wild_val": float(strategy_metrics["unknown_detection_rate"]),
    }
    wb.log(metrics)
    log_print(
        "Primary energy gate: "
        f"strategy={strategy_metrics['strategy']}, threshold={chosen:.6f}, "
        f"wild_val_known_false_unknown={strategy_metrics['known_false_unknown_rate']:.4f}, "
        f"wild_val_unknown_detection={strategy_metrics['unknown_detection_rate']:.4f}"
    )
    return float(chosen), metrics


def tune_and_eval_open_set(X_known, y_known, X_unknown, y_unknown, clf, centers, radii, args, wb, prefix):
    known_pred = clf.predict(X_known)
    rep = classification_report(y_known, known_pred, output_dict=True, zero_division=0)
    log_print(f"\n===== {prefix} known closed-set classification =====")
    log_print("\n" + classification_report(y_known, known_pred, zero_division=0))
    wb.log(classification_metrics_to_wandb(f"{prefix}/known_closed", rep))

    known_feats = ood_feature_matrix(X_known, clf, centers, radii, args.energy_temperature)
    unk_feats = ood_feature_matrix(X_unknown, clf, centers, radii, args.energy_temperature)
    feature_names = ["min_dist", "second_dist", "dist_margin", "norm_deepsvdd", "energy", "max_prob", "entropy", "prob_margin"]

    # HTAO-style score based OOD: energy. One-class style: DeepSVDD distance.
    energy_thresh, energy_bal = choose_threshold(known_feats[:, 4], unk_feats[:, 4], higher_is_unknown=True)
    svdd_thresh, svdd_bal = choose_threshold(known_feats[:, 3], unk_feats[:, 3], higher_is_unknown=True)

    # Learned rejector using validation known/OOD features. The first two methods above are still logged separately.
    Xrej = np.vstack([known_feats, unk_feats])
    yrej = np.array([0] * len(known_feats) + [1] * len(unk_feats))
    rejector = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    rejector.fit(Xrej, yrej)
    rej_prob_known = rejector.predict_proba(known_feats)[:, 1]
    rej_prob_unk = rejector.predict_proba(unk_feats)[:, 1]
    rej_thresh, rej_bal = choose_threshold(rej_prob_known, rej_prob_unk, higher_is_unknown=True)

    def summarize(name, ks, us, thresh):
        known_rej = ks > thresh
        unk_rej = us > thresh
        labels = np.concatenate([np.zeros_like(ks), np.ones_like(us)])
        scores = np.concatenate([ks, us])
        try:
            auroc = roc_auc_score(labels, scores)
        except Exception:
            auroc = float("nan")
        known_correct_not_rej = ((known_pred == y_known) & (~known_rej)).mean()
        metrics = {
            f"{prefix}/{name}/threshold": float(thresh),
            f"{prefix}/{name}/known_false_unknown_rate": float(known_rej.mean()),
            f"{prefix}/{name}/unknown_detection_rate": float(unk_rej.mean()),
            f"{prefix}/{name}/known_correct_not_rejected": float(known_correct_not_rej),
            f"{prefix}/{name}/distance_auroc": float(auroc),
        }
        wb.log(metrics)
        log_print(f"{prefix}/{name}: known_false_unknown={known_rej.mean():.4f}, unknown_detection={unk_rej.mean():.4f}, auroc={auroc:.4f}")
        return metrics

    energy_metrics = summarize("htao_energy", known_feats[:, 4], unk_feats[:, 4], energy_thresh)
    svdd_metrics = summarize("htao_deepsvdd", known_feats[:, 3], unk_feats[:, 3], svdd_thresh)
    rej_metrics = summarize("ensemble_rejector", rej_prob_known, rej_prob_unk, rej_thresh)

    log_print("Sample unknown predictions under ensemble_rejector:")
    nearest, min_d, _, norm_d = normalized_deepsvdd_score(X_unknown, centers, radii)
    for i in range(min(20, len(y_unknown))):
        log_print(str({
            "true_unknown_model": str(y_unknown[i]),
            "nearest_known_model": str(nearest[i]),
            "min_distance": float(min_d[i]),
            "normalized_deepsvdd": float(norm_d[i]),
            "energy": float(unk_feats[i, 4]),
            "rejector_prob_unknown": float(rej_prob_unk[i]),
            "rejected_as_unknown": bool(rej_prob_unk[i] > rej_thresh),
        }))

    return {
        "feature_names": feature_names,
        "energy_threshold": energy_thresh,
        "deepsvdd_threshold": svdd_thresh,
        "rejector_threshold": rej_thresh,
        "rejector": rejector,
        "energy_metrics": energy_metrics,
        "deepsvdd_metrics": svdd_metrics,
        "rejector_metrics": rej_metrics,
    }



def evaluate_open_set_fixed(X_known, y_known, X_unknown, y_unknown, clf, centers, radii, args, wb, prefix, val_bundle):
    known_pred = clf.predict(X_known)
    rep = classification_report(y_known, known_pred, output_dict=True, zero_division=0)
    log_print(f"\n===== {prefix} using wild-validation-tuned thresholds =====")
    log_print("Known closed-set classification:\n" + classification_report(y_known, known_pred, zero_division=0))
    wb.log(classification_metrics_to_wandb(f"{prefix}/known_closed", rep))
    feats_k = ood_feature_matrix(X_known, clf, centers, radii, args.energy_temperature)
    feats_u = ood_feature_matrix(X_unknown, clf, centers, radii, args.energy_temperature)
    rejector = val_bundle["rejector"]
    rej_prob_k = rejector.predict_proba(feats_k)[:, 1]
    rej_prob_u = rejector.predict_proba(feats_u)[:, 1]
    test_scores = {
        "htao_energy": (feats_k[:, 4], feats_u[:, 4], val_bundle["energy_threshold"]),
        "htao_deepsvdd": (feats_k[:, 3], feats_u[:, 3], val_bundle["deepsvdd_threshold"]),
        "wild_ood_rejector": (rej_prob_k, rej_prob_u, val_bundle["rejector_threshold"]),
    }
    all_metrics = {}
    for method, (ks, us, th) in test_scores.items():
        known_rej = ks > th
        unk_rej = us > th
        labels = np.concatenate([np.zeros_like(ks), np.ones_like(us)])
        scores = np.concatenate([ks, us])
        auroc = roc_auc_score(labels, scores) if len(set(labels)) == 2 else float("nan")
        known_correct_not_rej = ((known_pred == y_known) & (~known_rej)).mean()
        log_print(f"{prefix}/{method}: known_false_unknown={known_rej.mean():.4f}, unknown_detection={unk_rej.mean():.4f}, auroc={auroc:.4f}")
        metrics = {
            f"{prefix}/{method}/known_false_unknown_rate": float(known_rej.mean()),
            f"{prefix}/{method}/unknown_detection_rate": float(unk_rej.mean()),
            f"{prefix}/{method}/known_correct_not_rejected": float(known_correct_not_rej),
            f"{prefix}/{method}/distance_auroc": float(auroc),
        }
        wb.log(metrics)
        all_metrics[method] = metrics
    return all_metrics


def stage1_feature_matrix(X, human_llm_clf, known_model_clf, centers, radii, llm_center, args):
    feats_ood = ood_feature_matrix(X, known_model_clf, centers, radii, args.energy_temperature)
    probs_hl = human_llm_clf.predict_proba(X)
    classes = list(human_llm_clf.classes_)
    human_prob = probs_hl[:, classes.index("human")] if "human" in classes else np.zeros(len(X))
    llm_prob = probs_hl[:, classes.index("llm")] if "llm" in classes else np.ones(len(X))
    human_svdd = cosine_distances(X, llm_center.reshape(1, -1)).ravel()
    return np.column_stack([feats_ood, human_prob, llm_prob, human_svdd])


def evaluate_stage1_and_final(prefix, X_human, X_known, y_known, X_unknown, human_llm_clf, known_model_clf, centers, radii, llm_center, triage_clf, args, wb):
    X = np.vstack([X_human, X_known, X_unknown])
    y_stage1 = np.array(["human"] * len(X_human) + ["llm"] * len(X_known) + ["unknown"] * len(X_unknown))
    feats = stage1_feature_matrix(X, human_llm_clf, known_model_clf, centers, radii, llm_center, args)
    pred_stage1 = triage_clf.predict(feats)
    rep = classification_report(y_stage1, pred_stage1, output_dict=True, zero_division=0)
    log_print(f"\n===== {prefix} stage-1 triage: human vs known-LLM vs unknown =====")
    log_print("\n" + classification_report(y_stage1, pred_stage1, zero_division=0))
    wb.log(classification_metrics_to_wandb(f"{prefix}/stage1", rep))

    y_final = np.concatenate([np.array(["human"] * len(X_human)), y_known.astype(str), np.array(["unknown"] * len(X_unknown))])
    pred_final = []
    for i, p in enumerate(pred_stage1):
        if p == "human":
            pred_final.append("human")
        elif p == "unknown":
            pred_final.append("unknown")
        else:
            pred_final.append(str(known_model_clf.predict(X[i:i+1])[0]))
    pred_final = np.array(pred_final)
    rep_final = classification_report(y_final, pred_final, output_dict=True, zero_division=0)
    log_print(f"\n===== {prefix} final two-stage labels =====")
    log_print("\n" + classification_report(y_final, pred_final, zero_division=0))
    wb.log(classification_metrics_to_wandb(f"{prefix}/final_two_stage", rep_final))
    return rep, rep_final


def evaluate_human_energy_known_pipeline(
    prefix,
    X_human,
    X_known,
    y_known,
    X_unknown,
    human_llm_clf,
    known_model_clf,
    centers,
    radii,
    args,
    wb,
    energy_threshold,
):
    """Evaluate the primary production-style cascade:

    1. human_llm_clf decides human vs LLM.
    2. If human: return human.
    3. If LLM: use HTAO-style energy score as the unknown gate.
    4. If energy > threshold: return unknown.
    5. Otherwise: run known_model_clf and return the known LLM label.

    This keeps the strong known-model classifier from being bypassed by a weak
    3-way triage classifier, while still using the best OOD signal we observed.
    """
    X = np.vstack([X_human, X_known, X_unknown])
    y_final = np.concatenate([
        np.array(["human"] * len(X_human)),
        y_known.astype(str),
        np.array(["unknown"] * len(X_unknown)),
    ])

    n_h = len(X_human)
    n_k = len(X_known)
    n_u = len(X_unknown)

    human_llm_pred = human_llm_clf.predict(X)
    feats = ood_feature_matrix(X, known_model_clf, centers, radii, args.energy_temperature)
    energy = feats[:, 4]
    known_pred = known_model_clf.predict(X)

    pred_final = []
    for i in range(len(X)):
        if human_llm_pred[i] == "human":
            pred_final.append("human")
        elif energy[i] > energy_threshold:
            pred_final.append("unknown")
        else:
            pred_final.append(str(known_pred[i]))
    pred_final = np.array(pred_final)

    rep = classification_report(y_final, pred_final, output_dict=True, zero_division=0)
    log_print(f"\n===== {prefix}: human → energy unknown gate → known LLM classifier =====")
    log_print("\n" + classification_report(y_final, pred_final, zero_division=0))
    wb.log(classification_metrics_to_wandb(f"{prefix}/final_gated", rep))

    # Component/gating diagnostics.
    human_slice = slice(0, n_h)
    known_slice = slice(n_h, n_h + n_k)
    unknown_slice = slice(n_h + n_k, n_h + n_k + n_u)

    known_final = pred_final[known_slice]
    unknown_final = pred_final[unknown_slice]
    human_final = pred_final[human_slice]

    known_human_pred = human_llm_pred[known_slice]
    known_energy = energy[known_slice]
    known_model_pred = known_pred[known_slice].astype(str)
    known_true = y_known.astype(str)
    known_pass_mask = (known_human_pred != "human") & (known_energy <= energy_threshold)

    unknown_human_pred = human_llm_pred[unknown_slice]
    unknown_energy = energy[unknown_slice]
    unknown_pass_human_mask = unknown_human_pred == "human"
    unknown_energy_reject_mask = (unknown_human_pred != "human") & (unknown_energy > energy_threshold)

    raw_known_acc = float(np.mean(known_model_pred == known_true)) if n_k else float("nan")
    known_pass_rate = float(np.mean(known_pass_mask)) if n_k else float("nan")
    known_attr_acc_on_passed = (
        float(np.mean(known_model_pred[known_pass_mask] == known_true[known_pass_mask]))
        if n_k and np.any(known_pass_mask) else float("nan")
    )

    metrics = {
        f"{prefix}/gate/energy_threshold": float(energy_threshold),
        f"{prefix}/gate/human_recall": float(np.mean(human_final == "human")) if n_h else float("nan"),
        f"{prefix}/gate/known_false_human_rate": float(np.mean(known_final == "human")) if n_k else float("nan"),
        f"{prefix}/gate/known_false_unknown_rate": float(np.mean(known_final == "unknown")) if n_k else float("nan"),
        f"{prefix}/gate/known_pass_rate": known_pass_rate,
        f"{prefix}/gate/known_correct_final_rate": float(np.mean(known_final == known_true)) if n_k else float("nan"),
        f"{prefix}/gate/raw_known_model_accuracy_before_gate": raw_known_acc,
        f"{prefix}/gate/known_attr_accuracy_given_passed_gate": known_attr_acc_on_passed,
        f"{prefix}/gate/unknown_false_human_rate": float(np.mean(unknown_final == "human")) if n_u else float("nan"),
        f"{prefix}/gate/unknown_energy_reject_rate_among_not_human": float(np.mean(unknown_energy_reject_mask)) if n_u else float("nan"),
        f"{prefix}/gate/unknown_detection_rate": float(np.mean(unknown_final == "unknown")) if n_u else float("nan"),
    }
    wb.log(metrics)
    log_print(
        f"{prefix}/gate: known_false_unknown={metrics[f'{prefix}/gate/known_false_unknown_rate']:.4f}, "
        f"unknown_detection={metrics[f'{prefix}/gate/unknown_detection_rate']:.4f}, "
        f"known_pass_rate={metrics[f'{prefix}/gate/known_pass_rate']:.4f}, "
        f"known_attr_acc_given_passed={metrics[f'{prefix}/gate/known_attr_accuracy_given_passed_gate']:.4f}, "
        f"known_correct_final={metrics[f'{prefix}/gate/known_correct_final_rate']:.4f}, "
        f"human_recall={metrics[f'{prefix}/gate/human_recall']:.4f}"
    )
    log_print(
        f"{prefix}/attribution_after_gate: raw_known_acc_before_gate={raw_known_acc:.4f}, "
        f"known_pass_rate={known_pass_rate:.4f}, "
        f"known_attr_acc_given_passed_gate={known_attr_acc_on_passed:.4f}"
    )
    return {"report": rep, "metrics": metrics, "predictions": pred_final}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="StyleDistance/styledistance")
    parser.add_argument("--dataset_id", default="liamdugan/raid")
    parser.add_argument("--dataset_config", default="")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--dataset_splits", default="train,extra", help="Comma-separated RAID splits to use. test is unlabeled, so use train,extra by default.")
    parser.add_argument("--target_models", default=",".join(DEFAULT_RAID_LABELS))
    parser.add_argument("--known_models", default="llama-chat,mpt,mpt-chat,chatgpt,gpt3,gpt4,cohere")
    parser.add_argument("--wild_unknown_models", default="gpt2,cohere-chat", help="Wild OOD LLM models used to train the stage-1 unknown rejector. Inspired by wild-data OOD training.")
    parser.add_argument("--val_unknown_models", default="mistral")
    parser.add_argument("--test_unknown_models", default="mistral-chat")
    parser.add_argument("--num_samples", type=int, default=-1, help="-1 means keep scanning until per-label quotas are met.")
    parser.add_argument("--samples_per_label", type=int, default=20000, help="Balanced cap per model/human label. Set -1 to use every streamed matching row, which can be huge.")
    parser.add_argument("--balance_domains", action="store_true", help="Balance collection across RAID domains by capping each label/domain pair. This prevents early streamed domains, usually abstracts, from filling the whole label quota.")
    parser.add_argument("--target_domains", default=",".join(DEFAULT_RAID_DOMAINS), help="Comma-separated RAID domains to balance across. Defaults to the 8 RAID domains: abstracts, books, news, poetry, recipes, reddit, reviews, wiki.")
    parser.add_argument("--samples_per_label_per_domain", type=int, default=-1, help="Quota for each label/domain pair. If -1 and --balance_domains is set, this is ceil(samples_per_label / number_of_domains).")
    parser.add_argument("--include_adversarial", action="store_true", help="Include RAID adversarial attacked rows too.")
    parser.add_argument("--shuffle_buffer", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--embedding_layer", type=int, default=-2)
    parser.add_argument("--amp_dtype", choices=["fp16", "bf16", "none"], default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="open_set_detector")
    parser.add_argument("--energy_temperature", type=float, default=1.0)
    parser.add_argument("--energy_gate_strategy", choices=["max_known_false_unknown", "balanced"], default="max_known_false_unknown", help="Primary cascade energy threshold strategy. Use max_known_false_unknown to preserve known-model attribution; balanced reproduces the older balanced OOD threshold.")
    parser.add_argument("--energy_gate_max_known_false_unknown", type=float, default=0.10, help="Maximum known LLM false-unknown rate allowed when tuning the primary energy gate on wild validation data.")
    parser.add_argument("--run_triage_ablation", action="store_true", help="Also train/evaluate the old 3-way triage classifier for comparison. The primary saved predictor does not use it.")
    parser.add_argument("--svdd_radius_quantile", type=float, default=0.95)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "open-set-llm-detector"))
    parser.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--wandb_log_every", type=int, default=5)
    args = parser.parse_args()
    args.start_time = time.time()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    wb = WB(args.use_wandb, args.wandb_project, args.wandb_entity, args.wandb_mode, args.wandb_run_name, vars(args))

    log_print("===== CHTC / CUDA diagnostics =====")
    log_print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not_set')}")
    log_print(f"torch version={torch.__version__}")
    log_print(f"torch cuda available={torch.cuda.is_available()}")
    log_print(f"torch cuda device count={torch.cuda.device_count()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log_print(f"device name={torch.cuda.get_device_name(0)}")
    else:
        raise RuntimeError("No CUDA GPU visible to PyTorch. Check request_gpus and container image.")

    samples = collect_samples(args, wb)
    splits, known, wild_unknown, val_unknown, test_unknown = choose_splits_by_source(samples, args)
    wb.log({f"split_sizes/{k}": len(v) for k, v in splits.items()})

    log_print("Loading encoder/tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    encoder = AutoModel.from_pretrained(args.model_id).to(device)

    arrays = {}
    for name, rows in splits.items():
        texts, labels = texts_labels(rows)
        log_print(f"Extracting {name}: {len(texts)} examples")
        arrays[f"X_{name}"] = extract_embeddings(texts, encoder, tokenizer, device, args, wb, name)
        arrays[f"y_{name}"] = labels

    # Binary human vs LLM classifier, kept as a baseline alongside HTAO-style OOD.
    log_print("Training human-vs-LLM classifier on CPU...")
    X_binary_llm_train = np.vstack([arrays["X_llm_train"], arrays["X_llm_wild_unknown_train"]])
    X_binary_train = np.vstack([arrays["X_human_train"], X_binary_llm_train])
    y_binary_train = np.array(["human"] * len(arrays["X_human_train"]) + ["llm"] * len(X_binary_llm_train))
    human_llm_clf = fit_logreg(X_binary_train, y_binary_train)
    human_llm_clf.fit(X_binary_train, y_binary_train)

    for eval_name, X_llm, y_llm in [
        ("human_llm_val_known", arrays["X_llm_val_known"], arrays["y_llm_val_known"]),
        ("human_llm_test_known", arrays["X_llm_test_known"], arrays["y_llm_test_known"]),
        ("human_llm_wild_unknown", arrays["X_llm_wild_unknown_test"], arrays["y_llm_wild_unknown_test"]),
        ("human_llm_val_unknown", arrays["X_llm_val_unknown"], arrays["y_llm_val_unknown"]),
        ("human_llm_test_unknown", arrays["X_llm_test_unknown"], arrays["y_llm_test_unknown"]),
    ]:
        X_eval = np.vstack([arrays["X_human_test"], X_llm]) if "test" in eval_name else np.vstack([arrays["X_human_val"], X_llm])
        human_n = len(arrays["X_human_test"]) if "test" in eval_name else len(arrays["X_human_val"])
        y_eval = np.array(["human"] * human_n + ["llm"] * len(X_llm))
        pred = human_llm_clf.predict(X_eval)
        rep = classification_report(y_eval, pred, output_dict=True, zero_division=0)
        log_print(f"\n{eval_name}:\n" + classification_report(y_eval, pred, zero_division=0))
        wb.log(classification_metrics_to_wandb(eval_name, rep))

    # HTAO global DeepSVDD for human-as-OOD: fit compact center using all LLM train embeddings only.
    log_print("Fitting HTAO global DeepSVDD human detector on LLM embeddings...")
    X_all_llm_train_for_human_ood = np.vstack([arrays["X_llm_train"], arrays["X_llm_wild_unknown_train"]])
    llm_center = X_all_llm_train_for_human_ood.mean(axis=0)
    llm_train_dist = cosine_distances(X_all_llm_train_for_human_ood, llm_center.reshape(1, -1)).ravel()
    llm_val_dist = cosine_distances(np.vstack([arrays["X_llm_val_known"], arrays["X_llm_wild_unknown_val"]]), llm_center.reshape(1, -1)).ravel()
    human_val_dist = cosine_distances(arrays["X_human_val"], llm_center.reshape(1, -1)).ravel()
    human_ood_threshold, human_ood_bal = choose_threshold(llm_val_dist, human_val_dist, higher_is_unknown=True)
    wb.log({"htao_human_deepsvdd/threshold": human_ood_threshold, "htao_human_deepsvdd/balanced_score": human_ood_bal})
    log_print(f"HTAO human-as-OOD DeepSVDD threshold={human_ood_threshold:.6f}; balanced_score={human_ood_bal:.4f}")

    # Known LLM classifier.
    log_print("Training known-LLM classifier on CPU...")
    known_model_clf = fit_logreg(arrays["X_llm_train"], arrays["y_llm_train"])
    known_model_clf.fit(arrays["X_llm_train"], arrays["y_llm_train"])
    pred_val = known_model_clf.predict(arrays["X_llm_val_known"])
    log_print("\nKnown-LLM validation classification:\n" + classification_report(arrays["y_llm_val_known"], pred_val, zero_division=0))

    centers = compute_centroids(arrays["X_llm_train"], arrays["y_llm_train"])
    radii = class_radii(arrays["X_llm_train"], arrays["y_llm_train"], centers, args.svdd_radius_quantile)

    # Feed-Two-Birds style: use wild unknown models for OOD generalization/detection training.
    # The wild OOD rejector is tuned on known validation vs wild-unknown validation, then evaluated on held-out unknown models.
    val_bundle = tune_and_eval_open_set(
        arrays["X_llm_val_known"], arrays["y_llm_val_known"], arrays["X_llm_wild_unknown_val"], arrays["y_llm_wild_unknown_val"],
        known_model_clf, centers, radii, args, wb, "open_set_wild_val"
    )
    evaluate_open_set_fixed(
        arrays["X_llm_val_known"], arrays["y_llm_val_known"], arrays["X_llm_val_unknown"], arrays["y_llm_val_unknown"],
        known_model_clf, centers, radii, args, wb, "open_set_val_heldout", val_bundle
    )
    evaluate_open_set_fixed(
        arrays["X_llm_test_known"], arrays["y_llm_test_known"], arrays["X_llm_test_unknown"], arrays["y_llm_test_unknown"],
        known_model_clf, centers, radii, args, wb, "open_set_test_heldout", val_bundle
    )

    # Primary production-style two-stage/cascade decision rule:
    # human-vs-LLM gate -> HTAO energy unknown gate -> known-LLM classifier.
    # This replaces the old 3-way triage classifier as the main decision layer.
    # The primary gate is now tuned to preserve known attribution: choose the
    # best unknown-detection threshold subject to a cap on known false-unknowns.
    primary_energy_threshold, primary_energy_gate_metrics = choose_primary_energy_threshold(
        arrays["X_llm_val_known"],
        arrays["X_llm_wild_unknown_val"],
        known_model_clf,
        centers,
        radii,
        args,
        wb,
    )
    evaluate_human_energy_known_pipeline(
        "cascade_wild_val",
        arrays["X_human_val"], arrays["X_llm_val_known"], arrays["y_llm_val_known"], arrays["X_llm_wild_unknown_val"],
        human_llm_clf, known_model_clf, centers, radii, args, wb, primary_energy_threshold,
    )
    evaluate_human_energy_known_pipeline(
        "cascade_val_heldout",
        arrays["X_human_val"], arrays["X_llm_val_known"], arrays["y_llm_val_known"], arrays["X_llm_val_unknown"],
        human_llm_clf, known_model_clf, centers, radii, args, wb, primary_energy_threshold,
    )
    evaluate_human_energy_known_pipeline(
        "cascade_test_heldout",
        arrays["X_human_test"], arrays["X_llm_test_known"], arrays["y_llm_test_known"], arrays["X_llm_test_unknown"],
        human_llm_clf, known_model_clf, centers, radii, args, wb, primary_energy_threshold,
    )

    triage_clf = None
    if args.run_triage_ablation:
        # Optional ablation only: the previous 3-way triage classifier often hurt final
        # known-model attribution by bypassing the stronger known_model_clf.
        log_print("Training optional stage-1 triage ablation: human vs LLM vs unknown...")
        X_tri_train = np.vstack([arrays["X_human_train"], arrays["X_llm_train"], arrays["X_llm_wild_unknown_train"]])
        y_tri_train = np.array(["human"] * len(arrays["X_human_train"]) + ["llm"] * len(arrays["X_llm_train"]) + ["unknown"] * len(arrays["X_llm_wild_unknown_train"]))
        F_tri_train = stage1_feature_matrix(X_tri_train, human_llm_clf, known_model_clf, centers, radii, llm_center, args)
        triage_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
        triage_clf.fit(F_tri_train, y_tri_train)

        evaluate_stage1_and_final("triage_wild_val", arrays["X_human_val"], arrays["X_llm_val_known"], arrays["y_llm_val_known"], arrays["X_llm_wild_unknown_val"], human_llm_clf, known_model_clf, centers, radii, llm_center, triage_clf, args, wb)
        evaluate_stage1_and_final("triage_val_heldout", arrays["X_human_val"], arrays["X_llm_val_known"], arrays["y_llm_val_known"], arrays["X_llm_val_unknown"], human_llm_clf, known_model_clf, centers, radii, llm_center, triage_clf, args, wb)
        evaluate_stage1_and_final("triage_test_heldout", arrays["X_human_test"], arrays["X_llm_test_known"], arrays["y_llm_test_known"], arrays["X_llm_test_unknown"], human_llm_clf, known_model_clf, centers, radii, llm_center, triage_clf, args, wb)

    log_print("Saving artifacts...")
    joblib.dump(human_llm_clf, os.path.join(args.output_dir, "human_llm_clf.joblib"))
    joblib.dump(known_model_clf, os.path.join(args.output_dir, "known_model_clf.joblib"))
    joblib.dump(centers, os.path.join(args.output_dir, "class_deepsvdd_centers.joblib"))
    joblib.dump(radii, os.path.join(args.output_dir, "class_deepsvdd_radii.joblib"))
    joblib.dump(val_bundle["rejector"], os.path.join(args.output_dir, "wild_ood_rejector_energy_deepsvdd.joblib"))
    if triage_clf is not None:
        joblib.dump(triage_clf, os.path.join(args.output_dir, "stage1_triage_clf.joblib"))
    joblib.dump(llm_center, os.path.join(args.output_dir, "global_llm_deepsvdd_center.joblib"))

    metadata = {
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "dataset_splits": parse_csv_list(args.dataset_splits),
        "known_models": sorted(list(known)),
        "wild_unknown_models": sorted(list(wild_unknown)),
        "val_unknown_models": sorted(list(val_unknown)),
        "test_unknown_models": sorted(list(test_unknown)),
        "target_models": parse_csv_list(args.target_models),
        "samples_per_label": args.samples_per_label,
        "include_adversarial": bool(args.include_adversarial),
        "max_length": args.max_length,
        "embedding_layer": args.embedding_layer,
        "energy_temperature": args.energy_temperature,
        "feature_names": val_bundle["feature_names"],
        "energy_threshold": val_bundle["energy_threshold"],
        "deepsvdd_threshold": val_bundle["deepsvdd_threshold"],
        "rejector_threshold": val_bundle["rejector_threshold"],
        "primary_pipeline": "human_llm_then_htao_energy_gate_then_known_llm_classifier",
        "primary_unknown_gate": "htao_energy",
        "primary_energy_threshold": primary_energy_threshold,
        "primary_energy_gate_strategy": args.energy_gate_strategy,
        "primary_energy_gate_max_known_false_unknown": args.energy_gate_max_known_false_unknown,
        "primary_energy_gate_metrics": primary_energy_gate_metrics,
        "run_triage_ablation": bool(args.run_triage_ablation),
        "human_ood_threshold": human_ood_threshold,
        "svdd_radius_quantile": args.svdd_radius_quantile,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    wb.artifact_dir(args.output_dir)
    wb.finish()
    log_print("Done.")


if __name__ == "__main__":
    main()
