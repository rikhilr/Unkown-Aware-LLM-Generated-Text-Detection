#!/usr/bin/env python3
import argparse
import json
import os

import joblib
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances
from transformers import AutoModel, AutoTokenizer


def logsumexp_np(x, axis=1):
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def extract_embedding(text, model, tokenizer, device, max_length, embedding_layer):
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[embedding_layer]
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return pooled.cpu().numpy().astype(np.float32)


def decision_logits(pipe, X):
    scaler = pipe.named_steps.get("standardscaler")
    lr = pipe.named_steps.get("logisticregression")
    Xs = scaler.transform(X) if scaler is not None else X
    z = lr.decision_function(Xs)
    if z.ndim == 1:
        z = np.stack([-z, z], axis=1)
    return z


def ood_features(X, known_model_clf, centers, radii, T):
    names = list(centers.keys())
    C = np.vstack([centers[n] for n in names])
    d = cosine_distances(X, C)
    idx = d.argmin(axis=1)
    nearest = np.array([names[i] for i in idx])
    min_d = d[np.arange(len(X)), idx]
    part = np.partition(d, kth=1, axis=1)
    second_d = part[:, 1] if d.shape[1] > 1 else min_d
    norm_d = np.array([min_d[i] / max(radii.get(nearest[i], 1.0), 1e-8) for i in range(len(X))])
    logits = decision_logits(known_model_clf, X)
    probs = softmax_np(logits)
    max_prob = probs.max(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2] if probs.shape[1] > 1 else max_prob
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    energy = -T * logsumexp_np(logits / T, axis=1)
    feats = np.column_stack([min_d, second_d, second_d - min_d, norm_d, energy, max_prob, entropy, margin])
    return feats, nearest[0], float(min_d[0]), float(norm_d[0]), float(energy[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", default="open_set_detector")
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    with open(os.path.join(args.artifact_dir, "metadata.json")) as f:
        meta = json.load(f)

    human_llm_clf = joblib.load(os.path.join(args.artifact_dir, "human_llm_clf.joblib"))
    known_model_clf = joblib.load(os.path.join(args.artifact_dir, "known_model_clf.joblib"))
    centers = joblib.load(os.path.join(args.artifact_dir, "class_deepsvdd_centers.joblib"))
    radii = joblib.load(os.path.join(args.artifact_dir, "class_deepsvdd_radii.joblib"))
    rejector = joblib.load(os.path.join(args.artifact_dir, "ood_rejector_energy_deepsvdd.joblib"))
    global_center = joblib.load(os.path.join(args.artifact_dir, "global_llm_deepsvdd_center.joblib"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(meta["model_id"])
    encoder = AutoModel.from_pretrained(meta["model_id"]).to(device)
    X = extract_embedding(args.text, encoder, tokenizer, device, meta["max_length"], meta.get("embedding_layer", -2))

    # Binary baseline probability.
    binary_probs = human_llm_clf.predict_proba(X)[0]
    binary_classes = list(human_llm_clf.classes_)
    human_prob = float(binary_probs[binary_classes.index("human")]) if "human" in binary_classes else None
    llm_prob = float(binary_probs[binary_classes.index("llm")]) if "llm" in binary_classes else None

    # HTAO global DeepSVDD: human is OOD from LLM center.
    human_ood_score = float(cosine_distances(X, global_center.reshape(1, -1)).ravel()[0])
    htao_says_human = human_ood_score > meta["human_ood_threshold"]

    feats, nearest, min_d, norm_d, energy = ood_features(X, known_model_clf, centers, radii, meta["energy_temperature"])
    reject_prob = float(rejector.predict_proba(feats)[0, 1])
    is_unknown_llm = reject_prob > meta["rejector_threshold"]
    energy_unknown = energy > meta["energy_threshold"]
    deepsvdd_unknown = norm_d > meta["deepsvdd_threshold"]

    if htao_says_human and human_prob is not None and human_prob >= 0.5:
        final = "human"
    elif is_unknown_llm:
        final = "unknown_llm"
    else:
        final = str(known_model_clf.predict(X)[0])

    result = {
        "final_prediction": final,
        "binary_human_probability": human_prob,
        "binary_llm_probability": llm_prob,
        "htao_human_deepsvdd_score": human_ood_score,
        "htao_human_deepsvdd_threshold": meta["human_ood_threshold"],
        "nearest_known_model": nearest,
        "class_deepsvdd_min_distance": min_d,
        "class_deepsvdd_normalized_score": norm_d,
        "class_deepsvdd_threshold": meta["deepsvdd_threshold"],
        "class_deepsvdd_unknown": bool(deepsvdd_unknown),
        "energy_score": energy,
        "energy_threshold": meta["energy_threshold"],
        "energy_unknown": bool(energy_unknown),
        "ensemble_rejector_unknown_probability": reject_prob,
        "ensemble_rejector_threshold": meta["rejector_threshold"],
        "ensemble_rejector_unknown": bool(is_unknown_llm),
        "word_count": len(args.text.split()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
