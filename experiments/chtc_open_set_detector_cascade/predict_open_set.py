#!/usr/bin/env python3
import argparse
import json
import os

import joblib
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances
from transformers import AutoModel, AutoTokenizer


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_embedding(text, model, tokenizer, device, max_length, layer):
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer]
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


def softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def logsumexp_np(x, axis=1):
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def nearest_centroid_scores(X, centers):
    names = list(centers.keys())
    C = np.vstack([centers[n] for n in names])
    d = cosine_distances(X, C)
    idx = d.argmin(axis=1)
    nearest = np.array([names[i] for i in idx])
    min_d = d[np.arange(len(X)), idx]
    part = np.partition(d, kth=1, axis=1)
    second = part[:, 1] if d.shape[1] > 1 else min_d
    return nearest, min_d, second, d


def normalized_deepsvdd_score(X, centers, radii):
    nearest, min_d, second, _ = nearest_centroid_scores(X, centers)
    normed = np.array([min_d[i] / max(radii.get(nearest[i], 1.0), 1e-8) for i in range(len(X))])
    return nearest, min_d, second, normed


def ood_feature_matrix(X, known_model_clf, centers, radii, energy_temperature):
    nearest, min_d, second_d, norm_d = normalized_deepsvdd_score(X, centers, radii)
    logits = decision_logits(known_model_clf, X)
    probs = softmax_np(logits)
    max_prob = probs.max(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2] if probs.shape[1] > 1 else max_prob
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    energy = -energy_temperature * logsumexp_np(logits / energy_temperature, axis=1)
    return np.column_stack([min_d, second_d, second_d - min_d, norm_d, energy, max_prob, entropy, margin])


def stage1_feature_matrix(X, human_llm_clf, known_model_clf, centers, radii, llm_center, energy_temperature):
    feats_ood = ood_feature_matrix(X, known_model_clf, centers, radii, energy_temperature)
    probs_hl = human_llm_clf.predict_proba(X)
    classes = list(human_llm_clf.classes_)
    human_prob = probs_hl[:, classes.index("human")] if "human" in classes else np.zeros(len(X))
    llm_prob = probs_hl[:, classes.index("llm")] if "llm" in classes else np.ones(len(X))
    human_svdd = cosine_distances(X, llm_center.reshape(1, -1)).ravel()
    return np.column_stack([feats_ood, human_prob, llm_prob, human_svdd])


def predict(text, artifact_dir):
    meta = load_json(os.path.join(artifact_dir, "metadata.json"))
    model_id = meta["model_id"]
    max_length = meta.get("max_length", 512)
    layer = meta.get("embedding_layer", -2)
    energy_temperature = meta.get("energy_temperature", 1.0)

    human_llm_clf = joblib.load(os.path.join(artifact_dir, "human_llm_clf.joblib"))
    known_model_clf = joblib.load(os.path.join(artifact_dir, "known_model_clf.joblib"))
    centers = joblib.load(os.path.join(artifact_dir, "class_deepsvdd_centers.joblib"))
    radii = joblib.load(os.path.join(artifact_dir, "class_deepsvdd_radii.joblib"))
    # The primary predictor is now a cascade:
    # human-vs-LLM gate -> HTAO energy unknown gate -> known-LLM classifier.
    # The old 3-way triage model is optional and is not required for prediction.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoder = AutoModel.from_pretrained(model_id).to(device)
    X = extract_embedding(text, encoder, tokenizer, device, max_length, layer)

    # Stage 1: human vs LLM.
    human_llm_probs = human_llm_clf.predict_proba(X)[0]
    human_llm_classes = list(human_llm_clf.classes_)
    human_llm_pred = human_llm_classes[int(np.argmax(human_llm_probs))]
    human_llm_prob_map = {c: float(p) for c, p in zip(human_llm_classes, human_llm_probs)}

    # Stage 2a: only if LLM, decide known vs unknown using HTAO energy.
    feats = ood_feature_matrix(X, known_model_clf, centers, radii, energy_temperature)
    energy = float(feats[0, 4])
    energy_threshold = float(meta.get("primary_energy_threshold", meta.get("energy_threshold")))
    is_unknown = energy > energy_threshold

    result = {
        "pipeline": meta.get("primary_pipeline", "human_llm_then_htao_energy_gate_then_known_llm_classifier"),
        "known_models": meta.get("known_models", []),
        "human_llm_prediction": human_llm_pred,
        "human_llm_probabilities": human_llm_prob_map,
        "energy_score": energy,
        "energy_threshold": energy_threshold,
        "energy_says_unknown": bool(is_unknown),
    }

    if human_llm_pred == "human":
        result["final_prediction"] = "human"
        return result

    if is_unknown:
        result["final_prediction"] = "unknown"
        return result

    # Stage 2b: known LLM family attribution.
    probs = known_model_clf.predict_proba(X)[0]
    classes = list(known_model_clf.classes_)
    best = int(np.argmax(probs))
    result.update({
        "final_prediction": classes[best],
        "known_llm_probability": float(probs[best]),
        "known_llm_probabilities": {c: float(p) for c, p in zip(classes, probs)},
    })
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", default="open_set_detector")
    ap.add_argument("--text", required=True)
    args = ap.parse_args()
    print(json.dumps(predict(args.text, args.artifact_dir), indent=2))


if __name__ == "__main__":
    main()
