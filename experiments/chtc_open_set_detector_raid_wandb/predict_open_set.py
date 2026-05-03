#!/usr/bin/env python3
import argparse
import json
import os

import joblib
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances
from transformers import AutoModel, AutoTokenizer


def extract_embedding(text, model, tokenizer, device, max_length, hidden_layer):
    model.eval()
    encoded = tokenizer([text], return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
        hidden = outputs.hidden_states[hidden_layer]
        mask = encoded['attention_mask'].unsqueeze(-1).float()
        pooled = (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return pooled.cpu().numpy().astype(np.float32)


def nearest_centroid(x, centroids):
    labels = list(centroids.keys())
    C = np.vstack([centroids[k] for k in labels])
    D = cosine_distances(x, C)
    idx = int(D.argmin(axis=1)[0])
    return labels[idx], float(D[0, idx])


def predict(text, artifact_dir='open_set_detector'):
    with open(os.path.join(artifact_dir, 'metadata.json')) as f:
        meta = json.load(f)

    human_llm_clf = joblib.load(os.path.join(artifact_dir, 'human_llm_clf.joblib'))
    known_model_clf = joblib.load(os.path.join(artifact_dir, 'known_model_clf.joblib'))
    centroids = joblib.load(os.path.join(artifact_dir, 'centroids.joblib'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(meta['model_id'])
    model = AutoModel.from_pretrained(meta['model_id']).to(device)

    x = extract_embedding(text, model, tokenizer, device, meta['max_length'], meta.get('hidden_layer', -2))
    word_count = len(text.strip().split())

    binary_probs = human_llm_clf.predict_proba(x)[0]
    binary_classes = list(human_llm_clf.classes_)
    human_prob = float(binary_probs[binary_classes.index('human')])
    llm_prob = float(binary_probs[binary_classes.index('llm')])

    if human_prob >= llm_prob:
        return {
            'final_prediction': 'human',
            'type': 'human',
            'human_probability': human_prob,
            'llm_probability': llm_prob,
            'word_count': word_count,
        }

    nearest, dist = nearest_centroid(x, centroids)

    if word_count < meta.get('min_words_for_attribution', 80):
        return {
            'final_prediction': 'uncertain_too_short',
            'type': 'llm_likely',
            'human_probability': human_prob,
            'llm_probability': llm_prob,
            'nearest_known_model': nearest,
            'distance_to_nearest_known_model': dist,
            'unknown_threshold': meta['unknown_threshold'],
            'word_count': word_count,
        }

    if dist > meta['unknown_threshold']:
        return {
            'final_prediction': 'unknown_llm',
            'type': 'llm',
            'human_probability': human_prob,
            'llm_probability': llm_prob,
            'nearest_known_model': nearest,
            'distance_to_nearest_known_model': dist,
            'unknown_threshold': meta['unknown_threshold'],
            'word_count': word_count,
        }

    known_probs = known_model_clf.predict_proba(x)[0]
    known_classes = list(known_model_clf.classes_)
    idx = int(np.argmax(known_probs))

    return {
        'final_prediction': known_classes[idx],
        'type': 'known_llm',
        'human_probability': human_prob,
        'llm_probability': llm_prob,
        'known_model_probability': float(known_probs[idx]),
        'nearest_known_model': nearest,
        'distance_to_nearest_known_model': dist,
        'unknown_threshold': meta['unknown_threshold'],
        'word_count': word_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--artifact_dir', type=str, default='open_set_detector')
    args = parser.parse_args()
    print(json.dumps(predict(args.text, args.artifact_dir), indent=2))


if __name__ == '__main__':
    main()
