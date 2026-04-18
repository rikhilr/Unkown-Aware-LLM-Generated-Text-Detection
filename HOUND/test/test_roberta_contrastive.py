import torch
import torch.nn.functional as F
import joblib
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict


def load_model(model_dir: str):
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = DebertaV2ForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    le = joblib.load(model_dir / "label_encoder.pkl")
    prototypes = joblib.load(model_dir / "prototypes.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    prototypes = {k: v.to(device) for k, v in prototypes.items()}

    return model, tokenizer, le, prototypes, device


def predict(text: str, model, tokenizer, le, prototypes, device, max_length: int = 256):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    # Logit-based prediction
    logit_pred = le.classes_[out.logits.float().argmax(-1).item()]

    # Prototype-based prediction
    emb = F.normalize(out.hidden_states[-1][:, 0, :].float(), dim=-1)
    scores = {label: (emb @ proto.unsqueeze(-1)).item() for label, proto in prototypes.items()}
    proto_pred = max(scores, key=scores.get)

    return logit_pred, proto_pred, scores


if __name__ == "__main__":
    MODEL_DIR = "fine-tuned-models/run_deberta/final_model"
    model, tokenizer, le, prototypes, device = load_model(MODEL_DIR)

    train_ds = load_dataset("Shengkun/Raid_split", split="train")
    test_ds  = load_dataset("Shengkun/Raid_split", split="test")

    # -------------------------------------------------------
    # 1. Label overlap
    # -------------------------------------------------------
    train_models = set(train_ds["model"])
    test_models  = set(test_ds["model"])

    print("=" * 60)
    print("LABEL OVERLAP")
    print(f"  Train labels ({len(train_models)}): {sorted(train_models)}")
    print(f"  Test labels  ({len(test_models)}):  {sorted(test_models)}")
    print(f"  In test but NOT in train: {test_models - train_models}")
    print(f"  In train but NOT in test: {train_models - test_models}")
    print()

    # -------------------------------------------------------
    # 2. Logit vs prototype accuracy + per-class breakdown
    # -------------------------------------------------------
    N = 1000
    logit_correct  = 0
    proto_correct  = 0

    # per-class: {true_label: [logit_hit, proto_hit]}
    per_class_logit = defaultdict(lambda: [0, 0])
    per_class_proto = defaultdict(lambda: [0, 0])

    # confusion: where does each true label get misclassified?
    logit_confusion = defaultdict(lambda: defaultdict(int))
    proto_confusion = defaultdict(lambda: defaultdict(int))

    for row in test_ds.select(range(N)):
        true = row["model"]
        logit_pred, proto_pred, _ = predict(
            row["generation"], model, tokenizer, le, prototypes, device
        )

        per_class_logit[true][1] += 1
        per_class_proto[true][1] += 1

        if logit_pred == true:
            logit_correct += 1
            per_class_logit[true][0] += 1
        else:
            logit_confusion[true][logit_pred] += 1

        if proto_pred == true:
            proto_correct += 1
            per_class_proto[true][0] += 1
        else:
            proto_confusion[true][proto_pred] += 1

    print("=" * 60)
    print(f"OVERALL  (n={N})")
    print(f"  Logit accuracy:     {logit_correct / N:.4f}")
    print(f"  Prototype accuracy: {proto_correct / N:.4f}")
    print()

    print("=" * 60)
    print("PER-CLASS ACCURACY")
    print(f"  {'Label':<30} {'Logit':>8} {'Proto':>8} {'Count':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
    for label in sorted(per_class_logit.keys()):
        lhit, ltot = per_class_logit[label]
        phit, ptot = per_class_proto[label]
        print(f"  {label:<30} {lhit/ltot:>8.3f} {phit/ptot:>8.3f} {ltot:>8}")
    print()

    print("=" * 60)
    print("TOP LOGIT CONFUSIONS (true → predicted)")
    for true_label, preds in sorted(logit_confusion.items()):
        top = sorted(preds.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{p}({n})" for p, n in top)
        print(f"  {true_label:<30} → {top_str}")
    print()

    print("=" * 60)
    print("TOP PROTO CONFUSIONS (true → predicted)")
    for true_label, preds in sorted(proto_confusion.items()):
        top = sorted(preds.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{p}({n})" for p, n in top)
        print(f"  {true_label:<30} → {top_str}")