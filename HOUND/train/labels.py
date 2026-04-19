import joblib
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder

from collections import Counter

FAMILY_MAP = {
    "chatgpt": "GPT",
    "gpt-4": "GPT",
    "text-davinci-003": "GPT",
    "gpt-2": "GPT",
    "llama-2-70b": "Llama",
    "mistral-7b": "Mistral",
    "mistral-7b-chat": "Mistral",
    "mpt-30b": "MPT",
    "mpt-30b-chat": "MPT",
    "cohere": "Cohere",
    "cohere-chat": "Cohere",
}


def get_family_le(models):
    le_fam = LabelEncoder()
    families = [FAMILY_MAP.get(m, "Unknown") for m in models]
    le_fam.fit(families)
    return le_fam


def main():
    print("Loading dataset...")
    dataset = load_dataset("Shengkun/Raid_split", split="train")

    print(Counter(dataset["model"]))

    """
    print("Fitting LabelEncoder...")
    le = LabelEncoder()
    le.fit(dataset["model"])

    # Show mapping clearly
    print("\nLabel mapping:")
    for idx, label in enumerate(le.classes_):
        print(f"{idx} -> {label}")

    # Save encoder
    output_path = "label_encoder.pkl"
    joblib.dump(le, output_path)

    print(f"\nSaved LabelEncoder to {output_path}")
    """


if __name__ == "__main__":
    main()
