import joblib
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder

from collections import Counter

def main():
    print("Loading dataset...")
    dataset = load_dataset("Shengkun/Raid_split", split="train")

    
    print(Counter(dataset["model"]))

    '''
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
    '''
    

if __name__ == "__main__":
    main()