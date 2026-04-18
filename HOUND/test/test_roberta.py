import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.realpath(
    os.path.join(current_dir, "..", "fine-tuned-models", "RoBERTa-finetuned", "final_model")
)

# -----------------------
# Load model + tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

# -----------------------
# Pipeline — match training truncation settings
# -----------------------
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,
    max_length=256,
)

# -----------------------
# Inference — id2label is baked into the saved config
# -----------------------

from datasets import load_dataset
ds = load_dataset("Shengkun/Raid_split", split="test")

# Run a few examples and compare predicted vs actual

count = 0
correct = 0

for row in ds.select(range(1000)):
    result = classifier(row["generation"], truncation=True, max_length=256)[0]

    if (row['model'] == result['label']):
        correct += 1

    count +=1

    #print(f"True: {row['model']:<20} Predicted: {result['label']}")

print(correct / count)

'''
review_text = "The rain started lightly in the late afternoon, tapping against the window in a steady rhythm that made the room feel quieter than it actually was. Outside, the streetlights flickered on one by one, reflecting off wet pavement and turning the sidewalks into blurred streaks of gold and gray. Someone down the block was walking a dog that kept stopping to sniff every few steps, unwilling to rush through the weather. Inside, a kettle hummed softly in the background while a notebook sat open on the desk, half-filled with unfinished thoughts and crossed-out ideas. It was the kind of ordinary evening that didn’t seem important while it was happening, but would probably be remembered later for its calm and small, unremarkable details."

result = classifier(review_text, truncation=True, max_length=256)[0]

print("Predicted model:", result["label"])
print("Confidence:", result["score"])

# TODO: Normalize text and then retrain.
'''