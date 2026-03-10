from datasets import load_dataset

print("Downloading MAGE dataset...")
dataset = load_dataset("yaful/MAGE")
print("Saving to disk...")
dataset.save_to_disk("./mage_dataset")
print("Done! Dataset saved to ./mage_dataset")
