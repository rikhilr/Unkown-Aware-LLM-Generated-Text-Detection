from detect import load_detector, predict

# 1. Path to the 'final_model' folder created by your training script
MODEL_PATH = "./results/my_experiment/final_model"

# 2. Load everything (Model, Tokenizer, Prototypes, Encoder)
model, tokenizer, le, prototypes = load_detector(MODEL_PATH)

# 3. Run a test
test_text = "The quick brown fox jumps over the lazy dog."

# We are using a threshold of 15.0 here instead of 95.0.
# DeBERTa's energy scores usually hover between 10-20.
result = predict(test_text, model, tokenizer, le, prototypes, threshold=15.0)

print(f"Prediction: {result['result']}")
print(f"Energy Score: {result['energy']:.2f} (Higher = More likely Human/Unseen)")
print(f"Similarity: {result['sim']:.2f} (Higher = More likely AI)")
