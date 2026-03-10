#make sure you ran test.py to create the dataset before running this EDA script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# ------ Load the dataset ------

from datasets import load_from_disk
print("Loading dataset...")
dataset = load_from_disk("./mage_dataset")

train_df = pd.DataFrame(dataset['train'])
val_df = pd.DataFrame(dataset['validation'])
test_df = pd.DataFrame(dataset['test'])

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"Columns: {train_df.columns.tolist()}")
print(f"\nUnique sources: {train_df['src'].unique()}")

# Create output directory for plots
os.makedirs("deepfake_eda_plots", exist_ok=True)

# ------ Parse 'src' column into 'domain' and 'generator' ------
def parse_src(src) :
    """Parse 'src' column into domain and generator."""
    if src.endswith("_human"):
        return src[:-6], "human"
    elif "_machine_" in src:
        domain, generator = src.split("_machine_", 1)
        return domain, generator
    else:
        return src, "unknown"
    
for df in [train_df, val_df, test_df]:
    df[['domain', 'generator']] = pd.DataFrame(
        df['src'].map(parse_src).tolist(),
        index=df.index
    )

print("\n--- Domains ---")
print(train_df['domain'].value_counts())
print("\n--- Generators ---")
print(train_df['generator'].value_counts())
print("\n--- Labels (0=machine, 1=human) ---")
print(train_df['label'].value_counts())

# -------  Text Statistics -------
print("\n========== PERFORMING TEXT STATISTICS ==========")

for (df, name) in zip([train_df, val_df, test_df], ["Train", "Validation", "Test"]):
    print(f"\n--- {name} Set ---")
    df['text_length'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['sentence_count'] = df['text'].apply(lambda x: x.count('.') + x.count('!') + x.count('?'))
    df['avg_word_len'] = df['text'].apply(lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0)
    df['vocab_size'] = df['text'].apply(lambda x: len(set(x.lower().split())))
    df['type_token_ratio'] = df['vocab_size'] / df['word_count'].clip(lower=1)

    print(f"Average text length: {df['text_length'].mean():.2f}")
    print(f"Median text length: {df['text_length'].median()}")
    print(f"Text length std: {df['text_length'].std():.2f}")
    print(f"Average word count: {df['word_count'].mean():.2f}")
    print(f"Average type-token ratio: {df['type_token_ratio'].mean():.4f}")
    
    # Plot text length distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(df['text_length'], bins=30, kde=True)
    plt.title(f"{name} Set - Text Length Distribution")
    plt.xlabel("Text Length (characters)")
    plt.ylabel("Frequency")
    plt.savefig(f"deepfake_eda_plots/{name.lower()}_text_length_distribution.png")
    plt.close()

# -------  Distributions -------   
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (df, name) in zip(axes, [(train_df, "Train"), (val_df, "Validation"), (test_df, "Test")]):
    counts = df['label'].value_counts()
    labels = ['Machine (0)', 'Human (1)']
    ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=['red', 'purple'])
    ax.set_title(f"{name} Set — Label Distribution")
    ax.set_ylabel("Count")
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        ax.text(i, v + 500, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("deepfake_eda_plots/01_label_distribution.png", dpi=150)
plt.show()

# ------- Per Domain and Generator Analysis -------
print("\n========== PERFORMING PER DOMAIN AND GENERATOR ANALYSIS ==========")
fig, ax = plt.subplots(figsize=(14, 6))
gen_counts = train_df['generator'].value_counts()
gen_counts.plot(kind='barh', ax=ax, color='#2ecc71')
ax.set_title("Training Set — Samples per Generator")
ax.set_xlabel("Count")
plt.tight_layout()
plt.savefig("deepfake_eda_plots/02_generator_distribution.png", dpi=150)
plt.show()

#  Per-domain distribution
fig, ax = plt.subplots(figsize=(12, 6))
domain_counts = train_df['domain'].value_counts()
domain_counts.plot(kind='barh', ax=ax, color='#9b59b6')
ax.set_title("Training Set — Samples per Domain")
ax.set_xlabel("Count")
plt.tight_layout()
plt.savefig("deepfake_eda_plots/03_domain_distribution.png", dpi=150)
plt.show()

# Generator x Domain heatmap
cross = pd.crosstab(train_df['generator'], train_df['domain'])
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(cross, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
ax.set_title("Generator × Domain Distribution (Train)")
plt.tight_layout()
plt.savefig("deepfake_eda_plots/04_generator_domain_heatmap.png", dpi=150)
plt.show()

# ------- Text Length by Generator -------
fig, ax = plt.subplots(figsize=(12, 6))
train_df[train_df['label'] == 1]['word_count'].hist(bins=100, alpha=0.6, label='Human', color='#3498db', ax=ax)
train_df[train_df['label'] == 0]['word_count'].hist(bins=100, alpha=0.6, label='Machine', color='#e74c3c', ax=ax)
ax.set_xlim(0, 1000)
ax.set_title("Word Count Distribution: Human vs Machine")
ax.set_xlabel("Word Count")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig("deepfake_eda_plots/05_word_count_distribution.png", dpi=150)
plt.show()

# Word count by generator boxplot
fig, ax = plt.subplots(figsize=(14, 7))
order = train_df.groupby('generator')['word_count'].median().sort_values().index
sns.boxplot(data=train_df, x='generator', y='word_count', order=order, ax=ax)
ax.set_ylim(0, 1000)
plt.xticks(rotation=45, ha='right')
ax.set_title("Word Count Distribution by Generator")
plt.tight_layout()
plt.savefig("deepfake_eda_plots/06_word_count_by_generator.png", dpi=150)
plt.show()

# Vocab diversity
fig, ax = plt.subplots(figsize=(14, 7))
sns.boxplot(data=train_df, x='generator', y='type_token_ratio', order=order, ax=ax)
ax.set_ylim(0, 1)
plt.xticks(rotation=45, ha='right')
ax.set_title("Type-Token Ratio by Generator (Vocabulary Diversity)")
ax.set_ylabel("Type-Token Ratio (higher = more diverse)")
plt.tight_layout()
plt.savefig("deepfake_eda_plots/07_vocab_diversity_by_generator.png", dpi=150)
plt.show()

# Train vs test analysis
print("\n========== CROSS-DATASET COMPARISON ==========")
print("\nTrain generators:")
print(sorted(train_df['generator'].unique()))
print("\nTest generators:")
print(sorted(test_df['generator'].unique()))

train_gens = set(train_df['generator'].unique())
test_gens = set(test_df['generator'].unique())
print(f"\nGenerators in test but NOT in train: {test_gens - train_gens}")
print(f"Generators in train but NOT in test: {train_gens - test_gens}")
print(f"Shared generators: {train_gens & test_gens}")


# 8. EMBEDDING ANALYSIS

print("\n========== EMBEDDING ANALYSIS ==========")
print("This section requires transformers + torch. May take a while.\n")

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import cdist, cosine

    model_name = "roberta-base"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    roberta_model = AutoModel.from_pretrained(model_name)
    roberta_model.eval()

    device = torch.device("mps" if torch.backends.mps.is_available() 
                          else "cuda" if torch.cuda.is_available() else "cpu")
    roberta_model = roberta_model.to(device)
    print(f"Using device: {device}")

    def get_embeddings(texts, batch_size=16):
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                              max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = roberta_model(**inputs)
            embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeds.append(embeds)
            if (i // batch_size) % 10 == 0:
                print(f"  Processed {i+len(batch)}/{len(texts)} samples...")
        return np.vstack(all_embeds)

    # Sample per generator for embedding analysis
    N_PER_CLASS = 200
    sampled = train_df.groupby('generator').apply(
        lambda x: x.sample(min(N_PER_CLASS, len(x)), random_state=42)
    ).reset_index(drop=True)
    print(f"\nSampled {len(sampled)} texts for embedding analysis")
    print(sampled['generator'].value_counts())

    print("\nComputing embeddings...")
    embeddings = get_embeddings(sampled['text'].tolist())
    print(f"Embedding shape: {embeddings.shape}")

    # 8a. t-SNE visualization colored by generator
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(14, 10))
    generators = sampled['generator'].values
    unique_gens = sorted(set(generators))
    palette = sns.color_palette("husl", len(unique_gens))
    for i, gen in enumerate(unique_gens):
        mask = generators == gen
        marker = 'x' if gen == 'human' else 'o'
        size = 60 if gen == 'human' else 30
        ax.scatter(coords[mask, 0], coords[mask, 1],
                  label=gen, alpha=0.6, s=size, marker=marker, color=palette[i])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.set_title("t-SNE of Text Embeddings by Generator (RoBERTa)")
    plt.tight_layout()
    plt.savefig("deepfake_eda_plots/08_tsne_by_generator.png", dpi=150, bbox_inches='tight')
    plt.show()

    # 8b. t-SNE colored by human vs machine
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = sampled['label'].values
    colors = ['#e74c3c' if l == 0 else '#3498db' for l in labels]
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.5, s=20)
    ax.scatter([], [], c='#3498db', label='Human', s=40)
    ax.scatter([], [], c='#e74c3c', label='Machine', s=40)
    ax.legend(fontsize=12)
    ax.set_title("t-SNE: Human vs Machine-Generated Text")
    plt.tight_layout()
    plt.savefig("deepfake_eda_plots/09_tsne_human_vs_machine.png", dpi=150)
    plt.show()

    # 8c. Silhouette score
    sil_score = silhouette_score(embeddings, sampled['label'].values, sample_size=2000)
    print(f"\nSilhouette Score (Human vs Machine): {sil_score:.4f}")

    sil_gen = silhouette_score(embeddings, sampled['generator'].values, sample_size=2000)
    print(f"Silhouette Score (Per Generator): {sil_gen:.4f}")

    #  Per-generator centroid distances (cosine)
    print("\n--- Inter-Generator Centroid Distances (Cosine) ---")
    centroids = {}
    for gen in unique_gens:
        mask = generators == gen
        centroids[gen] = embeddings[mask].mean(axis=0)

    centroid_matrix = np.array([centroids[g] for g in unique_gens])
    dist_matrix = cdist(centroid_matrix, centroid_matrix, metric='cosine')
    dist_df = pd.DataFrame(dist_matrix, index=unique_gens, columns=unique_gens)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(dist_df, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax)
    ax.set_title("Cosine Distance Between Generator Centroids (RoBERTa Embeddings)")
    plt.tight_layout()
    plt.savefig("deepfake_eda_plots/10_centroid_distances.png", dpi=150)
    plt.show()

    #  Intra-class variance (compactness)
    print("\n--- Intra-Class Variance (lower = more compact) ---")
    for gen in unique_gens:
        mask = generators == gen
        gen_embeds = embeddings[mask]
        centroid = gen_embeds.mean(axis=0)
        variance = np.mean(np.sum((gen_embeds - centroid) ** 2, axis=1))
        print(f"  {gen:30s}: {variance:.4f}")

    #  Distance of each generator to human centroid (cosine, consistent with 8d)
    print("\n--- Cosine Distance to Human Centroid (key for unknown-aware detection!) ---")
    human_centroid = centroids['human']
    distances_to_human = []
    for gen in unique_gens:
        if gen == 'human':
            continue
        dist = cosine(centroids[gen], human_centroid)
        distances_to_human.append((gen, dist))
        print(f"  {gen:30s}: {dist:.4f}")

    distances_to_human.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(12, 6))
    gens = [d[0] for d in distances_to_human]
    dists = [d[1] for d in distances_to_human]
    ax.barh(gens, dists, color='#e67e22')
    ax.set_title("Cosine Distance of Each Generator's Centroid to Human Centroid")
    ax.set_xlabel("Cosine Distance in Embedding Space")
    plt.tight_layout()
    plt.savefig("deepfake_eda_plots/11_distance_to_human.png", dpi=150)
    plt.show()

    print("\n✓ Embedding analysis complete!")

except ImportError as e:
    print(f"Skipping embedding analysis — missing dependency: {e}")
    print("Install with: pip install torch transformers scikit-learn scipy")

# ============================================================
# 9. SUMMARY STATISTICS TABLE
# ============================================================
print("\n========== SUMMARY TABLE ==========")
summary = train_df.groupby('generator').agg(
    count=('text', 'size'),
    avg_words=('word_count', 'mean'),
    std_words=('word_count', 'std'),
    avg_ttr=('type_token_ratio', 'mean'),
    avg_word_len=('avg_word_len', 'mean')
).round(2)
print(summary.to_string())
summary.to_csv("deepfake_eda_plots/summary_stats.csv")

print("\n========================================")
print("EDA complete! All plots saved to ./deepfake_eda_plots/")
print("========================================")
