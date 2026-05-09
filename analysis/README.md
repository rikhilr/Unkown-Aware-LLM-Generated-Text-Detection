# analysis/

Scripts for post-hoc analysis and paper figures.

## plot_umap.py

Generates a 2-D UMAP projection of StyleDistance embeddings sampled from the
RAID dataset, used as a motivation figure in the ACL paper.

**Three classes are visualised:**
| Class | Colour | Generators |
|-------|--------|------------|
| Human | gray | RAID human texts |
| Known LLM | blue | chatgpt, gpt3, gpt4, cohere, llama-chat, mpt, mpt-chat |
| Unknown LLM | red | all other generators (gpt2, cohere-chat, mistral, mistral-chat, …) |

The embedding model, tokenisation, and layer-pooling are identical to those
used during training (`StyleDistance/styledistance`, layer `-2`, masked mean
pooling, max-length 512).

### Dependencies

```
pip install umap-learn matplotlib datasets transformers torch tqdm
```

### Usage

```bash
# From the project root:
python analysis/plot_umap.py

# With options:
python analysis/plot_umap.py \
    --n_per_class 800 \
    --cache_dir analysis/cache \
    --output latex/figures/umap_embeddings.pdf
```

On first run the script streams ~2 400 rows from `liamdugan/raid` on
HuggingFace, computes embeddings, and caches them to
`analysis/cache/embeddings_n800_seed42.npz`.  Subsequent runs load the cache
directly and only re-run UMAP + plotting (much faster).

Pass `--force_recompute` to ignore the cache and recompute embeddings from
scratch.

### Output

`latex/figures/umap_embeddings.pdf` — a vector PDF suitable for inclusion in
a LaTeX document.
