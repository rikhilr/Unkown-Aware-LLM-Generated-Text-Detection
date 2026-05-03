#!/bin/bash
set -euo pipefail
python -m pip install --upgrade pip
python -m pip install "datasets>=2.19" pandas
python - <<'PY'
from datasets import load_dataset
from collections import Counter

for split in ["train", "extra"]:
    print("\n===== SPLIT", split, "=====")
    ds = load_dataset("liamdugan/raid", split=split, streaming=True)
    rows = list(ds.take(200))
    print("First row keys:")
    print(list(rows[0].keys()))
    print("First row preview:")
    print({k: (str(v)[:500] if k == 'generation' else v) for k, v in rows[0].items()})
    for key in ["model", "domain", "attack", "decoding", "repetition_penalty"]:
        vals = [str(r.get(key)) for r in rows]
        print("\n", key)
        print(Counter(vals).most_common(30))
PY
