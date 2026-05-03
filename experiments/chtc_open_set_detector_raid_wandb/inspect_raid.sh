#!/bin/bash
set -euo pipefail
export HOME="${PWD}"
export HF_HOME="${PWD}/hf_cache"

python -m pip install --user --upgrade pip
python -m pip install --user datasets pandas

python - <<'PY'
from datasets import load_dataset
from collections import Counter

ds = load_dataset('liamdugan/raid', split='train', streaming=True)
ds = ds.shuffle(buffer_size=1000, seed=42)
rows = list(ds.take(50))
print('First row keys:')
print(list(rows[0].keys()))
print('\nFirst row preview:')
print({k: str(v)[:500] for k, v in rows[0].items()})
print('\nValue counts for scalar-looking columns:')
for key in rows[0].keys():
    vals = []
    for row in rows:
        val = row.get(key)
        if isinstance(val, (str, int, float, bool)) or val is None:
            s = str(val)
            if len(s) < 120:
                vals.append(s)
    if vals:
        print('\n' + key)
        print(Counter(vals).most_common(20))
PY
