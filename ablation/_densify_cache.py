"""Build a DENSE, contiguous dataset from the sparse cache.

The offline generator skipped ~28% of samples ("incomplete metadata"), leaving
hs_*.safetensors with gaps over indices 0..4999. Training with `--on-missing skip`
then crashes when the multipack sampler forms a fully-empty batch (anchors land on
padding). Fix: keep only the rows that HAVE hidden states, re-index them
contiguously, and symlink the hs files so the loader can use `--on-missing raise`.

Symlinks (not copies) keep this instant and avoid duplicating ~658GB.

Run:  source ablation/env.sh && .venv/bin/python ablation/_densify_cache.py
"""

import os
import re
import shutil
from pathlib import Path

from datasets import load_from_disk

SRC = Path(os.environ["DATASET"])  # original dataset dir
SRC_HS = Path(os.environ["CACHE_DIR"])  # original hidden_states dir
DST = SRC.parent / (SRC.name + "_dense")
DST_HS = DST / "hidden_states"

# 1. original indices that actually have a cache file
valid = sorted(
    int(m.group(1))
    for p in SRC_HS.glob("hs_*.safetensors")
    if (m := re.match(r"hs_(\d+)\.safetensors$", p.name))
)
print(f"valid cache samples: {len(valid)}")

# 2. filtered arrow dataset (rows aligned to the kept hidden states, in order)
data = load_from_disk(str(SRC))
print(f"original rows: {len(data)}")
dense = data.select(valid)
dense.save_to_disk(str(DST))  # creates DST
print(f"dense rows: {len(dense)}  ->  {DST}")

# 3. contiguous symlinks  new hs_{j} -> original hs_{valid[j]}
DST_HS.mkdir(parents=True, exist_ok=True)
for j, orig in enumerate(valid):
    link = DST_HS / f"hs_{j}.safetensors"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(SRC_HS / f"hs_{orig}.safetensors")
print(f"symlinked {len(valid)} hs files into {DST_HS}")

# 4. vocab-mapping files
for fn in ("d2t.npy", "t2d.npy", "token_freq.pt"):
    src = SRC / fn
    if src.exists():
        shutil.copy2(src, DST / fn)
        print(f"copied {fn}")

print("DONE")
