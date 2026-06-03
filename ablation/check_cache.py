"""Sanity-check the hidden-state cache: count files and inspect one sample's shape/dtype.

Run with the training venv:  .venv/bin/python ablation/check_cache.py
"""

import os
import sys
from pathlib import Path

from safetensors import safe_open

_env_cache_dir = os.environ.get("CACHE_DIR", "")
CACHE_DIR = (
    Path(_env_cache_dir)
    if _env_cache_dir
    else Path(__file__).resolve().parents[1]
    / "output_dir/Qwen3-8B_magpie_5k/hidden_states"
)


def main() -> int:
    files = sorted(CACHE_DIR.glob("hs_*.safetensors"))
    print(f"cache dir: {CACHE_DIR}")
    print(f"num cache files: {len(files)}")
    if not files:
        print("!! no cache files found")
        return 1
    with safe_open(files[0], framework="pt") as f:
        keys = list(f.keys())
        hs = f.get_slice("hidden_states")
        shape = list(hs.get_shape())
        print(f"sample: {files[0].name}  keys={keys}  hidden_states shape={shape}")
        # expect [seq_len, num_superset_layers(=6: aux 1 9 17 25 34 + target 36),
        #         hidden_size(=4096)]
        if len(shape) != 3:
            print("!! unexpected rank (want [seq, n_layers, hidden])")
            return 1
        print(f"  -> seq_len={shape[0]}, n_layers={shape[1]}, hidden={shape[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
