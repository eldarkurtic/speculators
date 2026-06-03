"""Validate the hidden-state cache and (optionally) delete corrupt files.

The offline generator moves the server's scratch file to hs_<idx>.safetensors and
THEN validates it. If the validation (or the write/move) raced, a truncated file
can be left at the target path -- safetensors then fails to deserialize it with
"incomplete metadata, file not fully covered". Such a file:
  * makes a resuming regen SKIP that index (the file "exists"), so the gap never fills;
  * gets symlinked by _densify_cache.py and crashes training (--on-missing raise).

This script header-reads every hs_*.safetensors (cheap; no full tensor load),
checks it opens and that hidden_states.shape[0] == len(token_ids), and reports /
deletes the bad ones so they can be regenerated.

Run:
  source ablation/env.sh
  .venv/bin/python ablation/_validate_cache.py            # report only
  .venv/bin/python ablation/_validate_cache.py --delete   # delete corrupt files
  HS_DIR=/path/to/hidden_states .venv/bin/python ablation/_validate_cache.py --delete
"""

import argparse
import os
import re
import sys
from pathlib import Path

from safetensors import safe_open

HS_DIR = Path(os.environ.get("HS_DIR", os.environ["CACHE_DIR"]))


def check_one(path: Path) -> str | None:
    """Return an error string if the file is bad, else None."""
    try:
        with safe_open(path, "pt") as f:
            keys = set(f.keys())
            if "hidden_states" not in keys or "token_ids" not in keys:
                return f"missing tensors (have {sorted(keys)})"
            tok = f.get_tensor("token_ids")
            hs_shape = list(f.get_slice("hidden_states").get_shape())
            n = tok.shape[0] if tok.dim() == 1 else tok.numel()
            if hs_shape[0] != n:
                return f"shape mismatch: hidden_states[0]={hs_shape[0]} vs token_ids={n}"
    except Exception as e:  # safetensors deserialize / truncation / IO
        return f"{type(e).__name__}: {e}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true", help="delete corrupt files")
    ap.add_argument("--dir", default=str(HS_DIR), help="hidden_states dir to scan")
    args = ap.parse_args()

    d = Path(args.dir)
    files = sorted(
        (int(m.group(1)), p)
        for p in d.glob("hs_*.safetensors")
        if (m := re.match(r"hs_(\d+)\.safetensors$", p.name))
    )
    print(f"scanning {len(files)} files in {d} ...")
    bad: list[tuple[int, Path, str]] = []
    for i, (idx, p) in enumerate(files):
        # follow symlinks; validate the real target
        err = check_one(p)
        if err is not None:
            bad.append((idx, p, err))
        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1}/{len(files)} checked, {len(bad)} bad so far")

    print(f"\n{len(files) - len(bad)}/{len(files)} valid; {len(bad)} corrupt")
    for idx, p, err in bad[:30]:
        print(f"  BAD hs_{idx}: {err}")
    if len(bad) > 30:
        print(f"  ... and {len(bad) - 30} more")

    if args.delete and bad:
        for _, p, _ in bad:
            # unlink the link/file (and its real target if it's a symlink in dense set)
            target = p.resolve() if p.is_symlink() else p
            p.unlink(missing_ok=True)
            if target != p:
                target.unlink(missing_ok=True)
        print(f"\ndeleted {len(bad)} corrupt files (+ their targets). Re-run gen_cache.sh to refill.")

    # print the still-missing index set vs 0..4999 for convenience
    have = {idx for idx, _ in files} - {idx for idx, _, _ in bad}
    missing = [i for i in range(5000) if i not in have]
    print(f"\nvalid indices: {len(have)}; missing (of 5000): {len(missing)}")
    if missing:
        print(f"  missing range {missing[0]}..{missing[-1]} (showing first 20): {missing[:20]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
