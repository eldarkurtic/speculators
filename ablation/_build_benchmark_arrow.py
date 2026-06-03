"""Turn verifier continuations (from _gen_benchmark.py / _gen_mrcr.py) into a
training-format HF dataset for HS extraction + acceptance eval. Run under .venv.

Each row: input_ids = [prompt] + [verifier greedy continuation], loss_mask = 1 only on
the continuation, seq_len = len, + a `category` column (the bucket/task label).

Reads BENCH (output dir), VOCAB_SRC (dir with d2t/t2d your drafts used) and TOTAL_SEQ_LEN
from the env (set by build_*_cache.sh); falls back to the speculator-benchmarks defaults.
"""

import json
import os
import shutil
from collections import Counter
from pathlib import Path

from datasets import Dataset, Sequence, Value

REPO = Path("/home/eldarkurtic/github/eldarkurtic/speculators")
BENCH = Path(os.environ.get("BENCH", str(REPO / "output_dir/Qwen3-8B_bench")))
VOCAB_SRC = Path(os.environ.get("VOCAB_SRC", str(REPO / "output_dir/Qwen3-8B_magpie_5k_dense")))
SEQ_LEN = int(os.environ.get("TOTAL_SEQ_LEN", "8192"))
JSONL = BENCH / "continuations.jsonl"


def main():
    rows, skipped, empty = [], 0, 0
    for line in open(JSONL):
        o = json.loads(line)
        pids, cont = o["prompt_ids"], o["continuation_ids"]
        if not cont:
            empty += 1
            continue
        input_ids = list(pids) + list(cont)
        if len(input_ids) > SEQ_LEN:
            skipped += 1  # skip rather than truncate (would clip the measured continuation)
            continue
        rows.append({
            "input_ids": input_ids,
            "loss_mask": [False] * len(pids) + [True] * len(cont),
            "seq_len": len(input_ids),
            "category": o.get("category", "bench"),
        })
    print(f"kept {len(rows)} rows; skipped {skipped} (>{SEQ_LEN} tok); {empty} empty")
    if not rows:
        raise SystemExit("no rows fit TOTAL_SEQ_LEN — raise it or lower the continuation --k")

    ds = Dataset.from_list(rows)
    ds = ds.cast_column("input_ids", Sequence(Value("int32")))
    ds = ds.cast_column("loss_mask", Sequence(Value("bool")))
    ds.set_format(type="torch", columns=["input_ids", "loss_mask", "seq_len"],
                  output_all_columns=True)
    ds.save_to_disk(str(BENCH))

    for fn in ("d2t.npy", "t2d.npy", "token_freq.pt"):
        src = VOCAB_SRC / fn
        if src.exists():
            shutil.copy2(src, BENCH / fn)
        else:
            print(f"WARN: {src} missing — eval needs the SAME vocab map your drafts used")
    print("per-category rows:", dict(Counter(r["category"] for r in rows)))
    print(f"DONE -> {BENCH}")


if __name__ == "__main__":
    main()
