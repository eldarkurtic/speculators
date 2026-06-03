"""Step 2 of the acceptance benchmark: turn the verifier continuations (from
_gen_benchmark.py) into a training-format HF dataset the HS-extraction + eval pipeline
consumes. Run under your training venv (.venv).

Each row: input_ids = [chat-templated prompt] + [verifier greedy continuation],
loss_mask = 1 ONLY on the continuation (acceptance is measured there), seq_len = len.
A `category` column is kept for the per-domain breakdown.

Reads BENCH (output dir) and VOCAB_SRC (dir with d2t.npy/t2d.npy/token_freq.pt that
your drafts were trained with) from the env (set by env.sh).

Run:  source env.sh && .venv/bin/python _build_benchmark_arrow.py
"""

import json
import os
import shutil
from pathlib import Path

from datasets import Dataset, Sequence, Value

BENCH = Path(os.environ["BENCH"])
JSONL = BENCH / "continuations.jsonl"
VOCAB_SRC = Path(os.environ["VOCAB_SRC"])  # must hold the SAME d2t/t2d your drafts used
SEQ_LEN = int(os.environ.get("TOTAL_SEQ_LEN", "8192"))  # rows longer than this are skipped


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
            skipped += 1
            continue
        rows.append({
            "input_ids": input_ids,
            "loss_mask": [False] * len(pids) + [True] * len(cont),
            "seq_len": len(input_ids),
            "category": o["category"],
        })
    print(f"kept {len(rows)} rows; skipped {skipped} (>{SEQ_LEN} tok); {empty} empty")

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
            print(f"copied {fn}")
        else:
            print(f"WARN: {src} missing — eval needs the SAME vocab map your drafts used")
    from collections import Counter
    print("per-category rows:", dict(Counter(r["category"] for r in rows)))
    print(f"DONE -> {BENCH}")


if __name__ == "__main__":
    main()
