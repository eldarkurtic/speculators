"""Generate verifier GREEDY continuations for one OpenAI-MRCR length bucket.
MRCR `prompt` is already a chat message-list (a long multi-turn conversation whose
final user turn asks to reproduce a specific earlier message). Run under .venv_vllm.

Run (via build_mrcr_cache.sh):
  .venv_vllm/bin/python _gen_mrcr.py --file <bucket.jsonl> --out <dir>/continuations.jsonl \
      --k 512 --max-model-len <N> [--max-samples M] [--bucket 4k-8k]
"""

import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

VERIFIER = os.environ["VERIFIER"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=512, help="continuation tokens (MRCR needle ~400-700)")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all")
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--bucket", default="mrcr", help="category label written per row")
    ap.add_argument("--thinking", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(VERIFIER, trust_remote_code=True)
    rows = []
    for i, line in enumerate(open(args.file)):
        if args.max_samples and i >= args.max_samples:
            break
        rows.append(json.loads(line))

    prompts, pids = [], []
    too_long = 0
    for r in rows:
        text = tok.apply_chat_template(r["prompt"], add_generation_prompt=True,
                                       enable_thinking=args.thinking, tokenize=False)
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) + args.k > args.max_model_len:  # can't prefill+generate
            too_long += 1
            continue
        pids.append(ids)
        prompts.append(TokensPrompt(prompt_token_ids=ids))
    lens = [len(x) for x in pids]
    print(f"{len(prompts)}/{len(rows)} prompts fit (max_model_len={args.max_model_len}; "
          f"{too_long} too long). prompt tokens: min={min(lens)} max={max(lens)}")

    llm = LLM(model=VERIFIER, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=0.9)
    sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=args.k)
    outs = llm.generate(prompts, sp)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fo:
        for p, o in zip(pids, outs):
            fo.write(json.dumps({"category": args.bucket, "prompt_ids": p,
                                 "continuation_ids": list(o.outputs[0].token_ids),
                                 "finish_reason": o.outputs[0].finish_reason}) + "\n")
    print(f"wrote {len(prompts)} -> {args.out}")


if __name__ == "__main__":
    main()
