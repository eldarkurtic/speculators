"""Step 1 of the acceptance benchmark: generate the verifier's GREEDY continuations
for the RedHatAI/speculator_benchmarks prompts. Run under .venv_vllm (needs vllm).

Acceptance must be measured against what *this verifier* (Qwen3-8B) would greedily
generate, not the gold references — so we generate continuations ourselves and write
them to a JSONL that _build_benchmark_arrow.py turns into the training-format dataset.

Run:  .venv_vllm/bin/python ablation/_gen_benchmark.py [--thinking] [--k 256]
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

VERIFIER = "/home/eldarkurtic/hf_models/Qwen/Qwen3-8B"
CATS = ["HumanEval", "math_reasoning", "qa", "question", "rag",
        "summarization", "tool_call", "translation", "writing"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=256, help="max continuation tokens")
    ap.add_argument("--thinking", action="store_true",
                    help="enable Qwen3 reasoning (default OFF — else 256 tokens are eaten by <think>)")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--out", default="output_dir/Qwen3-8B_bench/continuations.jsonl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(VERIFIER, trust_remote_code=True)
    # collect (category=file stem, raw prompt string)
    rows = []
    for cat in CATS:
        path = hf_hub_download("RedHatAI/speculator_benchmarks", f"{cat}.jsonl",
                               repo_type="dataset")
        for line in open(path):
            obj = json.loads(line)
            rows.append((cat, obj["prompt"]))
    print(f"loaded {len(rows)} prompts across {len(CATS)} categories")

    # chat-template each prompt -> flat prompt token ids (template adds special tokens)
    prompts, prompt_ids_all = [], []
    for _, prompt in rows:
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, enable_thinking=args.thinking, tokenize=False,
        )
        ids = tok.encode(text, add_special_tokens=False)
        prompt_ids_all.append(ids)
        prompts.append(TokensPrompt(prompt_token_ids=ids))
    longest = max(len(x) for x in prompt_ids_all)
    print(f"longest prompt = {longest} tokens (+{args.k} continuation; "
          f"max_model_len={args.max_model_len}); thinking={args.thinking}")

    llm = LLM(model=VERIFIER, dtype="bfloat16", max_model_len=args.max_model_len)
    sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=args.k)
    outs = llm.generate(prompts, sp)  # one batched call; order preserved

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_stop = 0
    with open(out_path, "w") as f:
        for (cat, _), pids, o in zip(rows, prompt_ids_all, outs):
            cont = list(o.outputs[0].token_ids)
            fr = o.outputs[0].finish_reason
            n_stop += fr == "stop"
            f.write(json.dumps({"category": cat, "prompt_ids": pids,
                                "continuation_ids": cont, "finish_reason": fr}) + "\n")
    print(f"wrote {len(rows)} rows -> {out_path}  ({n_stop} hit EOS, "
          f"{len(rows) - n_stop} hit length cap)")


if __name__ == "__main__":
    main()
