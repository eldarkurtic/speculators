"""Acceptance benchmark: per-position acceptance + average accept length (tau) for a
trained DFlash draft checkpoint, on the verifier's greedy continuations of the
RedHatAI/speculator_benchmarks prompts. Run under .venv (single GPU).

Reports, per category + aggregate:
  a_1..a_7  per-position acceptance (draft argmax == verifier greedy token | prefix correct)
  tau       average accept length = sum_{k=1..B-1} prod_{i<=k} a_i

Acceptance is accumulated as global correct/total COUNTS per block position (not a mean
of per-batch ratios), so it is an unbiased per-category estimate.

Run:
  source ablation/env.sh
  .venv/bin/python ablation/eval_acceptance.py --ckpt output_dir/abl_ckpts/<run>/checkpoint_best
"""

import argparse
import os
from collections import defaultdict

import torch
import torch._inductor.config as _inductor_config

# torch 2.10 inductor pattern-matcher bug crashes the @torch.compile'd DFlash forward;
# must be set BEFORE the first forward (mirrors scripts/train.py).
_inductor_config.pattern_matcher = False

import speculators  # noqa: F401  (registers the 'dflash' speculator on import)
import speculators.models.dflash.metrics as dm
from datasets import load_from_disk
from speculators.model import SpeculatorModel
from speculators.train.data import ArrowDataset, create_collate_fn

# set by env.sh; must match what your drafts were trained with
VERIFIER = os.environ.get("VERIFIER", "/path/to/verifier")
SUPERSET = [int(x) for x in os.environ.get("SUPERSET_LAYERS", "1 9 17 25 34 36").split()]
AUX = [int(x) for x in os.environ.get("BASELINE_AUX", "1 9 17 25 34").split()]

# --- capture raw per-position correct/total counts from the model's metric path ---
# dflash.metrics.compute_metrics looks up compute_accuracy_multi_step in ITS namespace
# (module-level `from ... import`), so patch the name there.
_captured = {}
_orig = dm.compute_accuracy_multi_step


def _capture(pred_ids, target_ids, loss_mask, pos_idx, num_pos):
    correct = torch.masked_select(pred_ids == target_ids, loss_mask.to(torch.bool))
    pos = torch.masked_select(pos_idx, loss_mask.to(torch.bool))
    sums = torch.zeros(num_pos, dtype=torch.long, device=correct.device)
    counts = torch.zeros(num_pos, dtype=torch.long, device=correct.device)
    sums.scatter_add_(0, pos, correct.long())
    counts.scatter_add_(0, pos, torch.ones_like(correct, dtype=torch.long))
    _captured["sums"] = sums
    _captured["counts"] = counts
    return _orig(pred_ids, target_ids, loss_mask, pos_idx, num_pos)


dm.compute_accuracy_multi_step = _capture


def pack(indices, lengths, max_len):
    """Greedy-pack same-category row indices into batches of <= max_len total tokens."""
    batches, cur, cur_len = [], [], 0
    for i in indices:
        if cur and cur_len + lengths[i] > max_len:
            batches.append(cur)
            cur, cur_len = [], 0
        cur.append(i)
        cur_len += lengths[i]
    if cur:
        batches.append(cur)
    return batches


def acc_tau(sum_correct, sum_total, block_size):
    a = [(sum_correct[k] / sum_total[k]) if sum_total[k] > 0 else 0.0
         for k in range(block_size)]
    tau, cum = 0.0, 1.0
    for k in range(1, block_size):
        cum *= a[k]
        tau += cum
    return a, tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="draft checkpoint dir (has config.json)")
    ap.add_argument("--bench", default="output_dir/Qwen3-8B_bench_dense")
    ap.add_argument("--total-seq-len", type=int, default=8192)
    args = ap.parse_args()

    torch.manual_seed(0)  # determinism for select_anchors' randperm (full coverage anyway)
    model = SpeculatorModel.from_pretrained(args.ckpt).to("cuda").to(torch.bfloat16).eval()
    # from_pretrained doesn't restore training-only loss attrs; the forward computes a
    # loss (which we ignore — acceptance uses argmax), so set a valid loss_type.
    model.loss_type = "ce"
    model.loss_gamma = 4.0
    model.label_smoothing = 0.0
    block_size = model.block_size
    hidden = model.config.transformer_layer_config.hidden_size

    ds = ArrowDataset(
        max_len=args.total_seq_len, datapath=args.bench,
        hidden_states_path=os.path.join(args.bench, "hidden_states"),
        on_missing="raise", split_ratio=1.0, transform=None,
        hidden_states_dtype=torch.bfloat16, model=VERIFIER,
        cache_layer_ids=SUPERSET, aux_layer_ids=AUX,
    )
    collate = create_collate_fn(args.total_seq_len, hidden, preprocess=None)
    raw = load_from_disk(args.bench)
    cats = raw["category"]
    lengths = list(raw["seq_len"])

    by_cat = defaultdict(list)
    for i in range(len(ds)):
        by_cat[cats[i]].append(i)

    # category -> [sum_correct(block_size), sum_total(block_size)]
    acc = {c: [torch.zeros(block_size, dtype=torch.long),
               torch.zeros(block_size, dtype=torch.long)] for c in by_cat}
    with torch.no_grad():
        for cat, idxs in by_cat.items():
            for batch_idx in pack(idxs, lengths, args.total_seq_len):
                batch = collate([ds[i] for i in batch_idx])
                batch = {k: (v.to("cuda", non_blocking=True) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                model(**batch)  # triggers the captured metric path
                acc[cat][0] += _captured["sums"].cpu()
                acc[cat][1] += _captured["counts"].cpu()

    # report
    hdr = "category".ljust(16) + "  " + "  ".join(f"a{k}" for k in range(1, block_size)) + "    tau"
    print(hdr)
    print("-" * len(hdr))
    tot_c = torch.zeros(block_size, dtype=torch.long)
    tot_t = torch.zeros(block_size, dtype=torch.long)
    rows = []
    for cat in sorted(acc):
        sc, st = acc[cat]
        tot_c += sc
        tot_t += st
        a, tau = acc_tau(sc.tolist(), st.tolist(), block_size)
        rows.append((cat, a, tau))
    for cat, a, tau in sorted(rows, key=lambda r: -r[2]):
        cells = "  ".join(f"{a[k]:.3f}" for k in range(1, block_size))
        print(f"{cat:16s}  {cells}   {tau:.3f}")
    a, tau = acc_tau(tot_c.tolist(), tot_t.tolist(), block_size)
    cells = "  ".join(f"{a[k]:.3f}" for k in range(1, block_size))
    print("-" * len(hdr))
    print(f"{'AGGREGATE':16s}  {cells}   {tau:.3f}")
    print(f"\ncheckpoint: {args.ckpt}\nbenchmark: {args.bench}  ({len(ds)} prompts)")


if __name__ == "__main__":
    main()
