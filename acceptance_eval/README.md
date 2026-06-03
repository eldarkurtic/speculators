# acceptance_eval — spec-decode acceptance metrics for DFlash drafts

Collects, for any DFlash draft checkpoint, the **per-position acceptance rates `a_1..a_7`** and the
**average accept length `τ` (= Σ_k Π_{i≤k} a_i)**, per benchmark domain + aggregate. Acceptance is
measured (teacher-forced, greedy) against **the verifier's own greedy continuation** of the
[RedHatAI/speculator_benchmarks](https://huggingface.co/datasets/RedHatAI/speculator_benchmarks)
prompts — i.e. how often the draft predicts the token the verifier would actually accept.

This is the standard EAGLE-style proxy (not a live sampled spec-decode loop), and it's far more
faithful to deployment than the in-distribution training-val EAL (which overstates real acceptance).

## What's in here
| file | role |
|------|------|
| **`eval_acceptance.py`** | **the metric collector** — run it per checkpoint |
| `env.sh` | the single config point — **edit the 4 marked lines** |
| `build_benchmark_cache.sh` | one-shot cache builder (gen → arrow → HS-extract → densify) |
| `_gen_benchmark.py` | verifier greedy continuations (vLLM, `.venv_vllm`) |
| `_build_benchmark_arrow.py` | continuations → training-format arrow dataset |
| `gen_cache.sh` | hidden-state extraction (reused by the build) |
| `_densify_cache.py`, `_validate_cache.py` | densify + validate the HS cache |

## Prerequisites
- A `speculators` repo (your `REPO`) with the two venvs it uses: **`.venv`** (speculators + torch, no
  vllm) and **`.venv_vllm`** (vllm) — same split as the training setup.
- Your draft checkpoints load via `SpeculatorModel.from_pretrained` (a registered `dflash` speculator
  with a `config.json`).
- **One REQUIRED library fix** (your library should already have it if it matches ours; verify):
  in `REPO/src/speculators/train/data.py`, `ArrowDataset.__init__` must set `start_file_idx` for the
  default `split_ratio=1.0` (the eval reads the whole benchmark). If yours still has `pass`, change it:
  ```python
          self.data = load_from_disk(datapath)
          if split_ratio == 1.0:
              self.start_file_idx = 0      # <-- was `pass`; REQUIRED, eval uses split_ratio=1.0
          elif 1.0 > split_ratio > 0:
  ```
- **For building a cache (recommended):** `REPO/scripts/data_generation_offline.py` should retry the
  output validation (the server's async HS write can lag its response, dropping ~good samples). If your
  copy lacks a `_check_safetensors_with_retry` around `check_safetensors_file`, wrap it in a 5–12×
  `time.sleep(1)` retry, and make sure `gen_cache.sh` passes `--request-timeout 600` (already does here).

## Setup
Edit the four `EDIT` lines in **`env.sh`**:
- `REPO` — your speculators repo path.
- `VERIFIER` — your verifier/target model.
- `SUPERSET_LAYERS` — the cache layer ids; **last id = the verifier target layer**.
- `BASELINE_AUX` — the aux layers your draft fuses (subset of the above, without the target).

> ⚠️ `SUPERSET_LAYERS` / `BASELINE_AUX` **must match what your drafts were trained with**, or the fused
> hidden states won't align with the draft's `fc` weights and the numbers are meaningless.

Also set `VOCAB_SRC` (in `env.sh`) to a dir holding the `d2t.npy`/`t2d.npy`/`token_freq.pt` your drafts
were trained with (usually your training dataset dir).

## Usage

### Case A — your drafts target the same verifier (e.g. Qwen3-8B) and you already have a cache
Copy the prebuilt `<verifier>_bench_dense/` cache next to your `output_dir/`, then:
```bash
source env.sh
.venv/bin/python eval_acceptance.py \
    --ckpt /path/to/draft/checkpoint_best \
    --bench /path/to/<verifier>_bench_dense
```

### Case B — build the cache for your verifier (do once), then eval each model
```bash
source env.sh                       # after editing it
bash build_benchmark_cache.sh       # add --thinking for Qwen3 reasoning mode; ~15-20 min, uses all GPUs
# -> writes $BENCH_dense

.venv/bin/python eval_acceptance.py --ckpt /path/to/draft/checkpoint_best --bench ${BENCH}_dense
```
Run the last line once per checkpoint (each takes a few minutes; single-GPU).

## Output
A table printed to stdout, e.g.:
```
category          a1     a2     a3     a4     a5     a6     a7      tau
math_reasoning    0.825  0.666  0.553  0.448  0.358  0.293  0.219   1.880
...
AGGREGATE         0.664  0.477  0.352  0.267  0.205  0.163  0.129   1.128
```
`a_k` = P(draft's k-th drafted token matches the verifier | earlier ones did); `tau` = expected accepted
length per verifier step (higher = bigger speedup). Tee it to a file to keep per-model results.

## Notes / caveats
- **Single-GPU.** `eval_acceptance.py` runs the draft as a `torch.compile`'d model on one GPU (it sets
  `torch._inductor.config.pattern_matcher=False` itself, working around a torch-2.10 inductor bug). No
  torchrun/FSDP. To pick a GPU: `CUDA_VISIBLE_DEVICES=N .venv/bin/python eval_acceptance.py ...`.
- **The cache is verifier-specific** (acceptance is vs *your* verifier's greedy output) — rebuild it if
  you change the verifier; reuse it across drafts of the same verifier.
- **Greedy / teacher-forced** τ, not sampled-acceptance nor a live generation loop (faithful for greedy).
- **Different prompts?** Edit the loading block in `_gen_benchmark.py` (it expects a `prompt` string per
  row); everything downstream is prompt-agnostic.
- The benchmark has 9 domains (~924 prompts); a few of the longest may be dropped during HS extraction —
  this doesn't affect rankings.
