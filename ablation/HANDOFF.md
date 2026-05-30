# DFlash Ablation Study — Session Handoff / Resume Checkpoint

> **Purpose:** This file lets a fresh Claude Code session (on the new 8×H100 server) resume the
> DFlash speculator ablation study with full context. Read this top-to-bottom, then follow
> **§RESUME HERE**. The original design lives in `/home/eldarkurtic/.claude/plans/prancy-popping-glacier.md`
> on the old box; this file is the portable, repo-tracked copy + live status.

---

## Work split across nodes
- **This (main) node** — all 8 GPUs are now free here. Runs Phase 1 cache + Families **A–D** (and
  Phase 3 promotion). Layer-selection (Family E) is **offloaded** and intentionally excluded here.
- **Dedicated node** — runs only the layer-selection sweep (Family E) at scale; self-contained
  checkpoint in `ablation/HANDOFF_LAYER_SELECTION.md`. Independent of A–D; rejoins at Phase 3.

## TL;DR — what we're doing
Improve a DFlash speculative-decoding draft model for **Qwen3-8B** (train/val loss + token
acceptance). Strategy: cache verifier hidden states **once** to disk (~1.8TB, 10-layer superset),
then sweep training/loss/architecture knobs as fast parallel 1-GPU runs. Screen every config
**@2 epochs**, promote winners **@5 epochs**, then stack compatible winners.

## RESUME HERE — checklist for the new server
1. **Verify env:**
   - `cd /home/eldarkurtic/github/speculators`
   - `venv_spec/bin/python -c "import speculators, torch; print(torch.cuda.device_count())"` → expect 8.
   - `nvidia-smi` → 8 GPUs free (no `llm-compressor` vLLM workers).
   - Confirm paths exist: dataset `output_dir/Qwen3-8B-FP8_magpie_5k/`, verifier
     `/home/eldarkurtic/hf_models/Qwen/Qwen3-8B`. If the home path changed, update `ablation/env.sh`.
   - `df -h /home` → need **≥1.8TB free** for the cache.
2. **Phase 1 — generate cache (once):** `bash ablation/gen_cache.sh` (launches vLLM HS server on 8
   GPUs, runs offline generation, validates, kills server). Confirms 5000 `hs_*.safetensors` with
   shape `[seq, 10, 4096]`. ~tens of minutes.
3. **Phase 1.5 — loader subset + baseline control:** apply the `data.py` subset change (see
   §Code-changes), then run `bash ablation/run.sh baseline-cache 0 -- --epochs 5` and confirm it
   matches the original wandb baseline within noise.
4. **Phase 2 — screening:** edit `ablation/queue.txt` with the run matrix (see §Ablation matrix),
   then `bash ablation/queue.sh` to fan out across 8 GPUs @2 epochs. Results land in
   `ablation/RESULTS.md`.
5. **Phase 3 — promote/combine** top configs @5 epochs; pick final; train long; export for vLLM.

## STATUS (update this section as you go)
- [x] Phase 0: handoff + scaffolding written (this commit). **Done on old box.**
- [ ] Phase 1: hidden-state cache generated & validated.
- [ ] Phase 1.5: loader subset change + baseline-from-cache control.
- [ ] Phase 2: screening ablations @2ep.
- [ ] Phase 3: promote/combine @5ep + final model.

---

## Environment facts (verified on old box — re-verify on new box)
- **venv (training):** `venv_spec/bin/python` (uv venv, editable `speculators`; the `python` on
  PATH). **Has NO vllm** and no ruff/mypy. **Plotting only** uses `data_for_plots/.venv`.
- **venv (vLLM serving):** `VLLM_PY` in `env.sh` = the llm-compressor `venv_lmeval` python (vllm
  0.20.2). `launch_vllm.py` execs `sys.executable -m vllm`, so the HS server MUST run under this; the
  data-gen *client* runs under `venv_spec` (needs speculators+openai, not vllm). `gen_cache.sh`
  already splits them. If `venv_lmeval` moves, set `VLLM_PY`.
- **Dataset:** `output_dir/Qwen3-8B_magpie_5k/` (non-FP8 — user switched from the FP8 variant;
  same structure) — 5000 examples; columns `input_ids`(int32), `loss_mask`(bool, **assistant-only**),
  `seq_len`; lengths 226–8192 (mean ~4025). 90/10 train/val **by index** (train 0–4499, val
  4500–4999). `d2t.npy`/`t2d.npy`/`token_freq.pt` = 32k draft vocab. (Baseline below used the FP8
  dataset; we now use the non-FP8 one. Cache must be regenerated against this dataset's input_ids.)
- **Qwen3-8B:** 36 layers, hidden 4096, 32 heads / 8 KV, head_dim 128, vocab 151936, inter 12288,
  silu, rms_eps 1e-6, rope_theta 1e6.
- **Baseline command (40 min, 1 GPU, 5 ep):**
  ```
  scripts/train.py --verifier-name-or-path /home/eldarkurtic/hf_models/Qwen/Qwen3-8B \
    --data-path output_dir/Qwen3-8B-FP8_magpie_5k --on-missing generate --on-generate delete \
    --scheduler-type cosine --draft-vocab-size 32000 --max-anchors 3072 \
    --target-layer-ids 1 9 17 25 34 --speculator-type dflash --num-layers 5 \
    --logger wandb --run-name test-qwen3-8b-fp8text --lr 0.0006 --epochs 5
  ```
- **Cache (Phase 1):** superset aux/target layers **`1 5 9 13 17 21 25 29 34 36`** (10; contains the
  baseline `1 9 17 25 34 36`). Loader uses `hidden_states[:, :-1]` = first 9 as aux, `[:, -1]` = layer
  36 = `verifier_last_hidden_states` (loss target).

## Key code map (knobs → file:line)
- **CLI:** `scripts/train.py` — `--lr`(1e-4 def), `--epochs`, `--scheduler-type`, `--num-layers`,
  `--block-size`(8), `--max-anchors`(256), `--noise-std`(0.05; applied **train AND val** at
  train.py:289,300,316), `--draft-arch`(llama/qwen3), `--draft-hidden-act`, `--total-seq-len`(8192),
  `--target-layer-ids`.
- **Optimizer:** `train/trainer.py:146` — `AdamW(lr=...)`, **wd=0**, default betas; **no CLI**.
- **Grad clip:** `train/trainer.py:208` — **hardcoded 1.0**.
- **LR sched:** `train/trainer.py:152-183` (warmup = 1% if unset).
- **DFlash loss:** `models/dflash/metrics.py:42-49` — `ce_loss` + `dflash_loss_decay(gamma=4.0)`;
  `gamma` not CLI-exposed (set in `core.py` compute_metrics call ~`core.py:294-304`).
- **Loss kit (already present):** `models/metrics.py` — `kl_div_loss`(fwd KL), `ce_loss`,
  `dflash_loss_decay`, `exp_loss_decay`, `loss_function(loss_fn, decay_fn)`. Fwd-KL = 1-line swap.
- **Loss targets:** `models/dflash/core.py:271-304` — verifier soft logits in-model (32k draft vocab)
  → KL/LK losses need **no data change**.
- **Fusion:** `core.py:78-86` — `fc=Linear(n_aux*hidden, hidden, bias=False)` + RMSNorm.
- **Loader/cache:** `train/data.py:223-330` — `_map_to_file_idx`, `_get_raw_data` (aux `[:,:-1]`/last `[:,-1]`).
- **Noise:** `train/noise_transforms.py` — `AddUniformNoise`/`AddGaussianNoise(std)`.

## Code changes — IMPLEMENTED & unit-verified (branch `dflash-ablation`)
All of these are done, py_compile-clean, and unit-tested (losses/dispatch/EAL/layer-subset on random
tensors; argparse `--help` clean). End-to-end validation = the baseline-from-cache control (§Phase 1.5).
`make quality` not run — ruff/mypy are NOT installed in `venv_spec` (dev extras missing); install via
`uv pip install ruff mypy` into `venv_spec` if you want the style/type gate.
1. **Loader aux-subset** (`train/data.py`): `ArrowDataset(cache_layer_ids=, aux_layer_ids=)` +
   `_resolve_layer_selection`; `--cache-layer-ids` in `train.py` (run.sh passes the superset).
   `[1,9,17,25,34]` from the 10-layer cache → channels `[0,2,4,6,8]`, target = last.
2. **Optimizer knobs** (`trainer.py` `_build_optimizer` + `TrainerConfig`; `train.py`): `--optimizer`
   {adamw,adam,sgd,rmsprop,adafactor,lion}, `--weight-decay`, `--adam-betas`, `--sgd-momentum`,
   `--grad-clip` (replaces the hardcoded 1.0). NOTE: `lion` needs `pip install lion-pytorch`.
3. **Loss variants** (`models/metrics.py` + `dflash/metrics.py` `_select_loss_fn`; `core.py` reads
   `self.loss_type/loss_gamma/label_smoothing` set in `from_training_args`): `--loss-type`
   {ce,kl,reverse_kl,kl_ce,lk}, `--loss-gamma`, `--label-smoothing`. `lk` = `1 - Σ min(p_d,p_t)`
   (acceptance-rate surrogate). Default `ce`+gamma 4.0 reproduces the old hardcoded behaviour.
4. **EAL metric:** `dflash/metrics.py compute_metrics` now logs `eal = Σ_k Π_{i≤k} acc_i`.
5. **`--noise-type` {uniform,gaussian}** (`train.py`). CORRECTION: the val loader already gets NO
   noise (it never passed `transform=`), so the "remove val noise" idea was moot and was dropped.
6. **`--loss-on-all-tokens`** (`train.py`/`data.py`): overrides the assistant-only mask on the TRAIN
   split only; val keeps assistant-only as the metric of record.
7. *(NOT done — stretch)* **Attention-drift alignment** (matrix F): needs verifier attention maps
   (not extracted) → KL(draft attn ‖ target attn). Heavy: changes data-gen + model. Only if Phase-2
   signal warrants.

## Ablation matrix (screen @2ep; ~50+ runs)
Rank by **val loss** + **EAL** + per-position acc. One knob at a time vs the cache-baseline control.

**A. Optimizer / schedule**
- optimizer ∈ {AdamW(base), SGD+momentum, Lion, Adafactor, RMSprop, (Muon/Sophia if avail)}
- lr ∈ {3e-4, 6e-4(base), 1e-3, 2e-3} (per-optimizer lr may need rescaling: Lion ~3–10× lower, SGD higher)
- weight_decay ∈ {0(base), 0.01, 0.1}; adam betas {(0.9,0.95) vs (0.9,0.999)}
- grad clip ∈ {0.5, 1.0(base), 2.0}; warmup ∈ {1%(base), 3%, 5%}; cosine vs linear
- total-seq-len ∈ {8192(base), 16384} (effective batch)

**B. Capacity / architecture**
- num-layers ∈ {3, 5(base), 7, 10}; draft-arch qwen3 vs llama; draft-hidden-act variants
- block-size ∈ {4, 8(base), 12}; max-anchors ∈ {1024, 3072(base)} (changes task → use per-pos acc, not raw loss)
- fusion fc: Linear → 2-layer MLP / gated (core.py:78-86)

**C. Loss**
- position weighting (D-PACE): gamma ∈ {1,2,4(base),8,∞}; `exp_loss_decay` vs `dflash_loss_decay`; learnable per-pos weights
- distillation: fwd KL vs reverse KL vs KL+CE blend vs CE+label-smoothing
- LK acceptance loss

**D. Data / augmentation**
- `--noise-std` ∈ {0, 0.01, 0.05(base), 0.1}; `--noise-type` gaussian vs uniform (val is already
  un-noised — no toggle needed)
- `--loss-on-all-tokens` (vs assistant-only base)

**E. Layer selection** — **OFFLOADED to a dedicated node**; see `HANDOFF_LAYER_SELECTION.md`. Do not
run on this node.

**F. Attention-drift** (stretch, see change #8).

## Run harness (in `ablation/`)
- `env.sh` — paths (verifier, dataset, cache dir, venv). Source it everywhere. Edit if paths move.
- `gen_cache.sh` — Phase 1 one-time cache generation (launch HS server → offline-gen → validate → kill).
- `run.sh <name> <gpu_id> -- <extra train.py args>` — one ablation: pins GPU, cache + `--on-missing
  raise`, names wandb run, tees `ablation/logs/<name>.log`.
- `queue.txt` — one run per line: `<name> | <extra args>`. `queue.sh` fans them across 8 GPUs (1/GPU).
- `eal.py` — compute EAL from per-position accuracies.
- `RESULTS.md` — leaderboard (append val-loss / EAL / per-pos acc per run).

## Verification
- Phase 1: 5000 cache files; sample shape `[seq,10,4096]`; `token_ids == input_ids`.
- Phase 1.5: baseline-from-cache ≈ original wandb curves.
- Each run: finishes 2 ep; logs val loss + per-pos acc + EAL.
- Each code change: finite/decreasing loss, shapes ok, `make quality` clean, eagle3 untouched.

## Risks / notes
- ~1.8TB cache — check `df -h` first.
- block_size/max_anchors change the objective → don't compare raw loss across them.
- `origin` = `vllm-project/speculators` (upstream, no push). **Transfer via rsync of the repo dir**,
  not `git push`. Work lives on branch `dflash-ablation` (committed locally).
- Papers D-PACE / LK losses / attention-drift are 2026/post-cutoff; implementations here derive from
  fetched abstracts — re-read PDFs (saved under the session `tool-results/` on old box) before coding.
