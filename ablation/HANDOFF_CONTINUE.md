# DFlash Ablation — Session Handoff (continue from here)

**Read this first.** It is the single entry point for continuing the DFlash-for-Qwen3-8B ablation.
Full results live in [`RESULTS.md`](RESULTS.md) (leaderboards) and
[`report/REPORT.md`](report/REPORT.md) (narrative + 8 plots). Durable findings are also in the agent
memory at `~/.claude/projects/-home-eldarkurtic-github-eldarkurtic-speculators/memory/`.

> NOTE: the original [`HANDOFF.md`](HANDOFF.md) / [`HANDOFF_LAYER_SELECTION.md`] describe the *pre-session*
> plan (optimizer/loss/layer "families A–E"). That plan was **superseded**: we dropped the
> layer-selection sweep and focused on **architecture variants + masking + loss functions**. Use THIS
> file as the source of truth.

---

## 1. Current state (TL;DR)
- **Machine:** 8×H100 node `specs-h100-02`. Idle now; all GPUs free; 1.4 TB free on `/home`.
- **Branch:** `dflash-ablation`. **All work is UNCOMMITTED** (see §8 git). Transfer between boxes by
  **rsync of the repo dir** (origin is upstream `vllm-project/speculators`, no push).
- **Best recipe found (EAL 1.314, +37% over baseline):**
  5-layer draft · **sliding-window-128 attention (all layers)** · **FFN 6144** · **causal** intra-block
  mask · linear fusion · CE-train then **jsd fine-tune @lr 3e-4**.
- **Best checkpoints** (`output_dir/abl_ckpts/<run>/checkpoint_best/`, has config.json + model.safetensors):
  - `prod-swa128-wh` — CE @15ep, EAL **1.262** (the strong single-stage model).
  - `p10-jsdft-3e4` — two-stage CE→jsd fine-tune, EAL **1.314** (overall best).
  - `p9-ce` — best arch + causal, CE @5ep (clean control).
  - Per-epoch checkpoints were pruned to reclaim disk; only `checkpoint_best` kept per run.
- wandb project: **`speculators-scripts-v2`** (runs named `abl-<name>`).

---

## 2. Environment & infrastructure
- **Two venvs** (paths in `ablation/env.sh`, which every script sources):
  - `PY=$REPO/.venv/bin/python` — training (editable `speculators`, torch 2.10, **no vLLM**, no ruff/mypy).
  - `VLLM_PY=$REPO/.venv_vllm/bin/python` — vLLM 0.22 (hidden-state extraction server only).
  - matplotlib was `uv pip install`-ed into `.venv` for the report.
- **Verifier:** `/home/eldarkurtic/hf_models/Qwen/Qwen3-8B` (36 layers, hidden 4096, 32/8 heads, FFN 12288).
- **Dataset:** `output_dir/Qwen3-8B_magpie_5k_dense/` — the **dense** 3613-sample set (see §5 gotcha).
  `env.sh` `CACHE_DIR` = its `hidden_states/` (contiguous symlinks; 6 layers `1 9 17 25 34 36`, ~658 GB).
- **Harness (`ablation/`):**
  - `env.sh` — paths/config + `WANDB_PROJECT=speculators-scripts-v2`. Source everywhere.
  - `gen_cache.sh [N]` — (re)generate the hidden-state cache (N samples; omit for all). Two-venv split.
  - `_densify_cache.py` — rebuild the dense contiguous dataset from a sparse cache.
  - `run.sh <name> <gpu> -- <override args>` — one run on one GPU off the cache. Injects baseline
    defaults (dflash, num-layers 5, aux `1 9 17 25 34`, lr 6e-4, cosine, max-anchors 3072, **epochs 2**,
    `--on-missing raise`, `--logger wandb`, save to `output_dir/abl_ckpts/<name>`).
  - `queue.sh [N]` — fan `queue.txt` across GPUs. **`GPUS="2 3 4"` env var picks specific GPUs.**
    `queue.txt` lines: `<name> | <override args>` (`#` comments skipped).
  - `check_cache.py`, `eal.py` (`--scan logs/`), `_smoke_variants.py` (GPU build+forward smoke test).
  - `report/make_report_plots.py` — regenerate all report plots from `logs/`.
- **Run a sweep:** edit `queue.txt` → `bash ablation/queue.sh` (or `GPUS="..." bash ablation/queue.sh`)
  → `.venv/bin/python ablation/eal.py --scan ablation/logs` → append to `RESULTS.md`.
- **Metric of record: EAL** (`val/eal_epoch` in logs) = `Σ_k Π_{i≤k} acc_i`. Rank by EAL, **not val-loss**
  (loss magnitudes differ across loss types). Screen @2ep, promote @5/15ep.

---

## 3. Code changes made this session
**New CLI knobs** (`scripts/train.py`) + config fields (`dflash/config.py`) + model logic. All default
to the original behavior (baseline reproduces byte-for-byte):
| flag | where | default |
|------|-------|---------|
| `--draft-intermediate-size` | `create_transformer_layer_config` | inherit verifier (12288) |
| `--draft-num-heads` / `--draft-num-kv-heads` | same | inherit (32/8) |
| `--no-decoder-mlp` | `model_definitions.Qwen3DFlashDecoderLayer` | MLP on |
| `--fusion-type {linear,mlp,gated,weighted_sum}` | `core._build_fusion` | linear |
| `--draft-sliding-window N` | `attention.create_anchor_block_mask_mod` (distance bound) | None (full) |
| `--draft-block-causal` | same (`same_block_mod`) | False (bidirectional) |
| `--swa-layer-pattern "ssfff"` | `core.forward` (per-layer mask) | None (uniform) |
| `--loss-type … jsd` + `jsd_loss` | `metrics.py` / `dflash/metrics._select_loss_fn` | ce |
- **`--from-pretrained` now sets `loss_type/loss_gamma/label_smoothing`** (was left None → crash) — this
  enables loss-switched fine-tuning (two-stage).
- **`scripts/train.py` top:** `torch._inductor.config.pattern_matcher = False` — works around a torch
  2.10 inductor joint-graph crash on the DFlash `@torch.compile` forward at real shapes. **Keep this.**
- **`queue.sh`:** fixed GPU-list parsing (`GPU_IDS=($(seq …))`, not `read <<<`) + added `GPUS` override.
- **`check_cache.py`:** fixed empty-`CACHE_DIR` fallback bug.

**Files touched:** `scripts/train.py`, `src/speculators/models/dflash/{config,core,model_definitions,
attention,metrics}.py`, `src/speculators/models/metrics.py`, `ablation/*`. **eagle3 untouched.**
Smoke-tested via `ablation/_smoke_variants.py` (all variants build + forward + backward, finite loss).
`make quality` NOT run (ruff/mypy absent from `.venv`).

---

## 4. Full experiment history (what we tried)
All @ the fixed baseline layer set (`1 9 17 25 34` + target 36), dense cache. EAL = headline metric.
| phase | what | result |
|------:|------|--------|
| 1 | Generate 6-layer hidden-state cache (8 GPUs) | 3613/5000 samples (28% dropped) → dense set |
| 2–3 | **Screening** 15 arch variants @2ep | sliding window dominates (+13%); smaller FFN, wsum, depth-2 help; deeper/complex-fusion/fewer-heads hurt |
| 4 | Promote winners @5ep + combos | clean combo (swa+FFN6144) > hand-stacked combo; **wsum & depth-2 regressed @5ep** (2ep mis-ranked marginals) |
| 5 | Refine window size @5ep | **smaller window better** (64–128 > 256 > 512 > full); win+FFN6144 best ~1.072 |
| 6 | **Production @15ep** | `prod-swa128-wh` EAL **1.262** (best single-stage); more epochs lift EAL a lot |
| 7 | Causal vs bidirectional intra-block mask | ~wash, tiny edge to causal → **adopt causal** |
| 8 | Mix SWA/full attention across 5 layers | **uniform all-SWA best; any full layer hurts** (monotonic) |
| 9 | Loss functions standalone @5ep | ce ≈ kl ≈ kl_ce (~1.07); **reverse_kl/jsd/lk fail to train from scratch** (vanishing grad) |
| 9b | lr-rescue for failing losses | reverse_kl trains only @lr1e-4 (0.766<CE); jsd/lk still flat at any lr |
| 10 | **Two-stage** CE→soft-loss fine-tune | **jsd-ft 1.314 / lk-ft 1.304 > CE-cont 1.277 > start 1.262** — soft losses regularize, generalize better |

**Knobs confirmed already-optimal (no change):** `fusion_type=linear`, uniform (non-mixed) attention.
**Knobs that hurt:** deeper draft (7), wider FFN, fewer heads, MLP/gated fusion.

---

## 5. Gotchas / landmines (read before running)
1. **Dense dataset:** the offline generator dropped ~28% ("incomplete metadata"); we train on the dense
   3613-sample set (`*_magpie_5k_dense`, symlinked cache). A **full-data regen** (point `env.sh DATASET`
   back to `Qwen3-8B_magpie_5k`, rerun `gen_cache.sh`, re-densify) would lift absolute numbers — relative
   rankings should hold.
2. **Inductor workaround is required** (`pattern_matcher=False`) — without it training crashes at compile.
3. **DFlash forward needs `@torch.compile`** for flex attention; can't run eager. Sliding window is
   implemented in the **flex anchored-block mask**, not via the attn `sliding_window` kwarg.
4. **EAL not val-loss** for cross-loss comparisons.
5. **`--from-pretrained`** is the clean way to do loss-switched / continued fine-tunes (loads arch from
   the saved `config.json`; arch flags like `--num-layers` are then ignored).
6. **Disk:** per-epoch checkpoints are huge (~5 GB each). They were pruned to `checkpoint_best` only;
   future long sweeps should prune or set fewer save points to avoid filling `/home`.
7. **`prod-combo`** resumed from an epoch-6 checkpoint (after a `queue.sh` bug, now fixed) — its epoch
   axis is offset; ignore it for per-epoch analysis.

---

## 6. Recommended next steps (prioritized)
1. **Finalize the deployable checkpoint:** run the full recipe end-to-end on regenerated **full data** —
   CE to convergence then `jsd` fine-tune @3e-4. Start from `p10-jsdft-3e4`'s config.
2. **Measure sampling-based acceptance** (not just greedy EAL). `lk`/TVD optimizes sampled-acceptance
   overlap and may win there even though it ~tied jsd on greedy EAL.
3. **vLLM-side support** for windowed/causal DFlash attention — required to actually *serve* the winner;
   the trained checkpoints are valid `speculators` models but the stock vLLM DFlash path won't run them.
4. **Two-stage tuning:** sweep fine-tune epochs / lr / `kl_ce`-then-`lk`; jsd-ft peaked then dipped
   slightly at 5ep (mild overfit) — try 2–3 ft epochs.
5. **Re-confirm depth/heads** on the final recipe (they were screened on the baseline, not the winner).

---

## 7. The winning recipe (commands)
```bash
source ablation/env.sh
# Stage 1 — CE to convergence (best single-stage; matches prod-swa128-wh but add causal):
bash ablation/run.sh final-stage1 0 -- \
  --num-layers 5 --draft-sliding-window 128 --draft-intermediate-size 6144 \
  --draft-block-causal --loss-type ce --epochs 15
# Stage 2 — jsd fine-tune from the stage-1 checkpoint:
bash ablation/run.sh final-stage2 0 -- \
  --from-pretrained output_dir/abl_ckpts/final-stage1/checkpoint_best \
  --num-layers 5 --draft-sliding-window 128 --draft-intermediate-size 6144 \
  --draft-block-causal --loss-type jsd --lr 0.0003 --epochs 5
```

## 8. Git / transfer
Branch `dflash-ablation`, **uncommitted** modified files: `scripts/train.py`,
`src/speculators/models/dflash/{config,core,model_definitions,attention,metrics}.py`,
`src/speculators/models/metrics.py`, all `ablation/*`. Untracked: `ablation/_densify_cache.py`,
`ablation/_smoke_variants.py`, `ablation/report/`, `CLAUDE.md`. Consider committing on the branch before
handoff. Transfer between machines via **rsync of the repo dir** (+ the cache if reusing it), not push.
