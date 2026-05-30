# DFlash Ablation Study — `ablation/`

Goal: improve the DFlash speculator for Qwen3-8B (train/val loss + token acceptance) via a large,
cache-accelerated ablation sweep.

**Start here:** [`HANDOFF.md`](HANDOFF.md) — full context + resume checklist (this is the portable
session checkpoint). Layer-selection is offloaded to a dedicated node:
[`HANDOFF_LAYER_SELECTION.md`](HANDOFF_LAYER_SELECTION.md).

## Files
| file | purpose |
|------|---------|
| `HANDOFF.md` | main resume checkpoint: context, env facts, code map, code changes, ablation matrix |
| `HANDOFF_LAYER_SELECTION.md` | self-contained checkpoint for the offloaded layer-selection sweep |
| `env.sh` | central paths/config — **source in every script**; edit `REPO`/`VERIFIER` if paths move |
| `gen_cache.sh` | Phase 1: generate hidden-state cache once (vLLM server → offline-gen → validate → kill) |
| `check_cache.py` | sanity-check cache file count + sample shape |
| `run.sh` | run one ablation on one GPU from cache (`run.sh <name> <gpu> -- <overrides>`) |
| `queue.txt` | the run matrix (`<name> | <args>`); fill after cache + code changes land |
| `queue.sh` | fan out `queue.txt` across 8 GPUs (1 run/GPU, launch-on-free) |
| `eal.py` | compute Expected Accepted Length; `--scan logs/` for a quick leaderboard |
| `RESULTS.md` | human-readable leaderboard |
| `logs/` | per-run stdout (`<name>.log`) |

## Quickstart (new node)
```bash
cd /home/eldarkurtic/github/speculators
venv_spec/bin/python -c "import speculators, torch; print(torch.cuda.device_count())"   # expect 8
bash ablation/gen_cache.sh 8          # dry-run cache on 8 samples; then full: bash ablation/gen_cache.sh
venv_spec/bin/python ablation/check_cache.py
# apply loader subset change (HANDOFF change #1), then:
bash ablation/run.sh baseline 0 -- --epochs 5     # control
# ... implement CLI knobs / loss variants, fill queue.txt, then:
bash ablation/queue.sh
```

**Transfer note:** `origin` is the upstream `vllm-project/speculators` (no push). Move work between
servers by **rsync of the repo dir** (and the cache if reusing it), not `git push`. Work is on branch
`dflash-ablation`.
