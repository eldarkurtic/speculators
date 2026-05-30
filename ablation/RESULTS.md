# DFlash Ablation — Results Leaderboard

Metric of record: **val loss** (lower better) and **EAL** = Σ_k Π_{i≤k} acc_i (higher better; see
`eal.py`). All screening runs are 2 epochs vs the cache-baseline control unless noted. wandb is the
source of truth; this file is the human-readable summary. Append a row per run.

## Control
| run | epochs | val loss | full acc | EAL | notes |
|-----|-------:|---------:|---------:|----:|-------|
| baseline (cache, aux 1·9·17·25·34) | 2 | — | — | — | reproduces original baseline within noise |

## A — optimizer / schedule
| run | val loss | EAL | notes |
|-----|---------:|----:|-------|

## B — capacity / architecture
| run | val loss | EAL | notes |
|-----|---------:|----:|-------|

## C — loss
| run | val loss | EAL | notes |
|-----|---------:|----:|-------|

## D — data / augmentation
| run | val loss | EAL | notes |
|-----|---------:|----:|-------|

## E — layer selection (offloaded, dedicated node)
See `HANDOFF_LAYER_SELECTION.md`. Paste winners here when reported back.
| aux layers | val loss | EAL | notes |
|-----------|---------:|----:|-------|

## Phase 3 — promoted / combined (5 epochs)
| run | config | val loss | EAL | notes |
|-----|--------|---------:|----:|-------|
