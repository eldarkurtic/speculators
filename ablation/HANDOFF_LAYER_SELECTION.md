# Offload Checkpoint — Verifier Layer-Selection Sweep (DFlash, Qwen3-8B)

> **Self-contained checkpoint for a DEDICATED node.** This ablation — *which verifier hidden-state
> layers (and how many) to feed the draft model* — is largely independent of the main study, so it
> is offloaded here to be brute-forced at scale. The main node (Families A–D) does NOT run this.
> Read `ablation/HANDOFF.md` first for shared context (env, dataset, code map); this file covers
> only what's specific to the layer sweep.

## What this knob is
The draft model fuses N verifier hidden-state layers via `fc = Linear(N*hidden, hidden)` (core.py:78-86)
and predicts the verifier's final-layer distribution. The baseline uses aux layers `1 9 17 25 34`
(target = layer 36). **Which** intermediate layers carry the most useful signal — and **how many** —
is an open, high-impact question worth an exhaustive sweep. Qwen3-8B has 36 layers; valid extraction
ids are `1..36` (36 = final/target).

## Dependencies (do these first on this node)
1. **Hidden-state cache.** This sweep needs a cache that contains *every candidate layer* you want to
   try. Two options:
   - **(A) Reuse the main 10-layer superset** `1 5 9 13 17 21 25 29 34 36` — rsync `CACHE_DIR` from the
     main node (~1.8TB). Lets you try any subset of the 9 aux layers. Moderate granularity.
   - **(B) Generate a DENSER superset here (recommended for true brute force).** Edit
     `ablation/env.sh` → `SUPERSET_LAYERS="2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36"`
     (every-other; 17 aux + target 36; **~3.2TB**, check `df -h` first), then `bash ablation/gen_cache.sh`.
2. **Loader aux-subset change** (HANDOFF.md change #1) — REQUIRED. The cache stores all superset
   layers; `train/data.py._get_raw_data` must select the channels matching each run's
   `--target-layer-ids` (keep `[:, -1]` as the layer-36 target). Implement + sanity-check that
   reproducing the baseline aux set from the cache matches the baseline curves.

## Sweep design (brute force)
Headline metric: **EAL** (`ablation/eal.py`) + val loss + per-position acc. Screen @2 epochs, promote
top configs @5. Keep target fixed at layer 36; vary the AUX set only. Suggested coverage:

- **Single-layer (which one layer is best):** one run per candidate aux layer (9 or 17 runs).
- **Count sweep (how many helps):** best-1 → add next-best greedily (forward selection), measuring
  EAL at N = 1,2,3,4,5,6,7,8(,...). Reveals diminishing/negative returns.
- **Fixed-N subset search:** for N ∈ {3,4,5}, enumerate C(pool, N) subsets (with the 9-layer pool:
  C(9,3)=84, C(9,4)=126, C(9,5)=126). With a dedicated node + cache reads, these are cheap @2ep —
  this is the "much larger sweep" the dedicated node is for.
- **Structured contrasts:** early-heavy {1,5,9} vs mid {13,17,21} vs late-heavy {25,29,34} vs
  uniform-spread vs baseline {1,9,17,25,34}.

Generate the queue programmatically (e.g. itertools.combinations) into `ablation/queue_layers.txt`
(same `<name> | <args>` format), then `bash ablation/queue.sh` (it reads `queue.txt` by default —
either rename, or `QUEUE=ablation/queue_layers.txt` after a tiny edit to queue.sh, or symlink).
Each line overrides only `--target-layer-ids`, e.g.:
```
laux-25-29-34 | --target-layer-ids 25 29 34
laux-1-9-17-25-34 | --target-layer-ids 1 9 17 25 34
```
(All candidate layer ids MUST be present in the cache's `SUPERSET_LAYERS`.)

## Run / verify
- `bash ablation/run.sh <name> <gpu> -- --target-layer-ids <ids>` (reads cache, 2 ep).
- Verify each run finishes 2 ep and logs per-position acc; rank by EAL via `ablation/eal.py --scan ablation/logs`.

## Report back to the main study
Return: best aux set(s) by EAL at N=3/4/5, the EAL-vs-N curve, and the single-best layers. The main
study's Phase-3 will stack the winning layer set with the best optimizer/loss/data settings.

## STATUS (update as you go)
- [ ] cache present (option A rsync, or option B dense-gen)
- [ ] loader subset change applied + baseline-from-cache verified
- [ ] single-layer sweep
- [ ] count/forward-selection sweep
- [ ] fixed-N subset search
- [ ] winners reported back to main node
