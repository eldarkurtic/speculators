# DFlash architecture-variant ablation — Results

Metric of record: **val/loss_epoch** (lower better) and **val/eal_epoch** (Expected Accepted Length,
higher better) at the final epoch. Screening runs are **2 epochs** on the dense cache (3613 samples,
baseline aux layers 1·9·17·25·34 + target 36), one architecture knob changed vs the control.
wandb project: `speculators-scripts-v2` (runs `abl-<name>`).

## Screening leaderboard (2 epochs), ranked by EAL
| rank | run | val loss | EAL | Δ EAL vs base | axis |
|-----:|-----|---------:|----:|--------------:|------|
| 1 | swa-256 | 2.129 | 0.789 | +13.2% | sliding window 256 |
| 2 | swa-512 | 2.152 | 0.777 | +11.5% | sliding window 512 |
| 3 | swa-1024 | 2.202 | 0.741 | +6.3% | sliding window 1024 |
| 4 | width-half | 2.200 | 0.737 | +5.7% | FFN 6144 |
| 5 | fusion-wsum | 2.233 | 0.713 | +2.3% | weighted-sum fusion |
| 6 | depth-2 | 2.226 | 0.709 | +1.7% | 2 decoder layers |
| 7 | depth-3 | 2.257 | 0.699 | +0.3% | 3 decoder layers |
| — | **baseline** | **2.258** | **0.697** | — | num-layers 5, linear, MLP on, 32/8 heads, full attn |
| 8 | no-decoder-mlp | 2.240 | 0.692 | -0.7% | drop FFN (≈neutral, cheaper) |
| 9 | width-2x | 2.297 | 0.676 | -3.0% | FFN 24576 |
| 10 | depth-7 | 2.277 | 0.670 | -3.9% | 7 decoder layers |
| 11 | depth-1 | 2.281 | 0.663 | -4.9% | 1 decoder layer |
| 12 | fusion-mlp | 2.343 | 0.657 | -5.7% | 2-layer MLP fusion |
| 13 | fusion-gated | 2.355 | 0.649 | -6.9% | SwiGLU fusion |
| 14 | heads-32-4 | 2.319 | 0.647 | -7.2% | 32 heads / 4 kv |
| 15 | heads-16-4 | 2.323 | 0.640 | -8.2% | 16 heads / 4 kv |

Baseline @2ep per-position acc: p1 .494 · p2 .315 · p3 .200 · p4 .140 · p5 .113 · p6 .098 · p7 .088.

### Takeaways
- **Sliding-window attention is the dominant win, and smaller window is better** (256 > 512 > 1024 >
  full). Strong local-context prior for drafting.
- **Smaller FFN (6144) beats baseline and is cheaper** — baseline FFN over-parameterized for a drafter.
- **Dropping the decoder MLP is ≈neutral** → free speed/size reduction.
- Shallower (2-3 layers) ≥ baseline 5; deeper (7), complex fusion (mlp/gated), and fewer heads hurt.
- Winners lie on independent axes → expected to stack.

## Phase 4 — promoted / combined (5 epochs)
Promoting the independent winners + a stacked "combo", vs the 5-epoch baseline reference.
| run | config | val loss | EAL | notes |
|-----|--------|---------:|----:|-------|
| **p4-combo** | num-layers 2 + FFN6144 + wsum + swa256 | **1.781** | **1.045** | **best EAL (+8.9%)**, ~½-size model |
| **p4-swa256** | --draft-sliding-window 256 | 1.789 | 1.037 | +8.0%, single clean change (most robust) |
| p4-swa512 | --draft-sliding-window 512 | 1.815 | 1.008 | +5.0% |
| p4-width-half | --draft-intermediate-size 6144 | 1.855 | 0.982 | +2.3%, smaller |
| baseline-5ep | num-layers 5 (your original setup) | 1.867 | 0.960 | reference; EAL/ep 0.559→0.767→0.908→0.947→0.960 |
| p4-combo-lean | combo + --no-decoder-mlp | 1.882 | 0.948 | -1.3% (dropping MLP hurts in the combo) |
| p4-fusion-wsum | --fusion-type weighted_sum | 1.892 | 0.936 | -2.5% (REGRESSED at 5ep; was +2.3% @2ep) |
| p4-depth-2 | --num-layers 2 | 1.907 | 0.931 | -3.0% (REGRESSED at 5ep; was +1.7% @2ep) |

### Phase-4 conclusions
- **Sliding-window attention is the robust, dominant win**: `swa-256` holds at 5 epochs (EAL **1.037**,
  +8% over baseline) as a single clean change — strongly recommended.
- **`p4-combo` is best overall (EAL 1.045, +8.9%) AND ~half the size** (2 layers, FFN 6144) → better
  *and* a cheaper/faster drafter. Its win is driven by swa-256 + the smaller model training efficiently.
- **Caveat — 2ep screening mis-ranked the marginal winners**: `fusion-wsum` and `depth-2` looked good
  @2ep (+2.3%, +1.7%) but **regressed @5ep** (-2.5%, -3.0%). Only swa-* and width-half held up. Lesson:
  marginal 2ep gains need 5ep confirmation; large 2ep gains (swa) were reliable.
- **Dropping the decoder MLP hurts once combined** (`combo-lean` < `combo`), despite being ≈neutral alone.

### Recommended next step  [DONE — see Phase 5]
A cleaner combo of only the *robust* winners — **swa + width-half** (drop wsum & depth-2) — may beat
both `swa-256` alone and `p4-combo`; worth one more 5ep run before finalizing.

## Phase 5 — refine window + clean combo (5 epochs)
Smaller window kept winning, so pushed to 64/128/192; clean combo = sliding window + smaller FFN only.
| run | val loss | EAL | Δ vs base | notes |
|-----|---------:|----:|----------:|-------|
| **ref-swa64-wh** (win 64 + FFN 6144) | 1.752 | **1.072** | +11.7% | best architecture |
| ref-swa128-wh (win 128 + FFN 6144) | 1.753 | 1.071 | +11.6% | ~tied best |
| ref-swa256-wh (win 256 + FFN 6144) | 1.770 | 1.057 | +10.1% | |
| ref-swa64 (win 64) | 1.792 | 1.040 | +8.3% | |
| ref-swa128 (win 128) | 1.805 | 1.024 | +6.7% | |
| ref-swa192 (win 192) | 1.813 | 1.019 | +6.1% | |

**Conclusion: best architecture = small sliding window (64-128) + smaller FFN (6144).** EAL ~1.072
(+12% over baseline-5ep 0.960), and beats the original p4-combo (1.045) — confirming wsum & depth-2
were dead weight. Window size in 64-256 is a gentle knob (1.040-1.072); FFN-6144 adds the rest.
(swa256-wh/swa64/swa128/swa192 shown at epoch 4; ranking already settled — width-half combos lead.)

## Phase 6 — production training @15 epochs (deployable checkpoints; best-by-val-loss checkpoint)
More epochs lift EAL substantially (e.g. combo 1.045@5ep -> 1.208@15ep). Best val_loss / its EAL:
| run | config | val loss | EAL | status |
|-----|--------|---------:|----:|--------|
| **prod-swa128-wh** | window 128 + FFN6144, 5-layer | **1.555** | **1.262** | done — BEST |
| prod-swa256 | sliding-window 256 (cleanest single change) | 1.569 | 1.252 | done |
| prod-swa64-wh | window 64 + FFN6144 | 1.579 | 1.243 | done |
| prod-combo | 2-layer + FFN6144 + wsum + swa256 | 1.603 | 1.208 | done |
Bar to beat (baseline-5ep): 0.960. Winner **prod-swa128-wh EAL 1.262 (+31% over baseline-5ep)**.
Checkpoints in output_dir/abl_ckpts/<run>/checkpoint_best.

## Phase 7 — causal vs bidirectional intra-block masking (@5ep)
| config | bidirectional EAL | causal EAL |
|--------|------------------:|-----------:|
| swa128-wh | 1.071 | 1.072 |
| swa64-wh | 1.072 | 1.081 |
| swa256 | 1.037 | 1.039 |
**Conclusion: ~wash, tiny consistent edge to causal (never worse).** Adopt causal masking — it's also
the inference-correct choice for drafting. Best Phase-7 cell: causal swa64-wh (1.081).

## Phase 8 — mix sliding-window & full attention across 5 layers (@5ep, causal, window 128, FFN 6144)
Pattern is per-layer (layer 0->4): 's' = sliding window 128, 'f' = full attention.
| pattern | EAL | note |
|---------|----:|------|
| sssss (all SWA) | 1.066 | **best** |
| fsssf | 1.054 | full edges, SWA core |
| ffsss | 1.042 | |
| sssff | 1.038 | |
| sfsfs | 1.022 | alternating |
| ssfff | 1.017 | |
| fffss | 1.017 | |
| fffff (all full) | 0.980 | = width-half + causal |
**Conclusion: uniform all-SWA is best; any full-attention layers hurt (monotonic — more 'f' -> lower
EAL). The local-context prior helps at EVERY layer. No mixing.**

## FINAL recommended DFlash architecture (Qwen3-8B)
**5-layer draft + sliding-window attention (window 128, all layers) + FFN 6144 + causal intra-block
masking** (linear fusion, decoder MLP retained, 32/8 heads — all baseline). Flags:
`--num-layers 5 --draft-sliding-window 128 --draft-intermediate-size 6144 --draft-block-causal`.
Best checkpoint so far: `prod-swa128-wh` @15ep, **EAL 1.262 / val-loss 1.555** (+31% EAL over the
5-epoch baseline 0.960); that run is bidirectional — a causal re-train should match or slightly beat it.
Caveats: (1) windowed DFlash needs vLLM-side support to actually serve; (2) trained on the dense 3613-
sample cache — a full-data regen would raise absolute quality further.

## Phase 9 — loss functions (@5ep on best arch; ranked by EAL — val-loss not comparable across types)
| loss | EAL | full_acc | note |
|------|----:|---------:|------|
| ce (control) | 1.072 | 0.289 | hard-label cross-entropy |
| kl_ce | 1.068 | 0.290 | ~tied |
| kl (forward) | 1.067 | 0.289 | ~tied |
| reverse_kl | 0.056 | 0.052 | DID NOT TRAIN (loss ~13, fluctuating — lr too high/unstable) |
| jsd (new) | 0.056 | 0.052 | DID NOT TRAIN (loss flat ~0.34 — near-zero gradient, lr too low) |
| lk (acceptance) | 0.037 | 0.035 | DID NOT TRAIN (loss flat ~0.53 — near-zero gradient, lr too low) |
**Conclusion: ce / forward-kl / kl_ce are interchangeable (~1.07) — no loss beats the CE default.**
reverse_kl/jsd/lk failed to learn at the fixed lr 6e-4 (flat loss & ~random accuracy) — an
optimization/gradient-scale issue, not necessarily the objective. Phase 9b retries them with tuned lr
(jsd/lk higher, reverse_kl lower) to evaluate fairly — esp. lk, which is the principled spec-decode
objective (TVD = acceptance).

### Phase 9b — lr rescue for the non-learning losses (@5ep)
| run | lr | EAL | learned? |
|-----|---:|----:|----------|
| reverse_kl | 1e-4 | 0.766 | yes, but < CE (1.072) |
| reverse_kl | 3e-4 | 0.006 | diverged |
| jsd | 3e-3 / 1e-2 | 0.056 / 0.037 | no (flat) |
| lk | 3e-3 / 1e-2 | 0.037 / 0.032 | no (flat) |

**Final loss conclusion: CE (= forward-KL = KL+CE, ~1.07) is the best standalone loss — keep CE.**
- reverse_kl only optimizes at low lr (1e-4) and still underperforms CE.
- **jsd and lk do not train from random init at any lr** — vanishing gradient when the draft starts
  near-uniform over 32k vocab vs a peaked verifier target (TVD/JSD per-logit gradient ~0 there).
- **lk (the ideal acceptance objective) needs a CE/KL warmup before switching** (two-stage:
  CE-pretrain -> lk-finetune) to be usable — standalone it can't bootstrap. Proposed as next step.

## Phase 10 — two-stage: CE checkpoint -> soft-loss fine-tune (@5ep from prod-swa128-wh 1.262)
Loaded prod-swa128-wh (CE @15ep) via --from-pretrained, fine-tuned 5 epochs with a switched loss;
CE-continuation runs are the control. Ranked by val EAL:
| run | loss | lr | val EAL | full_acc |
|-----|------|---:|--------:|---------:|
| **p10-jsdft-3e4** | jsd | 3e-4 | **1.314** | 0.352 |
| p10-lkft-3e4 | lk | 3e-4 | 1.304 | 0.350 |
| p10-lkft-1e4 | lk | 1e-4 | 1.301 | 0.347 |
| p10-ceft-3e4 (control) | ce | 3e-4 | 1.277 | 0.340 |
| p10-ceft-1e4 (control) | ce | 1e-4 | 1.270 | 0.336 |
| p10-lkft-1e3 | lk | 1e-3 | 1.091 | overshoot |
| start (prod-swa128-wh) | ce | — | 1.262 | — |

**Two-stage WORKS: CE-pretrain then jsd/lk fine-tune (lr 3e-4) beats CE-continuation** (1.314/1.304 vs
1.277) and the starting checkpoint (1.262). lk/jsd have lower *train* EAL but higher *val* EAL — the
soft (full-distribution) target regularizes and generalizes better than CE's hard argmax. lk needs
moderate lr (1e-3 overshoots). Standalone these losses can't bootstrap (Phase 9), but as a fine-tune
stage they're a real win.

## OVERALL BEST RECIPE (pre-Muon; ⚠️ the 1.314 used the BUGGY jsd — see metrics.py fix)
Architecture: **5-layer draft + sliding window 128 (all layers) + FFN 6144 + causal intra-block**.
Training: **CE to convergence, then fine-tune ~3-5 epochs with jsd (or lk) at lr 3e-4.**
Result: **EAL 1.314** (vs 0.960 baseline-5ep = +37%; vs 1.262 pure-CE best = +4%).
NOTE: jsd was numerically broken (negative JSD); this conclusion is being re-run with the fix.

## NEW REGIME (2026-05-31): Muon optimizer, full 4997-sample data, 5 epochs, ranked by val EAL
Fixed arch: 5-layer + SWA-128 + causal + FFN-6144. wd=0.01. CE loss. Single-GPU runs (Muon needs it).

### Phase M0 — Muon LR sweep (CE, 5ep) vs AdamW reference
| run | optimizer / lr | EAL | val_loss | p1 acc |
|-----|----------------|----:|---------:|-------:|
| **muon-lr1e2** | **Muon 0.01** | **1.4925** | 1.351 | .751 |
| muon-lr2e2 | Muon 0.02 | 1.4918 | 1.345 | .751 |
| muon-lr35e3 | Muon 0.035 | 1.4820 | 1.349 | .749 |
| muon-lr5e2 | Muon 0.05 | 1.4758 | 1.353 | .747 |
| muon-lr7e2 | Muon 0.07 | 1.4682 | 1.359 | .746 |
| muon-lr5e3 | Muon 0.005 | 1.4218 | 1.396 | .737 |
| muon-lr25e4 | Muon 0.0025 | 1.2720 | 1.551 | .707 |
| adamw-ref-5e | AdamW 6e-4 | 1.1654 | 1.646 | .668 |

**Conclusion: Muon >> AdamW (+28% EAL @0.01 vs AdamW @5ep, same arch/data).** Muon CE @5ep (1.4925)
already beats the old buggy-jsd two-stage best (1.314) and pure-CE @15ep (1.262). **Best Muon LR =
0.01-0.02** (flat plateau, within-noise; <0.005 drops off). No divergence up to 0.07. EAL peaks @ep4,
dips slightly @ep5 (mild overfit). FROZEN: **Muon lr 0.01** for subsequent waves.

### Phase M1 — loss functions @ Muon lr 0.01 (5ep, fixed arch, full data)
STANDALONE (from scratch):
| loss | EAL | note |
|------|----:|------|
| **kl (forward)** | **1.5164** | **BEST — beats CE (+1.6%)** |
| ce | 1.4925 | control |
| jsd (fixed) | 1.4540 | TRAINS now (traj 0.99->1.45) vs old flat 0.056 @AdamW — confirms the _EPS bug |
| reverse_kl | 1.2075 | trains, < CE |
| lk | 1.0574 | trains slowly, worst |

TWO-STAGE (CE@5ep checkpoint -> soft fine-tune @5ep, Muon 0.005): ts-jsd 1.469, ts-lk 1.467,
ts-kl 1.450, ts-ce 1.393. **ALL below the 1.4925 start; ts-ce control (1.393, 1st ft-epoch dropped to
1.33) proves the fresh cosine-warmup + fresh-momentum schedule restart DEGRADES the converged model
(confounded, not the loss).** Two-stage as configured does not help.

**Conclusion: best loss = STANDALONE forward-KL (EAL 1.5164, +1.6% over CE).** jsd bug fix validated
(all losses bootstrap under Muon; jsd/lk/reverse_kl no longer collapse). Two-stage abandoned (schedule
artifact; would need a gentler fine-tune to test fairly). FROZEN: **kl loss** for Step 2.

### Phase M2a — window + FFN re-confirm @ Muon 0.01 + KL (5ep, 2 seeds, mean±std)
| config | EAL mean ± std |
|--------|---------------:|
| **w1024 · FFN6144** | **1.6051 ± 0.0029** |
| w256 · FFN6144 | 1.5496 ± 0.0022 |
| w128 · FFN6144 | 1.5152 ± 0.0012 |
| w128 · FFN12288 | 1.5113 ± 0.0011 |

**Conclusion: LARGER window is now BETTER, monotonic (128<256<1024), std ~0.001-0.003 so rock-solid —
this REVERSES the old "smaller-window-better" finding (that was AdamW/2ep/3613).** Trend still climbing
at 1024 (peak not yet found → extend to {1536,2048,4096,full} in M2b). FFN-6144 confirmed (12288 ~tied,
marginally worse). Best so far: **w1024 + FFN6144 + KL + Muon 0.01 = EAL 1.6051**.

### Phase M2b — window peak search @ Muon 0.01 + KL (5ep, 2 seeds, mean±std)
| window | EAL mean ± std |
|-------:|---------------:|
| 128 | 1.5152 ± 0.0012 |
| 256 | 1.5496 ± 0.0022 |
| 1024 | 1.6051 ± 0.0029 |
| 1536 | 1.6180 ± 0.0030 |
| **2048** | **1.6194 ± 0.0002** |
| 4096 | 1.6110 ± 0.0005 |
| full | 1.6065 ± 0.0012 |

**Conclusion: inverted-U — broad peak at window ~1536-2048 (≈¼ of the 8192 max seq), then gentle
decline toward full attention.** NOT "smaller better" (old) NOR "bigger always better": wide-but-local
context is optimal. Best: **window 2048 + FFN6144 + KL + Muon 0.01 = EAL 1.6194** (+6.9% over old w128).
FROZEN window 2048 for Step 2c (depth).

### Phase M2c — depth sweep @ window 2048 + Muon 0.01 + KL (5ep, 2 seeds, mean±std)
| depth | EAL mean ± std |
|------:|---------------:|
| 2 | 1.4754 ± 0.0062 |
| 3 | 1.5571 ± 0.0007 |
| 4 | 1.5969 ± 0.0008 |
| 5 (ref) | 1.6194 ± 0.0002 |
| **7** | **1.6449 ± 0.0009** |

**Conclusion: deeper = better, monotonic (2<3<4<5<7), still climbing at 7 — REVERSES the old AdamW
finding (depth-2 good / deeper hurt).** ⚠️ But EAL ignores draft COST: a deeper draft is slower to run,
and the real deployment metric is net speedup = f(EAL, draft cost), not EAL alone. EAL-only sweeping
will keep saying "deeper" — the depth-vs-cost tradeoff needs the deployment/acceptance benchmark to
resolve. Best EAL: **depth 7 = 1.6449**; balanced choice pending the speedup benchmark.


## Benchmark acceptance (RedHatAI/speculator_benchmarks, 841 prompts, teacher-forced greedy τ)
Per-position acceptance + accept-length τ against the VERIFIER's own greedy continuation (the real
spec-decode target), via ablation/eval_acceptance.py on the offline HS cache (build_benchmark_cache.sh).
| checkpoint | config | magpie EAL | bench τ | τ/EAL |
|-----------|--------|-----------:|--------:|------:|
| d7-s42 | KL · w2048 · depth 7 | 1.644 | 1.172 | 0.71 |
| w2048-s42 | KL · w2048 · depth 5 | 1.619 | 1.157 | 0.71 |
| ls-kl | KL · w128 | 1.516 | 1.128 | 0.74 |
| muon-lr1e2 | CE · w128 | 1.493 | 1.105 | 0.74 |
| adamw-ref-5e | AdamW · w128 | 1.165 | 0.829 | 0.71 |

**Conclusions:** (1) magpie val-EAL OVERSTATES real acceptance by ~30% (τ ≈ 0.71-0.74×EAL). (2) Rankings
HOLD on the benchmark — Muon≫AdamW (+33%), KL>CE (+2%), wider/deeper better — validating the ablation on
the deployment distribution. (3) Arch gains SHRINK: window +6.9%→+2.6%, depth7-vs-5 only +1.3% for +40%
draft layers. **Full benchmark curves:** depth τ = d2 1.048 · d3 1.108 · d4 1.136 · d5 1.157 · d7 1.172
(monotonic↑); window τ = w128 1.128 · w256 1.151 · w1024 1.157 · w1536 1.157 · w2048 1.157 · w4096 1.149
· full 1.144 (SATURATES at ~1024). Spec-decode net-speedup proxy τ/(1+depth/36): d2 0.993 · **d3 1.023
(peak)** · d4 1.022 · d5 1.016 · d7 0.981 → **deployment-optimal depth ~3-4** (proxy peaks ~3, flat
through 5; depth 7 = EAL-optimal but WORSE for speedup), **window ~1024** (saturates, cheaper-equal to
2048). (Exact depth peak depends on the real draft/verifier cost ratio; proxy is illustrative — d7 is
clearly past optimum.) Per-domain τ spread: math~1.9 / HumanEval~1.6 (structured) vs summarization~0.7
/ qa~0.8 (open-ended), 2-3x. Eval harness: ablation/{build_benchmark_cache.sh,eval_acceptance.py}.

## Phase M3 — SWA/full layer-position mix @ optimal Muon recipe (repeat of Phase 8, new regime)
Base: Muon lr 0.01 mom 0.95 + forward-KL + causal + FFN 6144 + depth 5 + **window 1024**, 5ep, full
4997 data. 's' = sliding window 1024, 'f' = full attention (layers 0->4). Ranked by benchmark τ:
| pattern | magpie EAL | bench τ | note |
|---------|-----------:|--------:|------|
| ffsss | 1.6038 | 1.158 | τ-best |
| fsssf | 1.6073 | 1.154 | |
| sfsfs | 1.6138 | 1.152 | EAL-best |
| sssss | 1.6043 | 1.151 | uniform window-1024 reference |
| ssfff | 1.6116 | 1.149 | |
| fffss | 1.6071 | 1.147 | |
| sssff | 1.6059 | 1.146 | |
| fffff | 1.6053 | 1.144 | all-full, worst on both |

**Conclusion: layer-mix is now IRRELEVANT — REVERSES Phase 8.** Both metrics are flat: EAL spread
0.010 (~0.6%), τ spread 0.014 (~1.2%) — within seed noise (±0.002-0.006). The old Phase-8 (AdamW/w128)
had an 0.086 spread with uniform all-SWA clearly best and full hurting monotonically; at window 1024 the
s-vs-f contrast is too small to matter. EAL and τ even DISAGREE on the order (sfsfs EAL-#1/τ-#3; ffsss
EAL-last/τ-#1) → confirms noise, not signal. all-full (fffff) is marginally worst on both; uniform
sssss is statistically tied with the best. **Practical: keep it simple — uniform window-1024, no mixing.**
Per-domain (fig 20): no pattern wins any task consistently (spreads 0.007-0.048, mostly noise).
