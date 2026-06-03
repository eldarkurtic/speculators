"""Benchmark-acceptance plots (figs 15-19) for the report. Parses the per-checkpoint
eval_<ckpt>.log files written by ablation/eval_acceptance.py.

Run:  .venv/bin/python ablation/report/make_benchmark_plots.py
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGS = Path(__file__).resolve().parents[1] / "logs"
OUT = Path(__file__).resolve().parent / "plots"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, GREEN, RED, GREY, ORANGE, PURPLE = (
    "#2b6cb0", "#2f855a", "#c53030", "#a0aec0", "#dd6b20", "#805ad5")
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def parse_eval(name):
    """Return (per_domain {dom: tau}, aggregate {'a':[a1..a7], 'tau':tau})."""
    f = LOGS / f"eval_{name}.log"
    if not f.exists():
        return {}, None
    per, agg = {}, None
    for line in f.read_text(errors="ignore").splitlines():
        p = line.split()
        if len(p) < 9 or p[0] == "category":
            continue
        m = re.match(r"[A-Za-z_]+$", p[0])
        if not m:
            continue
        try:
            nums = [float(x) for x in p[1:9]]
        except ValueError:
            continue
        if p[0] == "AGGREGATE":
            agg = {"a": nums[:7], "tau": nums[7]}
        else:
            per[p[0]] = {"a": nums[:7], "tau": nums[7]}
    return per, agg


def tau(name):
    _, agg = parse_eval(name)
    return agg["tau"] if agg else None


# magpie val-EAL for the 5 headline checkpoints (from RESULTS.md)
MAIN = [("adamw-ref-5e", "AdamW CE w128", 1.1654, GREY),
        ("muon-lr1e2", "Muon CE w128", 1.4925, BLUE),
        ("ls-kl", "Muon KL w128", 1.5164, GREEN),
        ("w2048-s42", "Muon KL w2048 d5", 1.6192, ORANGE),
        ("d7-s42", "Muon KL w2048 d7", 1.6440, RED)]

# --- Fig 15: magpie val-EAL vs real benchmark tau (inflation + ranking preserved) ---
labels = [m[1] for m in MAIN]
eals = [m[2] for m in MAIN]
taus = [tau(m[0]) for m in MAIN]
x = np.arange(len(MAIN))
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(x - 0.2, eals, 0.4, label="magpie val-EAL", color="#90cdf4")
ax.bar(x + 0.2, taus, 0.4, label="benchmark τ (real)", color="#2f855a")
for i, (e, t) in enumerate(zip(eals, taus)):
    if t:
        ax.text(i + 0.2, t + 0.02, f"{t/e:.0%}", ha="center", fontsize=8, color=RED)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
ax.set_ylabel("accept length")
ax.set_title("Magpie val-EAL vs real benchmark acceptance — EAL overstates by ~30%\n"
             "(labels = τ/EAL; ranking is preserved)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "15_magpie_vs_benchmark.png", bbox_inches="tight")
plt.close(fig)

# --- Fig 16: depth -> benchmark tau + illustrative net-speedup (cost model) ---
DEPTHS = [(2, "d2-s42"), (3, "d3-s42"), (4, "d4-s42"), (5, "w2048-s42"), (7, "d7-s42")]
dvals = [(d, tau(n)) for d, n in DEPTHS if tau(n) is not None]
if dvals:
    ds = [d for d, _ in dvals]
    ts = [t for _, t in dvals]
    # illustrative spec-decode net-speedup proxy: tau / (1 + draft_cost),
    # draft_cost ~ depth/36 (draft layer ~ one verifier layer of the 36-layer Qwen3-8B)
    speed = [t / (1 + d / 36) for d, t in dvals]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(ds, ts, "o-", color=GREEN, label="benchmark τ (acceptance)")
    ax2 = ax.twinx()
    ax2.plot(ds, speed, "s--", color=RED, label="net-speedup proxy τ/(1+d/36)")
    pk = int(np.argmax(speed))
    ax2.annotate(f"speedup peak ~d{ds[pk]}", (ds[pk], speed[pk]),
                 textcoords="offset points", xytext=(0, -28), ha="center",
                 color=RED, fontsize=9, fontweight="bold")
    ax.set_xticks(ds)
    ax.set_xlabel("draft depth")
    ax.set_ylabel("benchmark τ", color=GREEN)
    ax2.set_ylabel("net-speedup proxy (illustrative)", color=RED)
    ax.set_title("Depth on the REAL metric: acceptance keeps rising, but net speedup\n"
                 "peaks at moderate depth — deeper draft costs more than it accepts back")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="lower center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "16_depth_benchmark_speedup.png", bbox_inches="tight")
    plt.close(fig)

# --- Fig 17: window -> benchmark tau (does the magpie window gain hold?) ---
WINS = [(128, "ls-kl"), (256, "w256-f6144-s42"), (1024, "w1024-f6144-s42"),
        (1536, "w1536-s42"), (2048, "w2048-s42"), (4096, "w4096-s42"),
        (8192, "wfull-s42")]
wvals = [(w, tau(n)) for w, n in WINS if tau(n) is not None]
if wvals:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot([w for w, _ in wvals], [t for _, t in wvals], "o-", color=BLUE)
    ax.set_xscale("log", base=2)
    ax.set_xticks([w for w, _ in wvals])
    ax.set_xticklabels([("full" if w == 8192 else str(w)) for w, _ in wvals])
    ax.set(xlabel="sliding window (tokens)", ylabel="benchmark τ",
           title="Window on the REAL metric — gain compresses vs magpie (still ~peaks 1.5-2k)")
    fig.tight_layout()
    fig.savefig(OUT / "17_window_benchmark.png", bbox_inches="tight")
    plt.close(fig)

# --- Fig 18: per-domain tau, best Muon vs AdamW (where does spec-decode help?) ---
per_best, _ = parse_eval("d7-s42")
per_adamw, _ = parse_eval("adamw-ref-5e")
if per_best:
    doms = sorted(per_best, key=lambda d: -per_best[d]["tau"])
    xd = np.arange(len(doms))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(xd - 0.2, [per_best[d]["tau"] for d in doms], 0.4, color=RED,
           label="Muon KL w2048 d7 (best)")
    if per_adamw:
        ax.bar(xd + 0.2, [per_adamw.get(d, {"tau": 0})["tau"] for d in doms], 0.4,
               color=GREY, label="AdamW CE w128")
    ax.set_xticks(xd)
    ax.set_xticklabels(doms, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("benchmark τ")
    ax.set_title("Per-domain acceptance — structured tasks accept 2-3x more than open-ended")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "18_per_domain.png", bbox_inches="tight")
    plt.close(fig)

# --- Fig 19: per-position acceptance a1..a7 on the benchmark (decay) ---
fig, ax = plt.subplots(figsize=(8, 4.4))
for name, lab, _eal, c in MAIN:
    _, agg = parse_eval(name)
    if agg:
        ax.plot(range(1, 8), agg["a"], "o-", color=c, label=lab, ms=4)
ax.set(xlabel="draft position k", ylabel="acceptance a_k (benchmark)",
       title="Per-position acceptance on the benchmark — Muon > AdamW at every position")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "19_per_position_benchmark.png", bbox_inches="tight")
plt.close(fig)

print("Wrote benchmark plots 15-19 to", OUT)
