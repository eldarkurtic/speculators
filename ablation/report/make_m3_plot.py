"""Phase M3 plot: per-task benchmark τ for the SWA/full layer-mix patterns, at the
optimal Muon recipe (window 1024). Color = deviation from each task's mean (shows the
pattern differences are within noise); annotations = raw τ.

Run: .venv/bin/python ablation/report/make_m3_plot.py
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGS = os.path.join(os.path.dirname(__file__), "..", "logs")
OUT = os.path.join(os.path.dirname(__file__), "plots")
PATS = ["ffsss", "fsssf", "sfsfs", "sssss", "ssfff", "fffss", "sssff", "fffff"]
DOMS = ["HumanEval", "math_reasoning", "qa", "question", "rag",
        "summarization", "tool_call", "translation", "writing"]


def per_domain(pat):
    f = os.path.join(LOGS, f"eval_mix-{pat}.log")
    out = {}
    for line in open(f):
        s = line.split()
        if len(s) >= 9 and re.match(r"[A-Za-z_]+$", s[0]) and s[0] != "AGGREGATE":
            try:
                out[s[0]] = float(s[8])
            except ValueError:
                pass
    return out


M = np.array([[per_domain(p).get(d, np.nan) for d in DOMS] for p in PATS])
centered = M - M.mean(axis=0, keepdims=True)  # deviation from each task's mean

fig, ax = plt.subplots(figsize=(11, 5))
vmax = np.nanmax(np.abs(centered))
im = ax.imshow(centered, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(DOMS)))
ax.set_xticklabels(DOMS, rotation=25, ha="right", fontsize=8)
ax.set_yticks(range(len(PATS)))
ax.set_yticklabels(PATS, fontfamily="monospace")
for i in range(len(PATS)):
    for j in range(len(DOMS)):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7)
ax.set_title("Phase M3 — per-task τ by SWA/full layer pattern (Muon, window 1024, depth 5)\n"
             "color = deviation from each task's mean → differences are within seed noise; "
             "no pattern wins consistently")
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("τ − task mean")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "20_layer_mix_per_task.png"), bbox_inches="tight")
print("wrote 20_layer_mix_per_task.png")
