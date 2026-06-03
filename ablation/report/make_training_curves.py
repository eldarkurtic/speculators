"""Training-dynamics plots for the Muon-regime ablations.

Final-number bar charts hide *how* a config gets there — convergence speed, overfit,
whether a loss bootstraps, when an ordering emerges. This script plots how the key
metrics EVOLVE over epochs for the compared configurations (Muon LR sweep, loss study,
window sweep) by parsing the per-epoch val metrics out of the run logs.

Run:  .venv/bin/python ablation/report/make_training_curves.py
Writes plots 9-12 into ablation/report/plots/ (companion to make_report_plots.py).
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


def parse(name):
    """Per-epoch val trajectories + final-epoch per-position acc from a run log."""
    f = LOGS / f"{name}.log"
    if not f.exists():
        return None
    txt = f.read_text(errors="ignore")
    def g(k):
        return [float(x) for x in re.findall(rf"val/{k}_epoch=([0-9.]+)", txt)]
    out = {"loss": g("loss"), "acc": g("full_acc"), "eal": g("eal"), "pos": {}}
    # rich logging wraps lines, so allow whitespace/newline between label and value
    for m in re.finditer(r"position (\d)\s+acc_epoch=([0-9.]+)", txt):
        out["pos"][int(m.group(1))] = float(m.group(2))  # last (final) epoch wins
    return out


def curve(ax, name, key, **kw):
    d = parse(name)
    if d and d.get(key):
        ax.plot(range(1, len(d[key]) + 1), d[key], **kw)


def mean_band(ax, names, color, label):
    """Plot seed-mean EAL vs epoch with a ±std shaded band."""
    arrs = [parse(n)["eal"] for n in names if parse(n) and parse(n)["eal"]]
    if not arrs:
        return
    length = min(len(a) for a in arrs)
    m = np.array([a[:length] for a in arrs])
    x = range(1, length + 1)
    ax.plot(x, m.mean(0), "o-", color=color, label=label, ms=4)
    if len(arrs) > 1:
        ax.fill_between(x, m.mean(0) - m.std(0), m.mean(0) + m.std(0),
                        color=color, alpha=0.18)


# --- Fig 9: M0 Muon LR sweep — EAL & val-loss vs epoch (Muon dominates AdamW) ---
muon_lrs = [("muon-lr25e4", "0.0025"), ("muon-lr5e3", "0.005"), ("muon-lr1e2", "0.01"),
            ("muon-lr2e2", "0.02"), ("muon-lr35e3", "0.035"), ("muon-lr5e2", "0.05"),
            ("muon-lr7e2", "0.07")]
cmap = plt.cm.viridis(np.linspace(0.12, 0.9, len(muon_lrs)))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
for (nm, lab), c in zip(muon_lrs, cmap):
    curve(a1, nm, "eal", color=c, marker="o", ms=3, label=f"Muon {lab}")
    curve(a2, nm, "loss", color=c, marker="o", ms=3)
curve(a1, "adamw-ref-5e", "eal", color=RED, ls="--", lw=2.2, marker="s", ms=5,
      label="AdamW 6e-4")
curve(a2, "adamw-ref-5e", "loss", color=RED, ls="--", lw=2.2, marker="s", ms=5)
a1.set(xlabel="epoch", ylabel="val EAL",
       title="M0 — EAL vs epoch: Muon LR sweep vs AdamW")
a2.set(xlabel="epoch", ylabel="val loss", title="M0 — val loss vs epoch")
a1.legend(fontsize=7, ncol=2, loc="lower right")
fig.suptitle("Muon converges faster AND higher than AdamW (same arch/data)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "9_muon_lr_curves.png", bbox_inches="tight")
plt.close(fig)

# --- Fig 10: M1 loss study — standalone bootstrapping + two-stage schedule dip ---
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
for nm, lab, c in [("muon-lr1e2", "ce", BLUE), ("ls-kl", "kl", GREEN),
                   ("ls-jsd", "jsd (fixed)", PURPLE), ("ls-revkl", "reverse_kl", ORANGE),
                   ("ls-lk", "lk", RED)]:
    curve(a1, nm, "eal", color=c, marker="o", ms=3, label=lab)
a1.set(xlabel="epoch", ylabel="val EAL",
       title="M1 — standalone losses: all bootstrap under Muon\n(jsd no longer flat at ~0.05)")
a1.legend(fontsize=8)
start = parse("muon-lr1e2")
start_eal = max(start["eal"]) if start and start["eal"] else None
for nm, lab, c in [("ts-ce", "ce (ctrl)", BLUE), ("ts-kl", "kl", GREEN),
                   ("ts-jsd", "jsd", PURPLE), ("ts-lk", "lk", RED)]:
    curve(a2, nm, "eal", color=c, marker="o", ms=3, label=lab)
if start_eal:
    a2.axhline(start_eal, color=GREY, ls=":", lw=1.6, label=f"CE start ({start_eal:.3f})")
a2.set(xlabel="epoch", ylabel="val EAL",
       title="M1 — two-stage CE→soft ft: schedule restart\ndips the converged model below its start")
a2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "10_loss_curves.png", bbox_inches="tight")
plt.close(fig)

# --- Fig 11: M2a window sweep — EAL vs epoch, seed-mean ±std band ---
fig, ax = plt.subplots(figsize=(7.8, 4.6))
for lab, names, c in [("window 128", ["w128-f6144-s42", "w128-f6144-s123"], BLUE),
                      ("window 256", ["w256-f6144-s42", "w256-f6144-s123"], GREEN),
                      ("window 1024", ["w1024-f6144-s42", "w1024-f6144-s123"], RED)]:
    mean_band(ax, names, c, lab)
ax.set(xlabel="epoch", ylabel="val EAL  (mean ± std, 2 seeds)",
       title="M2a — window sweep: larger window better at every epoch\n(reverses the old AdamW/2ep finding)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "11_window_curves.png", bbox_inches="tight")
plt.close(fig)

# --- Fig 12: final per-position acceptance — the deployment-facing metric ---
fig, ax = plt.subplots(figsize=(7.8, 4.4))
for nm, lab, c in [("adamw-ref-5e", "AdamW CE w128", GREY),
                   ("muon-lr1e2", "Muon CE w128", BLUE),
                   ("ls-kl", "Muon KL w128", GREEN),
                   ("w1024-f6144-s42", "Muon KL w1024 (best)", RED)]:
    d = parse(nm)
    if d and d["pos"]:
        ks = sorted(d["pos"])
        ax.plot(ks, [d["pos"][k] for k in ks], "o-", color=c, label=lab)
ax.set(xlabel="draft position k", ylabel="acceptance prob (final epoch)",
       title="Per-position acceptance — best Muon config vs AdamW baseline")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "12_per_position_muon.png", bbox_inches="tight")
plt.close(fig)

# --- Fig 13: window sweep — final EAL vs window size (inverted-U; peak ~2048) ---
def best_eal(name):
    d = parse(name)
    return max(d["eal"]) if d and d["eal"] else None


wins = [("128", ["w128-f6144-s42", "w128-f6144-s123"]),
        ("256", ["w256-f6144-s42", "w256-f6144-s123"]),
        ("1024", ["w1024-f6144-s42", "w1024-f6144-s123"]),
        ("1536", ["w1536-s42", "w1536-s123"]),
        ("2048", ["w2048-s42", "w2048-s123"]),
        ("4096", ["w4096-s42", "w4096-s123"]),
        ("full", ["wfull-s42", "wfull-s123"])]
xs, labs, means, stds = [], [], [], []
for i, (lab, names) in enumerate(wins):
    vals = [best_eal(n) for n in names if best_eal(n) is not None]
    if vals:
        xs.append(i)
        labs.append(lab)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
if means:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(xs, means, yerr=stds, fmt="o-", color=BLUE, capsize=4, ms=6, lw=1.8)
    pk = int(np.argmax(means))
    ax.annotate(f"peak: window {labs[pk]} = {means[pk]:.3f}", (xs[pk], means[pk]),
                textcoords="offset points", xytext=(0, 14), ha="center",
                fontsize=9, color=RED, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(labs)
    ax.set(xlabel="sliding window (tokens)", ylabel="val EAL  (mean ± std, 2 seeds)",
           title="Window sweep — broad peak ~1536-2048, then declines toward full attention")
    fig.tight_layout()
    fig.savefig(OUT / "13_window_size_curve.png", bbox_inches="tight")
    plt.close(fig)

# --- Fig 14: depth sweep — final EAL vs draft depth (deeper better; cost tradeoff) ---
depths = [("2", ["d2-s42", "d2-s123"]), ("3", ["d3-s42", "d3-s123"]),
          ("4", ["d4-s42", "d4-s123"]), ("5", ["w2048-s42", "w2048-s123"]),
          ("7", ["d7-s42", "d7-s123"])]
dx, dm, ds = [], [], []
for lab, names in depths:
    vals = [best_eal(n) for n in names if best_eal(n) is not None]
    if vals:
        dx.append(int(lab))
        dm.append(float(np.mean(vals)))
        ds.append(float(np.std(vals)))
if dm:
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.errorbar(dx, dm, yerr=ds, fmt="o-", color=GREEN, capsize=4, ms=6, lw=1.8)
    ax.set_xticks(dx)
    ax.set(xlabel="draft depth (# transformer layers)",
           ylabel="val EAL  (mean ± std, 2 seeds)",
           title="Depth sweep @ window 2048 — deeper better, monotonic, still climbing at 7\n"
                 "(EAL only; deeper = higher draft cost — net speedup needs the deploy benchmark)")
    fig.tight_layout()
    fig.savefig(OUT / "14_depth_curve.png", bbox_inches="tight")
    plt.close(fig)

print("Wrote Muon-regime training-dynamics plots 9-14 to", OUT)
