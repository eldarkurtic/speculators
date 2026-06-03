"""Parse ablation run logs and generate all plots for the DFlash project report."""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS = Path(__file__).resolve().parents[1] / "logs"
OUT = Path(__file__).resolve().parent / "plots"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, GREEN, RED, GREY, ORANGE, PURPLE = (
    "#2b6cb0", "#2f855a", "#c53030", "#a0aec0", "#dd6b20", "#805ad5")
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def parse(name):
    f = LOGS / f"{name}.log"
    if not f.exists():
        return None
    txt = f.read_text(errors="ignore")
    g = lambda k: [float(x) for x in re.findall(rf"val/{k}_epoch=([0-9.]+)", txt)]
    out = {"loss": g("loss"), "acc": g("full_acc"), "eal": g("eal"), "pos": {}}
    # rich logging wraps lines, so allow whitespace/newline between label and value
    for m in re.finditer(r"position (\d)\s+acc_epoch=([0-9.]+)", txt):
        out["pos"][int(m.group(1))] = float(m.group(2))  # last (final epoch) wins
    return out


def best_eal(name):
    d = parse(name)
    return max(d["eal"]) if d and d["eal"] else float("nan")


def barh(ax, labels, vals, colors, title, xlabel, ref=None, reflabel=None):
    y = list(range(len(labels)))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    if ref is not None:
        ax.axvline(ref, color=RED, ls="--", lw=1)
        ax.text(ref, -0.3, f" {reflabel}", color=RED, fontsize=8)


# 1: production ladder
fig, ax = plt.subplots(figsize=(8, 3.4))
ladder = [("Baseline\n(5 ep)", 0.960, GREY),
          ("Best arch\n(swa128-wh, 5 ep)", 1.072, BLUE),
          ("Best arch\n(15 ep, CE)", 1.262, GREEN),
          ("+ two-stage\n(jsd-ft, +5 ep)", 1.314, ORANGE)]
vals = [x[1] for x in ladder]
bars = ax.bar([x[0] for x in ladder], vals, color=[x[2] for x in ladder], width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center",
            fontweight="bold")
ax.set_ylabel("Expected Accepted Length (EAL)")
ax.set_title("DFlash Qwen3-8B: EAL improvement across the project", fontweight="bold")
ax.set_ylim(0.85, 1.4)
ax.axhline(0.960, color=GREY, ls=":", lw=1)
ax.annotate("", xy=(3, 1.30), xytext=(0, 0.97),
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5))
ax.text(1.5, 1.34, "+37% overall (0.960 -> 1.314)", color=ORANGE,
        fontweight="bold", ha="center")
fig.tight_layout()
fig.savefig(OUT / "1_ladder.png", bbox_inches="tight")
plt.close(fig)

# 2: screening @2ep
screen = ["swa-256", "swa-512", "swa-1024", "width-half", "fusion-wsum", "depth-2",
          "depth-3", "baseline", "no-decoder-mlp", "width-2x", "depth-7", "depth-1",
          "fusion-mlp", "heads-32-4", "fusion-gated", "heads-16-4"]
sv = [best_eal(s) for s in screen]
sc = [GREEN if v > 0.697 else (GREY if s == "baseline" else RED)
      for s, v in zip(screen, sv)]
fig, ax = plt.subplots(figsize=(7.5, 6))
barh(ax, screen, sv, sc, "Phase 2-3: architecture screening @2 epochs", "EAL",
     ref=0.697, reflabel="baseline 0.697")
fig.tight_layout()
fig.savefig(OUT / "2_screening.png", bbox_inches="tight")
plt.close(fig)

# 3: window sweep
# all @5ep; swa-1024 omitted (only a @2ep screening run exists -> not comparable)
win_alone = [("64", best_eal("ref-swa64")), ("128", best_eal("ref-swa128")),
             ("192", best_eal("ref-swa192")), ("256", best_eal("p4-swa256")),
             ("512", best_eal("p4-swa512")), ("full", best_eal("baseline-5ep"))]
# "full" = p8-fffff (all-full attention + FFN-6144, @5ep); width-half was only @2ep
win_wh = [("64", best_eal("ref-swa64-wh")), ("128", best_eal("ref-swa128-wh")),
          ("256", best_eal("ref-swa256-wh")), ("full", best_eal("p8-fffff"))]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([w[0] for w in win_alone], [w[1] for w in win_alone], "o-", color=BLUE,
        label="sliding window only")
ax.plot([w[0] for w in win_wh], [w[1] for w in win_wh], "s-", color=GREEN,
        label="window + FFN-6144")
ax.set_xlabel("sliding-window size (verifier-context positions)")
ax.set_ylabel("EAL @5ep")
ax.set_title("Phase 5: smaller sliding window is better", fontweight="bold")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "3_window_sweep.png", bbox_inches="tight")
plt.close(fig)

# 4: production curves
fig, ax = plt.subplots(figsize=(7.5, 4.3))
# prod-combo omitted: it resumed from an epoch-6 checkpoint so its epoch axis is offset
for nm, c in [("prod-swa128-wh", GREEN), ("prod-swa256", BLUE),
              ("prod-swa64-wh", PURPLE)]:
    d = parse(nm)
    if d and d["eal"]:
        ax.plot(range(1, len(d["eal"]) + 1), d["eal"], "o-", color=c, label=nm, ms=4)
db = parse("baseline-5ep")
if db and db["eal"]:
    ax.plot(range(1, len(db["eal"]) + 1), db["eal"], "x--", color=GREY,
            label="baseline-5ep")
ax.set_xlabel("epoch")
ax.set_ylabel("val EAL")
ax.set_title("Phase 6: production training curves", fontweight="bold")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "4_production_curves.png", bbox_inches="tight")
plt.close(fig)

# 5: per-position acceptance
fig, ax = plt.subplots(figsize=(7, 4))
for nm, c, lab in [("baseline-5ep", GREY, "baseline (5ep)"),
                   ("prod-swa128-wh", GREEN, "best: swa128-wh (15ep)")]:
    d = parse(nm)
    if d and d["pos"]:
        ks = sorted(d["pos"])
        ax.plot(ks, [d["pos"][k] for k in ks], "o-", color=c, label=lab)
ax.set_xlabel("draft position within block")
ax.set_ylabel("per-position acceptance accuracy")
ax.set_title("Per-position acceptance: baseline vs best", fontweight="bold")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "5_per_position.png", bbox_inches="tight")
plt.close(fig)

# 6: layer mix
mix = ["sssss", "fsssf", "ffsss", "sssff", "sfsfs", "ssfff", "fffss", "fffff"]
mv = [best_eal(f"p8-{m}") for m in mix]
mc = [GREEN if m == "sssss" else (RED if m == "fffff" else BLUE) for m in mix]
fig, ax = plt.subplots(figsize=(7.5, 4))
bars = ax.bar(mix, mv, color=mc)
for b, v in zip(bars, mv):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
ax.set_ylabel("EAL @5ep")
ax.set_xlabel("per-layer pattern (s=sliding window, f=full; layer 0->4)")
ax.set_title("Phase 8: all-SWA beats any SWA/full mix", fontweight="bold")
ax.set_ylim(0.95, 1.09)
fig.tight_layout()
fig.savefig(OUT / "6_layer_mix.png", bbox_inches="tight")
plt.close(fig)

# 7: loss functions standalone
loss = [("ce", "p9-ce"), ("kl", "p9-kl"), ("kl_ce", "p9-kl_ce"),
        ("reverse_kl", "p9-reverse_kl"), ("jsd", "p9-jsd"), ("lk", "p9-lk")]
lv = [best_eal(r) for _, r in loss]
lc = [GREEN if v > 1.0 else RED for v in lv]
fig, ax = plt.subplots(figsize=(7, 3.8))
bars = ax.bar([n for n, _ in loss], lv, color=lc)
for b, v in zip(bars, lv):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
ax.set_ylabel("EAL @5ep")
ax.set_title("Phase 9: loss functions standalone", fontweight="bold")
ax.text(4, 0.45, "fail to bootstrap\n(vanishing gradient)", ha="center", color=RED,
        fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "7_loss_standalone.png", bbox_inches="tight")
plt.close(fig)

# 8: two-stage
ts = [("jsd-ft\n3e-4", "p10-jsdft-3e4", ORANGE), ("lk-ft\n3e-4", "p10-lkft-3e4", GREEN),
      ("lk-ft\n1e-4", "p10-lkft-1e4", GREEN), ("CE-cont\n3e-4", "p10-ceft-3e4", GREY),
      ("CE-cont\n1e-4", "p10-ceft-1e4", GREY)]
tv = [best_eal(r) for _, r, _ in ts]
fig, ax = plt.subplots(figsize=(7.5, 4))
bars = ax.bar([n for n, _, _ in ts], tv, color=[c for _, _, c in ts])
for b, v in zip(bars, tv):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
ax.axhline(1.262, color=BLUE, ls="--", lw=1)
ax.text(4.4, 1.265, "CE start 1.262", color=BLUE, fontsize=8, ha="right")
ax.set_ylabel("val EAL")
ax.set_title("Phase 10: two-stage CE->soft-loss fine-tune", fontweight="bold")
ax.set_ylim(1.0, 1.35)
fig.tight_layout()
fig.savefig(OUT / "8_two_stage.png", bbox_inches="tight")
plt.close(fig)

print("wrote plots:")
for p in sorted(OUT.glob("*.png")):
    print(" ", p.name, p.stat().st_size, "bytes")
