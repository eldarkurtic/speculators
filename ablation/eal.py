"""Expected Accepted Length (EAL) — the headline speculative-decoding metric.

Given per-position acceptance accuracies a_1, a_2, ... within a draft block, the expected number of
accepted draft tokens (greedy verification) is:

    EAL = sum_k  prod_{i<=k} a_i

i.e. a token at position k is accepted only if all earlier positions were also correct. Higher EAL =
more tokens accepted per verifier step = bigger speedup.

Usage:
  venv_spec/bin/python ablation/eal.py --accs 0.92 0.78 0.61 0.45      # compute from numbers
  venv_spec/bin/python ablation/eal.py --scan ablation/logs           # best-effort scan of run logs
(wandb is the source of truth; the log scan is a convenience.)
"""

import argparse
import re
from pathlib import Path


def compute_eal(per_pos_accs: list[float]) -> float:
    eal, cum = 0.0, 1.0
    for a in per_pos_accs:
        cum *= a
        eal += cum
    return eal


def _scan_log(path: Path) -> dict[int, float]:
    """Return the last-seen {position: acc} from a run log (best effort)."""
    text = path.read_text(errors="ignore")
    accs: dict[int, float] = {}
    for m in re.finditer(r"position\s+(\d+)\s+acc['\"]?\s*[:=]?\s*([0-9]*\.?[0-9]+)", text):
        accs[int(m.group(1))] = float(m.group(2))
    return accs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accs", type=float, nargs="+", help="per-position accuracies a_1 a_2 ...")
    ap.add_argument("--scan", type=str, help="dir of *.log files to scan")
    args = ap.parse_args()

    if args.accs:
        print(f"EAL = {compute_eal(args.accs):.4f}  (from {len(args.accs)} positions)")
    if args.scan:
        rows = []
        for log in sorted(Path(args.scan).glob("*.log")):
            accs = _scan_log(log)
            if accs:
                ordered = [accs[k] for k in sorted(accs)]
                rows.append((log.stem, compute_eal(ordered), ordered))
        rows.sort(key=lambda r: -r[1])
        print(f"{'run':40s} {'EAL':>7s}  per-position-acc")
        for name, eal, ordered in rows:
            print(f"{name:40s} {eal:7.4f}  {[round(a, 3) for a in ordered]}")


if __name__ == "__main__":
    main()
