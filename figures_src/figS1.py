"""Supplementary Figure S1 from assembly_info.json of the S10 run: nearest-neighbour Jaccard similarity density on each
origin's training union (kernel density as recorded by the engine), with the registered antimode and the analysed
near-duplicate thresholds. Okabe-Ito colours plus line styles, so the figure reads in greyscale."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib import font_manager                    # noqa: E402

_NAMES = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((n for n in ("Arial", "Helvetica", "Liberation Sans") if n in _NAMES), "DejaVu Sans")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [FONT], "font.size": 7.5, "axes.linewidth": 1.0,
                     "lines.linewidth": 1.2})

info = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out = Path(sys.argv[2])
fig, ax = plt.subplots(figsize=(6.3, 2.6))
styles = {"1": ("#0072B2", "-"), "2": ("#D55E00", "--"), "3": ("#009E73", ":")}
for o, (col, ls) in styles.items():
    prof = info["origins"][o]["nn_profile"]
    ax.plot(prof["kde_on_grid"]["at"], prof["kde_on_grid"]["density"], color=col, linestyle=ls,
            label=f"Origin {o} (n = {prof['n']:,} traces; registered antimode {info['origins'][o]['tau_J_registered']:.3f})")
regs = [info["origins"][o]["tau_J_registered"] for o in styles]
ax.axvline(sum(regs) / len(regs), color="black", linestyle=(0, (1, 1.5)), linewidth=0.9,
           label=f"Registered antimode thresholds ({min(regs):.3f}\u2013{max(regs):.3f})")
ax.axvline(0.7, color="#555555", linestyle="-.", linewidth=0.8, label="Analysed thresholds 0.7 and 0.9")
ax.axvline(0.9, color="#555555", linestyle="-.", linewidth=0.8)
ax.axvline(0.8, color="#555555", linestyle="-", linewidth=1.6, label="Primary threshold 0.8")
ax.set_xlim(0, 1)
ax.set_ylim(bottom=0)
ax.set_xlabel("Largest Jaccard index to another trace in the training union")
ax.set_ylabel("Density")
ax.legend(frameon=False, fontsize=6, loc="upper left")
fig.tight_layout()
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
print("font:", FONT)
