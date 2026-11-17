"""S9 diagnostic figures (Supplementary Information). Okabe-Ito colours with hatching, so panels read in greyscale.

Usage: python plots_s9.py --root WORK   (reads outputs/*.json and work/nn_similarities_*.npy; writes PDF + 300 dpi PNG)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                    # noqa: E402
from matplotlib import font_manager                   # noqa: E402

OKABE = {"B": "#0072B2", "F": "#D55E00", "P": "#009E73", "grey": "#999999", "y": "#E69F00"}
HATCH = {"B": "", "F": "//", "P": ".."}
FONT = "Arial" if any("Arial" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": FONT, "font.size": 8, "axes.linewidth": 1.0, "lines.linewidth": 1.2})


def panel(ax, letter):
    ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    root = Path(ap.parse_args().root)
    rep = json.loads((root / "outputs" / "s9_selftest_report.json").read_text())
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.2))
    a = axs[0, 0]                                     # a: token class shares, primary versus boundary reading
    prim, alt = rep["tokens"]["class_share_primary"], rep["tokens"]["class_share_boundary_alt"]
    x = np.arange(3)
    for k, c in enumerate("BFP"):
        a.bar(k - 0.18, 100 * prim[c], 0.34, color=OKABE[c], hatch=HATCH[c], edgecolor="black", linewidth=0.6)
        a.bar(k + 0.18, 100 * alt[c], 0.34, color="white", hatch=HATCH[c], edgecolor=OKABE[c], linewidth=1.0)
    for k, c in enumerate("BFP"):
        a.text(k - 0.18, 100 * prim[c] + 2, f"{100 * prim[c]:.1f}", ha="center", fontsize=6.5)
    a.set_xticks(x, ["Behaviour (B)", "Footprint (F)", "Probing (P)"])
    a.set_ylabel("Share of collapsed tokens (%)")
    a.set_ylim(0, 100)
    a.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="grey", ec="black"), plt.Rectangle((0, 0), 1, 1, fc="white",
                      ec="grey")], labels=["Primary reading", "Boundary rules reversed"], frameon=False, fontsize=7)
    panel(a, "a")
    b = axs[0, 1]                                     # b: nearest-neighbour similarity and tau_J
    nn = np.load(root / "work" / "nn_similarities_T.npy")
    b.hist(nn, bins=np.linspace(0, 1, 41), color=OKABE["B"], edgecolor="black", linewidth=0.5)
    tau = rep["dedup"]["tau_J"]
    b.axvline(tau, color=OKABE["F"], linestyle="--", linewidth=1.2)
    b.text(tau, b.get_ylim()[1] * 0.92, f" tau_J = {tau:.2f} ({rep['dedup']['rule']['rule'].split(' ')[0]})",
           color="black", fontsize=7)
    b.set_xlabel("Nearest-neighbour Jaccard similarity (API 3-grams)")
    b.set_ylabel("Traces")
    b.yaxis.get_major_locator().set_params(integer=True)
    panel(b, "b")
    c = axs[1, 0]                                     # c: rules that fired
    rules = sorted(rep["tokens"]["rule_firing_share"].items(), key=lambda kv: -kv[1])[:10]
    c.barh([r for r, _ in rules][::-1], [100 * v for _, v in rules][::-1], color=OKABE["grey"], edgecolor="black",
           linewidth=0.5)
    c.set_xlabel("Share of collapsed tokens (%)")
    c.set_xscale("log")
    panel(c, "c")
    d = axs[1, 1]                                     # d: learner cost at the sequence cap
    t = rep["learners"]["seconds_per_trace_fwd_bwd_at_L"]
    ks = [k for k in ("transformer", "bilstm", "cnn") if k in t]
    d.bar(ks, [1000 * t[k] for k in ks], color=[OKABE["B"], OKABE["y"], OKABE["P"]], edgecolor="black",
          hatch=["", "//", ".."], linewidth=0.6)
    d.set_ylabel("Milliseconds per trace")
    d.set_title(f"Forward and backward pass, L = {rep['tokens']['L']}, 1 CPU", fontsize=7.5)
    d.set_xticks(range(len(ks)), ["Transformer", "BiLSTM", "1D-CNN"][:len(ks)])
    panel(d, "d")
    fig.tight_layout()
    out = root / "outputs" / "s9_diagnostics"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"), "font", FONT)


if __name__ == "__main__":
    main()
