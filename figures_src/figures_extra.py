"""Additional manuscript figures drawn from the measured results.json and assembly_info.json of the S10 run:
audit design (estimands), evidence flow and sample sizes, learner comparison, temporal stability and referral, and
robustness to the near-duplicate threshold. Okabe-Ito colours plus hatching or line styles (greyscale-safe); the
typeface of Figs 1-3; vector PDF and 600 dpi PNG. Error bars are 95% cluster-bootstrap percentile intervals."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                     # noqa: E402
from matplotlib import font_manager                    # noqa: E402
from matplotlib.patches import FancyBboxPatch          # noqa: E402

_NAMES = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((n for n in ("Arial", "Helvetica", "Liberation Sans") if n in _NAMES), "DejaVu Sans")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [FONT], "font.size": 7.5, "axes.linewidth": 1.0,
                     "lines.linewidth": 1.2, "errorbar.capsize": 2, "mathtext.fontset": "dejavusans"})
OK = {"blue": "#0072B2", "verm": "#D55E00", "green": "#009E73", "orange": "#E69F00", "sky": "#56B4E9",
      "grey": "#999999", "pink": "#CC79A7", "light": "#F2F2F2"}
LEARN = {"transformer": "Transformer", "bilstm": "BiLSTM", "cnn": "1D-CNN", "lgbm": "n-gram LightGBM"}
ORDER = ["transformer", "bilstm", "cnn", "lgbm"]
NAMES = {"I1": "MELD-DS-448", "I2": "WinMET"}


def save(fig, path: Path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def panel(ax, letter, x=-0.12):
    ax.text(x, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


def box(ax, x, y, w, h, text, fc="white", ec="black", fs=6.8, bold_first=False, lw=0.9, ha="center"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=fc, ec=ec, lw=lw))
    lines = text.split("\n")
    if bold_first:
        ax.text(x if ha == "center" else x - w / 2 + 1.2, y + h / 2 - 2.2, lines[0], ha=ha, va="top", fontsize=fs,
                fontweight="bold")
        ax.text(x if ha == "center" else x - w / 2 + 1.2, y + h / 2 - 2.2 - fs * 0.62, "\n".join(lines[1:]), ha=ha,
                va="top", fontsize=fs, linespacing=1.25)
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, text=None, fs=6):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black",
                                                                    shrinkA=0, shrinkB=0, mutation_scale=7))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 1.2, text, ha="center", va="bottom", fontsize=fs, style="italic")


# ------------------------------------------------------------------------------------------------------- design
def fig_design(out: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 100), ax.set_ylim(0, 64), ax.axis("off")
    ax.text(14, 62, "Models trained on installation I", ha="center", fontsize=7.5, fontweight="bold")
    ax.text(47, 62, "Test data", ha="center", fontsize=7.5, fontweight="bold")
    ax.text(82, 62, "Estimands and registered tests", ha="center", fontsize=7.5, fontweight="bold")
    models = [(51, "Length-only probe $f^{\\mathrm{len}}$\nsees trace length only", OK["light"]),
              (38, "Artefact-only probe $f^{\\mathrm{art}}$\nbehaviour tokens masked", "#FBE3D6"),
              (25, "Base model $f^{\\mathrm{base}}$\nall tokens", "#D9EAF5"),
              (10, "F-masked model $f^{\\mathrm{mit}}$\nfootprint tokens masked,\nartefact dropout", "#D9F0E8")]
    for y, t, c in models:
        box(ax, 14, y, 25, 9.5 if y != 10 else 11.5, t, fc=c)
    box(ax, 47, 44, 25, 14, "Own installation, $\\mathcal{D}_I$\nmasks: none, $M_F$, $M_P$, $M_{FP}$,\n"
        "$M_{\\mathrm{rand}}$ (matched to $M_{FP}$), $M_B$", fc="white")
    box(ax, 47, 17, 25, 12, "Other installation, $\\mathcal{D}_{I'}$\nsame families, other machines\nno mask; $M_F$ for $f^{\\mathrm{mit}}$",
        fc="white")
    est = [(53, "$\\psi_1 = F1(f^{\\mathrm{art}}; M_B(\\mathcal{D}_I)) - F1(f^{\\mathrm{len}}; \\mathcal{D}_I)$\n"
                "H1 supported if $\\psi_1 > 3$ in both installations"),
           (40, "$\\psi_2 = F1(f^{\\mathrm{base}}; M_{\\mathrm{rand}}(\\mathcal{D}_I)) - F1(f^{\\mathrm{base}}; M_{FP}(\\mathcal{D}_I))$\n"
                "H2 supported if $\\psi_2 > 3$; rejected if within $\\pm 3$"),
           (27, "$\\psi_3 = F1(f^{\\mathrm{base}}; \\mathcal{D}_I) - F1(f^{\\mathrm{base}}; \\mathcal{D}_{I'})$\n"
                "H3 supported if $\\psi_3 > 3$ in both directions"),
           (11, "$\\rho = \\Sigma\\,[F1^{\\mathrm{mit}}_{\\mathrm{cross}} - F1^{\\mathrm{base}}_{\\mathrm{cross}}]\\, /\\, \\Sigma\\,\\psi_3$\n"
                "$\\psi_{4b} = F1(f^{\\mathrm{mit}}; M_F(\\mathcal{D}_I)) - F1(f^{\\mathrm{base}}; \\mathcal{D}_I)$\n"
                "H4 supported if $\\rho > 0.10$ and $\\psi_{4b} \\geq -3$")]
    for y, t in est:
        box(ax, 82, y, 34, 10.5 if y != 11 else 13.5, t, fc="white", fs=6.3)
    for y in (51, 38, 25):
        arrow(ax, 26.6, y, 34.4, 44 + (y - 38) * 0.25)
    arrow(ax, 26.6, 25, 34.4, 18)
    arrow(ax, 26.6, 10, 34.4, 15.5)
    arrow(ax, 26.6, 12, 34.4, 41)
    for y in (53, 40):
        arrow(ax, 59.6, 46 if y == 53 else 43, 64.9, y)
    arrow(ax, 59.6, 41, 64.9, 28.5)
    arrow(ax, 59.6, 17.5, 64.9, 26)
    arrow(ax, 59.6, 15, 64.9, 12)
    arrow(ax, 59.6, 39.5, 64.9, 14.5)
    ax.text(50, 1.0, "Macro-F1 is family-balanced, averaged over five seeds (and three origins for MELD-DS-448). "
            "\nOne-sided tests at 0.025 per claim; installations combined by the intersection\u2013union rule; "
            "Holm adjustment over H1\u2013H4.", ha="center", va="bottom", fontsize=6)
    save(fig, out / "Fig_design")


# --------------------------------------------------------------------------------------------------------- flow
def fig_flow(a: dict, out: Path):
    c, r = a["cells"], a["records"]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.set_xlim(0, 100), ax.set_ylim(0, 100), ax.axis("off")
    box(ax, 27, 93, 44, 10, f"MELD-DS-448 (installation I1)\n{r['I1']['read']:,} CAPEv2 reports released\n"
        f"(Hugging Face; SHA-256 checked)", fc="#D9EAF5")
    box(ax, 73, 93, 44, 10, f"WinMET (installation I2)\n{r['I2']['read']:,} CAPEv2 traces released\n"
        f"(Zenodo; MD5 checked)", fc="#FBE3D6")
    box(ax, 27, 80.5, 44, 8, f"{r['I1']['retained']:,} reports retained\n"
        f"({r['I1']['read'] - r['I1']['retained']} removed by the MALVADA curation criteria)")
    box(ax, 73, 80.5, 44, 8, f"{r['I2']['retained']:,} traces retained\n(none removed by the curation criteria)")
    box(ax, 50, 68, 90, 8.5, "Sealed in the hash-chained evidence ledger (130,758 entries), harmonised to the common hook "
        f"set and tokenised;\n{a['lexicon']['token_types_seen']:,} token types tagged as behaviour, footprint or probing "
        "by the frozen lexicon", fc=OK["light"])
    months = {"1": "March\u2013April", "2": "May\u2013June", "3": "July\u2013August"}
    for k, x in zip("123", (17, 50, 83)):
        m, w = c[f"I1_o{k}"], c[f"I2_o{k}"]
        fam = len(a["origins"][k]["families"])
        box(ax, x, 50, 31, 17, f"Origin {k}: {fam} shared families\nMELD-DS-448 test window:\n{months[k]} 2025\n"
            f"MELD-DS-448: {m['n_train']:,} / {m['n_val']:,} / {m['n_own_all']:,}\n"
            f"WinMET: {w['n_train']:,} / {w['n_val']:,} / {w['n_own_all']:,}\n(training / validation / test)", fs=6.3)
        km, kw = m["kept_own"]["0.8"], w["kept_own"]["0.8"]
        box(ax, x, 27.5, 31, 13, f"Near-duplicate control\n(Jaccard < 0.8; binary not in training)\n"
            f"MELD-DS-448 kept {km:,} of {m['n_own_all']:,} ({100 * km / m['n_own_all']:.1f}%)\n"
            f"WinMET kept {kw:,} of {w['n_own_all']:,} ({100 * kw / w['n_own_all']:.1f}%)", fs=6.3)
        arrow(ax, x, 63.6, x, 58.6)
        arrow(ax, x, 41.4, x, 34.1)
        arrow(ax, x, 20.9, x if x == 50 else (x + 50) / 2, 14.6)
    for x in (27, 73):
        arrow(ax, x, 88.4, x, 84.6)
        arrow(ax, x, 76.4, x, 72.3)
    tm = sum(c[f"I1_o{k}"]["kept_own"]["0.8"] for k in "123")
    tw = sum(c[f"I2_o{k}"]["kept_own"]["0.8"] for k in "123")
    box(ax, 50, 9.5, 90, 9, f"Analysed: {tm:,} MELD-DS-448 and {tw:,} WinMET test traces\nfour learners \u00d7 five seeds "
        "\u00d7 three origins, within and across installations; 9,999 cluster-bootstrap replicates", fc="#D9F0E8")
    save(fig, out / "Fig_flow")


# ----------------------------------------------------------------------------------------------------- learners
def fig_learners(d: dict, out: Path):
    F = {(r["learner"], r["role"], r["where"], r["mask"], r["inst"]): (r["f1"], r["ci_lo"], r["ci_hi"]) for r in d["f1"]}
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.45, wspace=0.25)
    conds = [(("base", "own", "none"), "Within", OK["blue"], ""), (("base", "other", "none"), "Cross", OK["verm"], "//"),
             (("mit", "other", "F"), "Cross, F-masked training", OK["green"], "..")]
    for j, inst in enumerate(("I1", "I2")):
        ax = fig.add_subplot(gs[0, j])
        for i, (key, lab, col, hat) in enumerate(conds):
            for k, ln in enumerate(ORDER):
                v, lo, hi = F[(ln,) + key + (inst,)]
                ax.bar(k + (i - 1) * 0.27, v, 0.25, color=col, hatch=hat, edgecolor="black", linewidth=0.5,
                       yerr=[[v - lo], [hi - v]], label=lab if k == 0 else None)
        ax.set_xticks(range(4), [LEARN[ln].replace("n-gram ", "n-gram\n") for ln in ORDER], fontsize=6.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Family-balanced macro-F1 (points)")
        other = NAMES["I2" if inst == "I1" else "I1"]
        ax.set_title(f"Trained on {NAMES[inst]}; cross = tested on {other}", fontsize=7)
        if j == 0:
            ax.legend(frameon=False, fontsize=6, loc="upper right")
        panel(ax, "ab"[j])
    ax = fig.add_subplot(gs[1, :])
    masks = [("none", "None"), ("F", "F"), ("P", "P"), ("FP", "F and P"), ("rand_FP", "Matched random")]
    rows = [(ln, inst) for ln in ORDER for inst in ("I1", "I2")]
    M = np.array([[F[(ln, "base", "own", m, inst)][0] for m, _ in masks] for ln, inst in rows])
    im = ax.imshow(M.T, cmap="cividis", aspect="auto", vmin=30, vmax=75)
    for x in range(M.shape[0]):
        for y in range(M.shape[1]):
            ax.text(x, y, f"{M[x, y]:.1f}", ha="center", va="center", fontsize=6.3,
                    color="white" if M[x, y] < 52 else "black")
    ax.set_xticks(range(len(rows)), [LEARN[ln].replace("n-gram ", "n-gram\n") + f"\n{NAMES[inst]}" for ln, inst in rows],
                  fontsize=6.0)
    ax.set_yticks(range(len(masks)), [lab for _, lab in masks])
    ax.set_ylabel("Tokens masked at test")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Macro-F1 (points)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    panel(ax, "c", x=-0.06)
    save(fig, out / "Fig_learners")


# ----------------------------------------------------------------------------------------------------- temporal
def fig_temporal(d: dict, out: Path):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.3})
    ax = axs[0]
    styles = {"transformer": (OK["blue"], "-", "o"), "bilstm": (OK["orange"], "--", "s"), "cnn": (OK["green"], ":", "^")}
    months = None
    for ln, (col, ls, mk) in styles.items():
        p = d["checks"][f"I1_o1|{ln}"]["P9"]
        months = sorted(p["series"])
        ax.plot(range(len(months)), [p["series"][m] for m in months], color=col, linestyle=ls, marker=mk, ms=3.5,
                label=f"{LEARN[ln]} (Kendall \u03c4 = {p['tau']:.2f}, P = {p['p_one_sided']:.3f})".replace("-", "\u2212"))
    n = d["checks"]["I1_o1|transformer"]["P9"]["n_by_month"]
    ax.set_xticks(range(len(months)), [f"{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][int(m[5:]) - 1]}\nn = {n[m]}"
                                       for m in months], fontsize=6.3)
    ax.set_ylim(0, 60)
    ax.set_ylabel("Family-balanced macro-F1 (points)")
    ax.set_xlabel("MELD-DS-448 test month, 2025 (origin 1)")
    ax.legend(frameon=False, fontsize=5.8, loc="lower left")
    panel(ax, "a", x=-0.16)
    ax = axs[1]
    cells = ["I1_o1", "I1_o2", "I1_o3", "I2_o1", "I2_o2", "I2_o3"]
    for i, cell in enumerate(cells):
        ck = d["checks"][f"{cell}|transformer"]
        r, nexp = ck["referral"], ck["n_explained"]
        ax.bar(i - 0.2, 100 * r["err_referred"], 0.38, color=OK["verm"], hatch="//", edgecolor="black", linewidth=0.5,
               label="Referred traces" if i == 0 else None)
        ax.bar(i + 0.2, 100 * r["err_accepted"], 0.38, color=OK["sky"], edgecolor="black", linewidth=0.5,
               label="Accepted traces" if i == 0 else None)
        top = 100 * max(r["err_referred"], r["err_accepted"])
        ax.text(i, top + 2.5, f"n = {round(r['rate'] * nexp)}\nP = {r['P6_p']:.3f}", ha="center", va="bottom", fontsize=5.6)
    ax.set_xticks(range(len(cells)), [f"Origin {c[-1]}" for c in cells], fontsize=6.2)
    for x, inst in ((1, "I1"), (4, "I2")):
        ax.text(x, -0.17, NAMES[inst], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.6)
    ax.axvline(2.5, color="#777777", linewidth=0.6, linestyle=":")
    ax.set_ylim(0, 85)
    ax.set_ylabel("Misattributed traces (%)")
    ax.legend(frameon=False, fontsize=6, loc="upper left", ncol=2)
    panel(ax, "b", x=-0.14)
    save(fig, out / "Fig_temporal")


# ---------------------------------------------------------------------------------------------------- threshold
def fig_threshold(d: dict, a: dict, out: Path):
    taus = ["registered", "0.7", "0.8", "0.9"]
    labels = ["Registered\n(0.528\u20130.535)", "0.7", "0.8\n(primary)", "0.9"]

    def rows(t):
        dec = d["decisions"] if t == "0.8" else d["sensitivity"][t]["decisions"]
        return {r["component"]: (r["estimate"], r["ci_lo"], r["ci_hi"]) for r in dec["transformer"]["rows"]}
    R = {t: rows(t) for t in taus}
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0), gridspec_kw={"hspace": 0.55, "wspace": 0.3})
    ax = axs[0, 0]
    for j, (inst, col, hat) in enumerate((("I1", OK["blue"], ""), ("I2", OK["verm"], "//"))):
        kept = [sum(a["cells"][f"{inst}_o{k}"]["kept_own"][t] for k in "123") for t in taus]
        ax.bar(np.arange(4) + (j - 0.5) * 0.38, kept, 0.36, color=col, hatch=hat, edgecolor="black", linewidth=0.5,
               label=NAMES[inst])
        for x, v in zip(np.arange(4) + (j - 0.5) * 0.38, kept):
            ax.text(x, v + 40, f"{v:,}", ha="center", va="bottom", fontsize=5.4, rotation=90)
    ax.set_xticks(range(4), labels, fontsize=6.2)
    ax.set_ylim(0, 3300)
    ax.set_ylabel("Test traces analysed")
    ax.set_xlabel("Near-duplicate threshold $\\tau_J$")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    panel(ax, "a", x=-0.18)
    spec = [(axs[0, 1], ("psi3|I1", "psi3|I2"), ("MELD-DS-448 to WinMET", "WinMET to MELD-DS-448"),
             "Transfer gap, $\\psi_3$ (points)", "b"),
            (axs[1, 0], ("psi2|I1", "psi2|I2"), ("MELD-DS-448", "WinMET"),
             "Excess drop over matched\nrandom mask, $\\psi_2$ (points)", "c"),
            (axs[1, 1], ("rho",), ("Pooled over directions",), "Recovered share of the gap, $\\rho$", "d")]
    for ax, comps, labs, ylab, letter in spec:
        for j, (comp, lab) in enumerate(zip(comps, labs)):
            off = (j - (len(comps) - 1) / 2) * 0.18
            v = np.array([R[t][comp] for t in taus])
            col, mk = ((OK["blue"], "o"), (OK["verm"], "s"))[j] if len(comps) > 1 else ("black", "D")
            ax.errorbar(np.arange(4) + off, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=mk, color=col,
                        ms=4, label=lab)
        if letter == "b":
            ax.axhline(3, color="black", linestyle="--", linewidth=0.9)
            ax.set_ylim(0, 50)
        if letter == "c":
            ax.axhspan(-3, 3, color="#E5E5E5", zorder=0)
            ax.axhline(3, color="black", linestyle="--", linewidth=0.9)
        if letter == "d":
            ax.axhline(0.10, color="black", linestyle="--", linewidth=0.9)
            ax.axhline(0, color="#777777", linewidth=0.6)
        ax.set_xticks(range(4), labels, fontsize=6.2)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylabel(ylab)
        ax.set_xlabel("Near-duplicate threshold $\\tau_J$")
        if len(comps) > 1:
            ax.legend(frameon=False, fontsize=6, loc="upper left" if letter == "b" else "lower left")
        panel(ax, letter, x=-0.18)
    save(fig, out / "Fig_threshold")


if __name__ == "__main__":
    d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    a = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    out = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    fig_design(out)
    fig_flow(a, out)
    fig_learners(d, out)
    fig_temporal(d, out)
    fig_threshold(d, a, out)
    print("font:", FONT)
