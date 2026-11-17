"""Figs. 4-6 and Table 2 from results.json (S6 display-item plan). Okabe-Ito colours plus hatching, so every panel
reads in greyscale; Arial when installed, else DejaVu Sans; vector PDF and 300 dpi PNG. Error bars are 95% cluster-
bootstrap percentile intervals. Synthetic rehearsals carry a watermark in every figure and table."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402
import numpy as np                                 # noqa: E402
from matplotlib import font_manager                # noqa: E402

from .config import DELTA                           # noqa: E402  margin in points (3 after amendment 5)

OK = {"I1": "#0072B2", "I2": "#D55E00", "B": "#0072B2", "F": "#D55E00", "P": "#009E73", "grey": "#999999",
      "y": "#E69F00", "sky": "#56B4E9", "pink": "#CC79A7"}
HATCH = {"I1": "", "I2": "//"}
LEARN = {"transformer": "Transformer", "bilstm": "BiLSTM", "cnn": "1D-CNN", "lgbm": "n-gram LightGBM"}
MASKS = (("none", "None"), ("F", "F"), ("P", "P"), ("FP", "F and P"), ("rand_FP", "Matched\nrandom"))
FONT = "Arial" if any("Arial" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": FONT, "font.size": 7.5, "axes.linewidth": 1.0, "lines.linewidth": 1.2,
                     "errorbar.capsize": 2})


def _f1(res, learner, inst, role, where, mask):
    for r in res["f1"]:
        if (r["learner"], r["inst"], r["role"], r["where"], r["mask"]) == (learner, inst, role, where, mask):
            return r["f1"], r["ci_lo"], r["ci_hi"]
    return np.nan, np.nan, np.nan


def _comp(res, learner, comp):
    for r in res["decisions"][learner]["rows"]:
        if r["component"] == comp:
            return r["estimate"], r["ci_lo"], r["ci_hi"]
    return np.nan, np.nan, np.nan


def _err(v, lo, hi):
    return np.array([[v - lo], [hi - v]])


def _panel(ax, letter):
    ax.text(-0.16, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


def _finish(fig, res, path: Path, names):
    if "SYNTHETIC" in res["label"]:
        fig.suptitle(f"{res['label']}: synthetic data, not results", fontsize=8, color="black", y=0.995)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def fig4(res, out: Path, names):
    L = res["confirmatory"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.5, 1]})
    a = axs[0]
    x = np.arange(len(MASKS))
    for j, inst in enumerate(("I1", "I2")):
        for i, (m, _) in enumerate(MASKS):
            v, lo, hi = _f1(res, L, inst, "base", "own", m)
            a.bar(i + (j - 0.5) * 0.38, v, 0.36, color=OK[inst], hatch=HATCH[inst], edgecolor="black", linewidth=0.5,
                  yerr=_err(v, lo, hi), label=names[inst] if i == 0 else None)
    a.set_xticks(x, [lab for _, lab in MASKS])
    a.set_ylabel("Family-balanced macro-F1 (points)")
    a.set_ylim(0, 100)
    a.set_title(f"{LEARN[L]}: tokens masked at test", fontsize=7.5)
    a.legend(frameon=False, fontsize=7)
    _panel(a, "a")
    b = axs[1]
    learners = list(res["decisions"])
    for j, inst in enumerate(("I1", "I2")):
        for i, ln in enumerate(learners):
            v, lo, hi = _comp(res, ln, f"psi2|{inst}")
            b.errorbar(v, i + (j - 0.5) * 0.3, xerr=_err(v, lo, hi), fmt="o" if inst == "I1" else "s",
                       color=OK[inst], markersize=4, label=names[inst] if i == 0 else None)
    b.axvspan(-DELTA, DELTA, color=OK["grey"], alpha=0.25, lw=0)
    b.axvline(DELTA, color="black", linestyle="--", linewidth=1.0)
    b.set_yticks(range(len(learners)), [LEARN[ln] for ln in learners])
    b.set_xlabel("Excess drop over a matched random mask (points)")
    b.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    _panel(b, "b")
    _finish(fig, res, out / "Fig4_masking_effects", names)


def fig5(res, out: Path, names):
    L = res["confirmatory"]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0))
    a = axs[0, 0]
    labels = [f"{names['I1']} to {names['I2']}", f"{names['I2']} to {names['I1']}"]
    wrapped = [f"{names['I1']}\nto {names['I2']}", f"{names['I2']}\nto {names['I1']}"]
    for j, inst in enumerate(("I1", "I2")):
        vals = [_f1(res, L, inst, "base", "own", "none"), _f1(res, L, inst, "base", "other", "none"),
                _f1(res, L, inst, "mit", "other", "F")]
        for i, ((v, lo, hi), col, hat) in enumerate(zip(vals, (OK["B"], OK["F"], OK["P"]), ("", "//", ".."))):
            a.bar(j + (i - 1) * 0.27, v, 0.25, color=col, hatch=hat, edgecolor="black", linewidth=0.5,
                  yerr=_err(v, lo, hi), label=("Within", "Cross", "Cross, F-masked training")[i] if j == 0 else None)
    a.set_xticks([0, 1], wrapped)
    a.set_ylabel("Family-balanced macro-F1 (points)")
    a.set_ylim(0, 100)
    a.legend(frameon=False, fontsize=6.5)
    _panel(a, "a")
    learners = list(res["decisions"])
    b = axs[0, 1]
    for j, inst in enumerate(("I1", "I2")):
        for i, ln in enumerate(learners):
            v, lo, hi = _comp(res, ln, f"psi3|{inst}")
            b.errorbar(v, i + (j - 0.5) * 0.3, xerr=_err(v, lo, hi), fmt="o" if inst == "I1" else "s",
                       color=OK[inst], markersize=4, label=labels[j] if i == 0 else None)
    b.axvline(DELTA, color="black", linestyle="--", linewidth=1.0)
    b.set_yticks(range(len(learners)), [LEARN[ln] for ln in learners])
    b.set_xlabel("Transfer gap, psi3 (points)")
    b.set_ylim(-0.6, len(learners) - 0.4)
    b.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    _panel(b, "b")
    c = axs[1, 0]
    for i, ln in enumerate(learners):
        v, lo, hi = _comp(res, ln, "rho")
        c.errorbar(v, i, xerr=_err(v, lo, hi), fmt="D", color="black", markersize=4)
    c.axvline(0.10, color="black", linestyle="--", linewidth=1.0)
    c.set_yticks(range(len(learners)), [LEARN[ln] for ln in learners])
    c.set_xlabel("Recovered share of the pooled gap, rho")
    _panel(c, "c")
    d = axs[1, 1]
    for j, inst in enumerate(("I1", "I2")):
        for i, ln in enumerate(learners):
            v, lo, hi = _comp(res, ln, f"psi4b|{inst}")
            d.errorbar(v, i + (j - 0.5) * 0.3, xerr=_err(v, lo, hi), fmt="o" if inst == "I1" else "s",
                       color=OK[inst], markersize=4, label=names[inst] if i == 0 else None)
    d.axvline(-DELTA, color="black", linestyle="--", linewidth=1.0)
    d.set_yticks(range(len(learners)), [LEARN[ln] for ln in learners])
    d.set_xlabel("Within change after F-masked training (points)")
    d.legend(frameon=False, fontsize=6.5)
    _panel(d, "d")
    _finish(fig, res, out / "Fig5_transfer_and_recovery", names)


def fig6(res, out: Path, names):
    L = res["confirmatory"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.1, 1.3, 0.9]})
    a = axs[0]
    for j, inst in enumerate(("I1", "I2")):
        vals = [_f1(res, L, inst, "len", "own", "none"), _f1(res, L, inst, "art", "own", "B"),
                _f1(res, L, inst, "base", "own", "none")]
        for i, ((v, lo, hi), col, hat) in enumerate(zip(vals, (OK["grey"], OK["F"], OK["B"]), ("..", "//", ""))):
            a.bar(j + (i - 1) * 0.27, v, 0.25, color=col, hatch=hat, edgecolor="black", linewidth=0.5,
                  yerr=_err(v, lo, hi), label=("Length only", "Artefact only", "All tokens")[i] if j == 0 else None)
    a.set_xticks([0, 1], [names["I1"], names["I2"]])
    a.set_ylabel("Family-balanced macro-F1 (points)")
    a.set_ylim(0, 100)
    a.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    _panel(a, "a")
    b = axs[1]
    deep = [ln for ln in res["decisions"] if ln != "lgbm"]
    k = 0
    ticks = []
    for ln in deep:
        for role in ("base", "mit"):
            sh = [res["checks"][ck][f"phi_{role}"] for ck in res["checks"] if ck.endswith("|" + ln)]
            if not sh:
                continue
            bottom = 0.0
            for cl in "BFP":
                v = 100 * float(np.mean([s[cl] for s in sh]))
                b.bar(k, v, 0.7, bottom=bottom, color=OK[cl], hatch={"B": "", "F": "//", "P": ".."}[cl],
                      edgecolor="black", linewidth=0.5, label=cl if k == 0 else None)
                bottom += v
            ticks.append((k, f"{LEARN[ln]}\n{'base' if role == 'base' else 'F-masked'}"))
            k += 1
    b.set_xticks([t for t, _ in ticks], [lab for _, lab in ticks], fontsize=6)
    b.set_ylabel("Share of absolute attribution (%)")
    b.set_ylim(0, 100)
    b.legend(frameon=False, fontsize=6, title="Token class", title_fontsize=6, loc="upper center",
             bbox_to_anchor=(0.5, -0.25), ncol=3)
    _panel(b, "b")
    c = axs[2]
    rows = [(ck, v["env_id"]) for ck, v in res["checks"].items() if v.get("env_id", {}).get("tested")]
    for i, (ck, e) in enumerate(rows):
        c.bar(i, 100 * e["balanced_accuracy"], 0.6, color=OK["sky"], edgecolor="black", linewidth=0.5)
        c.text(i, max(e["balanced_accuracy"], e["chance"]) * 100 + 4, f"P = {e['p_permutation']:.2f}", ha="center",
               fontsize=6)
        c.hlines(100 * e["chance"], i - 0.35, i + 0.35, colors="black", linestyles="--")
    c.set_xticks(range(len(rows)), [ck.replace("|", "\n") for ck, _ in rows], fontsize=6)
    c.set_ylabel("Balanced accuracy, machine (%)")
    c.set_ylim(0, 100)
    _panel(c, "c")
    _finish(fig, res, out / "Fig6_probes_and_attributions", names)


def fmt_p(p: float, n_boot: int) -> str:
    floor = 1 / (n_boot + 1)
    return f"{p:.4f}" + (" (floor)" if abs(p - floor) < 1e-12 else "")


def table2(res, out: Path, names) -> str:
    rows = res["decisions"][res["confirmatory"]]["rows"]
    nb = res["plan"]["n_boot"]
    comp_name = lambda c: c.replace("psi1|", "psi1, ").replace("psi2|", "psi2, ").replace("psi3|", "psi3, ") \
        .replace("psi4b|", "psi4b, ").replace("I1", names["I1"]).replace("I2", names["I2"])
    lines = [f"Table 2. Confirmatory tests of H1-H4 ({LEARN[res['confirmatory']]}); estimates in macro-F1 points "
             f"(rho as a share), 95% cluster-bootstrap percentile intervals, {nb:,} replicates.", "",
             "| Hypothesis | Component | Estimate | 95% CI | Margin | P (component) | P (support, IUT) | Holm P | Verdict |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    with open(out / "Table2_H1_H4.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hypothesis", "component", "estimate", "ci_lo", "ci_hi", "margin", "p_component", "p_support",
                    "holm_support", "verdict"])
        for r in rows:
            dec = 3 if r["component"] == "rho" else 1
            lines.append(f"| {r['hypothesis']} | {comp_name(r['component'])} | {r['estimate']:.{dec}f} | "
                         f"{r['ci_lo']:.{dec}f} to {r['ci_hi']:.{dec}f} | {r['margin']} | "
                         f"{fmt_p(r['p_component'], nb)} | {fmt_p(r['p_support'], nb)} | "
                         f"{(r['holm_support'] if r['holm_support'] is not None else float('nan')):.4f} | {r['verdict']} |")
            w.writerow([r[k] for k in ("hypothesis", "component", "estimate", "ci_lo", "ci_hi", "margin",
                                       "p_component", "p_support", "holm_support", "verdict")])
    if "SYNTHETIC" in res["label"]:
        lines.insert(0, f"**{res['label']}: synthetic data, not results.**\n")
    text = "\n".join(lines) + "\n"
    (out / "Table2_H1_H4.md").write_text(text)
    return text


def table_tau(res, out: Path, names) -> str | None:
    """Supplementary Table S_tau (deviation D1): H1-H4 of the confirmatory learner at every tau_J analysed, from the
    same models; the primary analysis first."""
    if not res.get("sensitivity"):
        return None
    L, nb = res["confirmatory"], res["plan"]["n_boot"]
    order = [res["primary_analysis"]] + list(res["sensitivity"])
    get = lambda k: res if k == res["primary_analysis"] else res["sensitivity"][k]          # noqa: E731
    comps = ["psi1|I1", "psi1|I2", "psi2|I1", "psi2|I2", "psi3|I1", "psi3|I2", "rho", "psi4b|I1", "psi4b|I2"]
    hyp = {c: ("H4" if c == "rho" else "H" + c[3]) for c in comps}
    label = lambda k: ("registered antimode" if k == "registered" else f"tau_J = {k}") + (                  # noqa: E731
        " (primary)" if k == res["primary_analysis"] else "")
    nm = lambda c: c.replace("|I1", f", {names['I1']}").replace("|I2", f", {names['I2']}")                 # noqa: E731
    lines = [f"Supplementary Table S_tau. Confirmatory tests of H1-H4 ({LEARN[L]}) at every near-duplicate threshold "
             f"tau_J (deviation D1); estimates in macro-F1 points (rho as a share), 95% cluster-bootstrap percentile "
             f"intervals ({nb:,} replicates); kept = within-installation test traces kept per origin.", "",
             "| Analysis | tau_J by origin | Kept (" + names["I1"] + ") | Kept (" + names["I2"] + ") | H1 | H2 | H3 | H4 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    with open(out / "TableS_tau_sensitivity.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["analysis", "primary", "tau_J_by_origin", "hypothesis", "component", "estimate", "ci_lo", "ci_hi",
                    "p_support", "holm_support", "verdict"])
        for k in order:
            r = get(k)
            an, dec = r["analysis"], r["decisions"][L]
            tau = ", ".join(f"{an['tau_J_by_origin'][o]:.3f}" for o in sorted(an["tau_J_by_origin"]))
            kept = {i: ", ".join(f"{v['own']:,}/{v['own_of']:,}" for ck, v in an["kept"].items()
                                 if ck.startswith(i + "_")) for i in ("I1", "I2")}
            lines.append(f"| {label(k)} | {tau} | {kept['I1']} | {kept['I2']} | "
                         + " | ".join(dec["verdict"][h] for h in ("H1", "H2", "H3", "H4")) + " |")
            rows = {x["component"]: x for x in dec["rows"]}
            for c in comps:
                x = rows.get(c)
                est = x["estimate"] if x else r["psi"][L].get(c, float("nan"))
                w.writerow([k, int(k == res["primary_analysis"]), tau, hyp[c], c, est,
                            x["ci_lo"] if x else "", x["ci_hi"] if x else "", dec["p_support"].get(hyp[c], ""),
                            (dec["holm_support"] or {}).get(hyp[c], ""), dec["verdict"][hyp[c]]])
    lines += ["", "| Component | " + " | ".join(label(k) for k in order) + " |",
              "| --- | " + " | ".join("---" for _ in order) + " |"]
    for c in comps:
        cells = []
        for k in order:
            r = get(k)
            x = {y["component"]: y for y in r["decisions"][L]["rows"]}.get(c)
            d = 3 if c == "rho" else 1
            if x:
                cells.append(f"{x['estimate']:.{d}f} ({x['ci_lo']:.{d}f} to {x['ci_hi']:.{d}f})")
            else:
                v = r["psi"][L].get(c)
                cells.append("not tested" if v is None else f"{v:.{d}f} (not tested)")
        lines.append(f"| {nm(c)} | " + " | ".join(cells) + " |")
    if "SYNTHETIC" in res["label"] or "FIXTURE" in res["label"]:
        lines.insert(0, f"**{res['label']}: not results.**\n")
    text = "\n".join(lines) + "\n"
    (out / "TableS_tau_sensitivity.md").write_text(text)
    return text


def make_all(res, out: Path, names=None):
    names = names or {"I1": "I1", "I2": "I2"}
    out.mkdir(parents=True, exist_ok=True)
    fig4(res, out, names)
    fig5(res, out, names)
    fig6(res, out, names)
    table_tau(res, out, names)
    return table2(res, out, names)
