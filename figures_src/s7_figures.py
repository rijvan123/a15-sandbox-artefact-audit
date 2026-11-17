#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S7 diagrams D1-D6 for the A15 path (behavioural malware evidence, sandbox-artefact audit).

Deterministic Matplotlib drawings in millimetre coordinates; no image-generation model is used
(Springer Nature AI policy). Run:  python s7_figures.py OUTDIR

Outputs in OUTDIR
  S7_D1 ... S7_D6      one vector PDF, an SVG (text kept as text) and a 600 dpi PNG per diagram
  S7_Fig1/2/3          two-panel composites (Fig. 1 = D1 + D2, Fig. 2 = D3 + D4, Fig. 3 = D5 + D6)
                       as vector PDF, 600 dpi PNG and a 300 dpi preview PNG
  check_report.txt     geometry checks: text overflow, text overlap, text crossing arrows

Drawing rules (R3): one sans-serif typeface at one size (7 pt) for every label, bold lower-case
panel letters, white background, no 3D or shading, lines >= 1 pt, Okabe-Ito palette with
line-style redundancy (B solid, P dotted, F dashed), full text width of the one-column
manuscript (160 mm). Arial is used when installed; otherwise Liberation Sans, which is
metric-compatible with Arial, so layout does not change at the final Cowork rendering.
"""
import itertools
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(OUTDIR, exist_ok=True)

_NAMES = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((n for n in ("Arial", "Helvetica", "Liberation Sans") if n in _NAMES), "DejaVu Sans")
FS = 7.0
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FONT], "font.size": FS,
    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": FONT + ":italic",
    "mathtext.bf": FONT + ":bold", "mathtext.sf": FONT, "mathtext.cal": FONT, "mathtext.fallback": "stixsans",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "hatch.linewidth": 1.0, "savefig.facecolor": "white", "figure.facecolor": "white",
})

MM = 1 / 25.4
W = 160.0                       # full text width of the one-column manuscript
LH = FS * 1.2 / 72 * 25.4       # line pitch, mm
PAD = 0.8

OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7"}


def tint(hexc, a):
    """Mix an Okabe-Ito colour with white; a = share of colour."""
    h = hexc.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)


GREY = "#7F7F7F"
try:
    from matplotlib.ft2font import FT2Font
    _CMAP = FT2Font(font_manager.findfont(FONT)).get_charmap()
except Exception:
    _CMAP = {}
GE = "≥" if ord("≥") in _CMAP else r"$\geq$"
GREY_T = "#404040"
LIGHT = "#F2F2F2"
DASH = (0, (5, 2))
DOT = (0, (1, 1.6))
CLS = {  # token classes: colour and line style carry the same information (greyscale-safe)
    "B": dict(fc=tint(OI["blue"], .16), ec=OI["blue"], ls="-"),
    "P": dict(fc=tint(OI["orange"], .22), ec=OI["orange"], ls=DOT),
    "F": dict(fc=tint(OI["verm"], .16), ec=OI["verm"], ls=DASH),
}


def _patch(cls, *args, hatchcolor=None, **kw):
    """Create a patch; use the hatch colour argument when this Matplotlib supports it."""
    if hatchcolor is not None:
        try:
            return cls(*args, hatchcolor=hatchcolor, **kw)
        except (TypeError, AttributeError):
            pass
    return cls(*args, **kw)


class Canvas:
    """Axes in millimetres, y growing downwards, with a registry for geometry checks."""

    def __init__(self, fig, rect, wc, hc, name):
        self.ax = fig.add_axes(rect)
        self.ax.set_xlim(0, wc)
        self.ax.set_ylim(hc, 0)
        self.ax.axis("off")
        self.W, self.H, self.name = wc, hc, name
        self.texts, self.obstacles, self.paths = [], [], []

    # ---------- text ----------
    def text(self, x, y, s, ha="center", va="center", bold=False, italic=False, color="black",
             container=None, rotation=0, z=6):
        t = self.ax.text(x, y, s, ha=ha, va=va, fontsize=FS, fontweight="bold" if bold else "normal",
                         fontstyle="italic" if italic else "normal", color=color, rotation=rotation,
                         rotation_mode="anchor", zorder=z)
        self.texts.append((t, container))
        return t

    def block(self, x, yc, lines, ha="center", container=None):
        y0 = yc - (len(lines) - 1) * LH / 2
        for i, ln in enumerate(lines):
            s, st = (ln, "") if isinstance(ln, str) else ln
            self.text(x, y0 + i * LH, s, ha=ha, bold="b" in st, italic=("t" in st or "i" in st),
                      color=GREY_T if "t" in st else "black", container=container)

    # ---------- shapes ----------
    def box(self, x, y, w, h, lines=(), fc="white", ec="black", ls="-", lw=1.0, r=0.9, ha="center",
            hatch=None, z=2, obstacle=True):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                           lw=lw, ls=ls, hatch=hatch, zorder=z)
        self.ax.add_patch(p)
        if obstacle:
            self.obstacles.append(("rect", (x, y, x + w, y + h)))
        if lines:
            tx = x + w / 2 if ha == "center" else x + PAD + 0.4
            self.block(tx, y + h / 2, lines, ha=ha,
                       container=("rect", (x + 0.25, y + 0.25, x + w - 0.25, y + h - 0.25)))
        return p

    def uml(self, x, y, w, h, title, attrs, fc=LIGHT, ec="black", ls="-", th=4.4):
        """UML-style concept box: title compartment over an attribute compartment."""
        self.box(x, y, w, h, (), fc=fc, ec=ec, ls=ls)
        self.ax.add_line(Line2D([x, x + w], [y + th, y + th], color=ec, lw=1.0, ls=ls, zorder=3))
        self.text(x + w / 2, y + th / 2, title, bold=True,
                  container=("rect", (x + 0.25, y + 0.25, x + w - 0.25, y + th)))
        self.block(x + PAD + 0.6, y + th + (h - th) / 2, attrs, ha="left",
                   container=("rect", (x + 0.25, y + th, x + w - 0.25, y + h - 0.25)))

    def diamond(self, cx, cy, w, h, lines, fc="white", ec="black", ls="-"):
        pts = [(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)]
        self.ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=1.0, ls=ls, zorder=2,
                                  joinstyle="miter"))
        self.obstacles.append(("diamond", (cx, cy, w, h)))
        self.block(cx, cy, lines, container=("diamond", (cx, cy, w, h)))

    def cylinder(self, x, y, w, h, lines, fc="white", ec="black", ry=1.3):
        """Data store: top ellipse, straight sides, lower arc."""
        ax = self.ax
        ax.add_patch(Ellipse((x + w / 2, y + h - ry), w, 2 * ry, fc=fc, ec=ec, lw=1.0, zorder=2))
        ax.add_patch(Rectangle((x, y + ry), w, h - 2 * ry, fc=fc, ec="none", zorder=2.1))
        ax.add_line(Line2D([x, x], [y + ry, y + h - ry], color=ec, lw=1.0, zorder=2.2))
        ax.add_line(Line2D([x + w, x + w], [y + ry, y + h - ry], color=ec, lw=1.0, zorder=2.2))
        ax.add_patch(Ellipse((x + w / 2, y + ry), w, 2 * ry, fc=fc, ec=ec, lw=1.0, zorder=2.3))
        self.obstacles.append(("rect", (x, y, x + w, y + h)))
        self.block(x + w / 2, y + 2 * ry + (h - 2 * ry) / 2 - 0.3, lines,
                   container=("rect", (x + 0.4, y + 2 * ry, x + w - 0.4, y + h - 0.1)))

    def line(self, pts, color="black", lw=1.0, ls="-", z=4):
        xs, ys = zip(*pts)
        self.ax.add_line(Line2D(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="butt",
                                dash_capstyle="butt", solid_joinstyle="miter", zorder=z))
        self.paths.append([tuple(p) for p in pts])

    def arrow(self, pts, color="black", lw=1.0, ls="-", head=True, z=4, hl=1.6):
        """Polyline with a solid filled head at the last point (dashed shafts keep solid heads)."""
        pts = [(float(a), float(b)) for a, b in pts]
        self.paths.append(pts)
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        seg = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / seg, (y1 - y0) / seg
        shaft = pts[:-1] + [(x1 - ux * hl * 0.8, y1 - uy * hl * 0.8)] if head else pts
        xs, ys = zip(*shaft)
        self.ax.add_line(Line2D(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="butt",
                                dash_capstyle="butt", solid_joinstyle="miter", zorder=z))
        if head:
            ms = (hl / 25.4 * 72) / 0.4
            self.ax.add_patch(FancyArrowPatch((x1 - ux * hl, y1 - uy * hl), (x1, y1), arrowstyle="-|>",
                                              mutation_scale=ms, color=color, lw=lw, shrinkA=0,
                                              shrinkB=0, zorder=z + 0.1, joinstyle="miter"))


def letter(cv, s):
    cv.text(0.8, 0.9, s, ha="left", va="top", bold=True)


# =====================================================================================
# D1  Conceptual model of behavioural malware evidence (Fig. 1a)
# =====================================================================================
def draw_D1(cv):
    letter(cv, "a")
    cv.uml(6, 4, 54, 17, "Malware activity",
           ["ATT&CK tactics and techniques", "e.g. process injection (T1055)", "sandbox checks (T1497)"])
    cv.uml(100, 4, 54, 17, "Analysis environment",
           ["CAPEv2 version, guest OS build", "machine, hook set, timeout", "network route, analysis agent"])
    cv.arrow([(60, 12.6), (100, 12.6)])
    cv.text(80, 10.4, "is executed in")

    # execution trace with its three token classes
    cv.box(6, 29, 148, 28, ())
    cv.box(9, 31, 44, 19.5, [("B  Behaviour", "b"), "calls made in any environment",
                             "e.g. encryption, persistence keys"], ha="left", **CLS["B"])
    cv.box(56, 31, 46, 19.5, [("P  Probing", "b"), "malware checks of its environment",
                              "ATT&CK T1497: system,", "user-activity and time checks"],
           ha="left", lw=1.2, **CLS["P"])
    cv.box(105, 31, 46, 19.5, [("F  Footprint", "b"), "values the environment sets:",
                               "user and host names, agent", "activity, skipped sleeps,",
                               "failed network replies"], ha="left", **CLS["F"])
    cv.text(9, 53.9, "Execution trace: one API-call sequence per run", ha="left",
            container=("rect", (6.25, 29.25, 153.75, 56.75)))

    cv.arrow([(20, 21), (20, 31)])
    cv.text(21.4, 25.6, "performs", ha="left")
    cv.arrow([(48, 21), (66, 31)])
    cv.text(59.5, 24.0, "performs", ha="left")
    cv.arrow([(128, 21), (128, 31)])
    cv.text(129.4, 25.6, "sets", ha="left")
    cv.arrow([(110, 21), (94, 31)], ls=DASH)
    cv.text(101.5, 23.5, "answers checks", ha="right")

    # evidence item, finding, report
    cv.uml(6, 66, 40, 17, "Evidence item",
           ["trace and report SHA-256", r"environment record $e$", "label record, custody entry"])
    cv.uml(58, 66, 44, 17, "Analytical finding",
           ["family attribution (top three)", "within- or cross-installation", "error rate; B, P, F shares"])
    cv.uml(114, 66, 40, 17, "Forensic report",
           ["finding and error rate", "ledger proof", "FRE 702(d); BSA s. 63"])
    cv.arrow([(46, 75.5), (58, 75.5)])
    cv.text(52, 73.3, "supports")
    cv.arrow([(102, 75.5), (114, 75.5)])
    cv.text(108, 73.3, "reported")

    cv.arrow([(22, 57), (22, 66)])
    cv.text(23.4, 63.2, r"captured with $e$", ha="left")
    cv.arrow([(79, 50.5), (79, 66)], color=OI["orange"], ls=DOT, lw=1.2)
    cv.text(77.6, 59.8, "environment-contingent reliance", ha="right")
    cv.arrow([(112, 50.5), (96, 66)], color=OI["verm"], ls=DASH, lw=1.2)
    cv.text(106.5, 61.0, "contaminating reliance", ha="left")

    # cross-cutting requirements
    cv.text(6, 85.2, "Cross-cutting requirements", ha="left")
    cv.box(6, 87, 48, 7.6, [("Integrity", "b"), "digests, hash chain, tokens"],
           fc=tint(OI["green"], .14), ec=OI["green"])
    cv.box(56, 87, 48, 7.6, [("Provenance", "b"), "source, version, licence"],
           fc=tint(OI["sky"], .18), ec=OI["sky"])
    cv.box(106, 87, 48, 7.6, [("Privacy", "b"), "argument classes only"],
           fc=tint(OI["purple"], .14), ec=OI["purple"])


# =====================================================================================
# D2  Layered architecture (Fig. 1b)
# =====================================================================================
def draw_D2(cv):
    letter(cv, "b")
    cv.block(13, 5.0, [("Privacy and", "b"), ("governance", "b")])
    cv.text(77.5, 5.0, "Layers and the artefact each passes on", bold=True)
    cv.block(146.75, 5.0, [("Standards", "b"), ("plane", "b")])

    # privacy and governance plane
    cv.box(2, 9, 22, 93.5, (), fc=tint(OI["purple"], .08), ec=OI["purple"], ls=DASH)
    items = [["Class tokens", "only"], ["Personal-data", "scan before", "release"], ["HMAC actor", "pseudonyms"],
             ["Role-based", "access, log"], ["Retention", "schedule"], ["Licence-bound", "sharing", "(NC, NC-SA)"],
             ["No network", "indicators"]]
    hs = [len(i) * LH + 1.8 for i in items]
    gap = (89.5 - sum(hs)) / (len(items) - 1)
    y = 11.0
    for it, h in zip(items, hs):
        cv.box(3.5, y, 19, h, it, fc="white", ec=OI["purple"])
        y += h + gap

    # layers
    lx0, lx1, tw, lhh = 27.0, 128.0, 19.0, 14.5
    ys = [9.5, 29.0, 48.5, 68.0, 87.5]
    tints = [OI["sky"], OI["green"], OI["orange"], OI["blue"], OI["purple"]]
    titles = [[("L1", "b"), "Acquisition"], [("L2", "b"), "Preservation"], [("L3", "b"), "Feature", "engineering"],
              [("L4", "b"), "Analysis"], [("L5", "b"), "Explanation", "and reporting"]]
    comps = [
        [["MELD-DS-448", "CAPEv2 reports"], ["WinMET", "MALVADA JSON"], ["Environment", r"record $e$", "extractor"],
         ["Label-field", "stripper,", "provenance"]],
        [["SHA-256 per", "file and trace"], ["Hash-chained", "append-only", "ledger"], ["Merkle roots,", "RFC 3161", "tokens"],
         ["Write-once", "store, access", "log, anchor"]],
        [["Class", "tokeniser", r"$(a,\,\kappa,\,r)$"], ["Common hook", "set, repeats", "merged"],
         ["Frozen", "lexicon tags", "B, P, F"], ["Masks, near-", "duplicate", "removal"]],
        [["Transformer", "encoder"], ["BiLSTM with", "attention"], ["1D-CNN and", "n-gram", "LightGBM"],
         ["2 × 2 masking", "and dropout;", "three probes"]],
        [["Integrated", "Gradients by", "token class"], ["Error rates", "within and", "across"],
         ["Environment-", "match gate,", "referral"], ["Report and", "certificate", "fields"]],
    ]
    cw = (lx1 - lx0 - tw - 1.0 - 3 * 1.0 - 1.0) / 4
    for i, yy in enumerate(ys):
        cv.box(lx0, yy, lx1 - lx0, lhh, (), fc="white", ec="black")
        cv.box(lx0, yy, tw, lhh, titles[i], fc=tint(tints[i], .22), ec="black")
        x = lx0 + tw + 1.0
        for c in comps[i]:
            cv.box(x, yy + 1.2, cw, lhh - 2.4, c, fc="white", ec=GREY)
            x += cw + 1.0

    labels = ["trace, environment and label records", "sealed evidence items: ID, digests, ledger index",
              "class-tagged token sequences, split manifests", "posteriors, embeddings, per-seed predictions"]
    for i in range(4):
        y_from, y_to = ys[i] + lhh, ys[i + 1]
        cv.arrow([(62, y_from), (62, y_to)])
        cv.text(64, (y_from + y_to) / 2, labels[i], ha="left")
    cv.arrow([(62, ys[4] + lhh), (62, ys[4] + lhh + 3.6)])
    cv.text(64, ys[4] + lhh + 2.0, "finding record, report and ledger proof", ha="left")

    # digests of models, results and findings return to the ledger
    cv.arrow([(lx1, ys[4] + 7.25), (131.0, ys[4] + 7.25), (131.0, ys[1] + 7.25), (lx1, ys[1] + 7.25)], ls=DASH)
    cv.text(133.4, (ys[1] + ys[4]) / 2 + 7.25, "digests of models, results and findings", rotation=90)

    # standards plane
    std = [["ISO/IEC 27037", "5.4.2–5.4.4;", "SP 800-86", "collection"],
           ["ISO/IEC 27037", "5.4.5, 6.9;", "CSF RS.AN-06,", "RS.AN-07"],
           ["ISO/IEC 27042", "examination;", "SP 800-86", "examination"],
           ["ISO/IEC 27042", "analysis;", "ISO/IEC 27041", "validation"],
           ["ISO/IEC 27041,", "27042 reporting;", "SP 800-86;", "CSF RS.AN-03"]]
    cv.box(135.5, 9, 22.5, 93.5, (), fc=LIGHT, ec=GREY, ls=DASH)
    for i, yy in enumerate(ys):
        cv.box(136.5, yy, 20.5, lhh, std[i], fc="white", ec=GREY)


# =====================================================================================
# D3  End-to-end evidence workflow, swim lanes (Fig. 2a)
# =====================================================================================
def draw_D3(cv):
    letter(cv, "a")
    lanes = [("Investigator", 4.0, 35.0), ("Acquisition and preservation", 35.0, 79.0),
             ("Analysis", 79.0, 120.0), ("Reporting", 120.0, 158.0)]
    for i, (nm, x0, x1) in enumerate(lanes):
        cv.ax.add_patch(Rectangle((x0, 7.0), x1 - x0, cv.H - 7.6, fc="white" if i % 2 == 0 else "#FAFAFA",
                                  ec=GREY, lw=1.0, zorder=0.5))
        cv.ax.add_patch(Rectangle((x0, 1.0), x1 - x0, 6.0, fc=LIGHT, ec=GREY, lw=1.0, zorder=0.6))
        cv.text((x0 + x1) / 2, 4.0, nm, bold=True, container=("rect", (x0 + 0.2, 1.2, x1 - 0.2, 6.8)))

    R = {1: 15.0, 2: 29.5, 3: 44.0, 4: 58.5, 5: 74.5, 6: 92.0, 7: 106.0, 8: 120.0}
    h3 = 3 * LH + 1.6
    h2 = 2 * LH + 1.6

    def b3(x0, x1, yc, lines, **kw):
        cv.box(x0, yc - h3 / 2, x1 - x0, h3, lines, **kw)

    # R1-R2: intake, acquisition, integrity check
    b3(6, 33, R[1], [("Case intake", "b"), "trace and authority", ("27037 5.4.2", "t")])
    b3(38, 76, R[1], [("Acquire report", "b"), r"extract $e$, strip labels", ("27037 5.4.3–5.4.4", "t")])
    cv.arrow([(33, R[1]), (38, R[1])])
    cv.diamond(47, R[2], 24, 12, ["Digest", "matches?"])
    cv.box(63, R[2] - h2 / 2, 14.5, h2, ["Quarantine,", "re-acquire"])
    cv.arrow([(59, R[2]), (63, R[2])])
    cv.text(61.0, R[2] - 1.9, "no")
    cv.arrow([(70.25, R[2] - h2 / 2), (70.25, R[1] + h3 / 2)])
    cv.arrow([(47, R[2] + 6), (47, R[3] - h3 / 2)])
    cv.text(48.4, (R[2] + 6 + R[3] - h3 / 2) / 2, "yes", ha="left")

    # R3-R4: seal, tokenise, ledger, inference
    b3(38, 76, R[3], [("Seal evidence item", "b"), "ledger entry, RFC 3161 token",
                      ("27037 6.9; CSF RS.AN-06/07", "t")])
    b3(82, 117, R[3], [("Tokenise to classes", "b"), r"hook set $H_c$, F masked", ("27042 examination", "t")])
    cv.arrow([(76, R[3]), (82, R[3])])
    cv.cylinder(42, 53.5, 30, 11.5, ["Ledger and", "write-once store"])
    cv.arrow([(57, R[3] + h3 / 2), (57, 53.5)])
    # S8 decision D1 (1 Nov 2026): the finding comes from the mitigated model on the F-masked trace;
    # the referral share phi_F is computed with the base model on the unmasked trace.
    b3(82, 117, R[4], [("Infer and attribute", "b"), r"mitigated model; base $\varphi_F$",
                       ("27042 analysis; 800-86 3.3", "t")])
    cv.arrow([(99.5, R[3] + h3 / 2), (99.5, R[4] - h3 / 2)])

    # R5: environment-match gate
    cv.diamond(99.5, R[5], 40, 15, ["Environment", "validated and", r"$\varphi_F \leq \theta_F$?"])
    cv.arrow([(99.5, R[4] + h3 / 2), (99.5, R[5] - 7.5)])
    b3(124, 156, R[5], [("Automated finding", "b"), "within-installation rate", ("27041 validation scope", "t")])
    cv.arrow([(119.5, R[5]), (124, R[5])])
    cv.text(121.75, R[5] - 2.5, "yes")
    b3(6, 33, R[5], [("Analyst review", "b"), "cross-installation", "error rate"])
    cv.arrow([(79.5, R[5]), (33, R[5])])
    cv.text(75.5, R[5] - 2.5, "no")
    cv.text(97.3, R[5] + 9.6, "27041 scope", ha="right", italic=True, color=GREY_T)
    cv.text(101.7, R[5] + 9.6, "FRE 702(d)", ha="left", italic=True, color=GREY_T)

    # R6: validated environments and re-validation loop
    cv.cylinder(87, 87.5, 25, 10.5, ["Validated set", r"$E_{\mathrm{val}}$"])
    cv.arrow([(99.5, 87.5), (99.5, R[5] + 7.5)])
    cv.arrow([(27, R[5] + h3 / 2), (27, 93.0), (87, 93.0)], ls=DASH)
    cv.text(57, 91.0, "new installation: re-validate")

    # R7-R8: report, retention
    b3(124, 156, R[7], [("Report", "b"), "finding, ledger proof", ("BSA s. 63; 27042", "t")])
    cv.arrow([(140, R[5] + h3 / 2), (140, R[7] - h3 / 2)])
    cv.arrow([(12, R[5] + h3 / 2), (12, R[7]), (124, R[7])])
    cv.text(60, R[7] - 2.0, "with analyst decision")
    b3(124, 156, R[8], [("Retention or disposal", "b"), "digests kept", ("GDPR Art. 5(1)(e)", "t")])
    cv.arrow([(140, R[7] + h3 / 2), (140, R[8] - h3 / 2)])


# =====================================================================================
# D4  Chain of custody, batch anchoring and tamper evidence (Fig. 2b)
# =====================================================================================
def draw_D4(cv):
    letter(cv, "b")
    xs = [17.0, 55.0, 93.0, 131.0]
    bw, by, bh = 26.0, 4.0, 19.0
    ents = [
        [("Entry 1: acquire", "b"), r"ID, digest $d_1$", r"actor $\mathrm{HMAC}_k$", r"time token $\tau_1$",
         r"previous hash $h_0$"],
        [("Entry k: freeze", "b"), r"lexicon $\Lambda$, H1–H4,", "margins, manifests", r"time token $\tau_k$",
         r"previous hash $h_{k-1}$"],
        [("Entry k+1: score", "b"), "first test score,", "model digest", r"time token $\tau_{k+1}$",
         r"previous hash $h_k$"],
        [("Entry n: finding", "b"), "finding record,", "error rate, gate", r"time token $\tau_n$",
         r"previous hash $h_{n-1}$"],
    ]
    for i, (x, e) in enumerate(zip(xs, ents)):
        anchor = i == 1
        cv.box(x, by, bw, bh, e, fc=tint(OI["green"], .16) if anchor else "white", ec="black",
               lw=1.5 if anchor else 1.0, ha="left")
    yc = by + bh / 2
    cv.block(6.6, yc, [r"$h_0$ =", "SHA-256", "(study ID)"])
    cv.arrow([(12.9, yc), (17.0, yc)])
    cv.arrow([(43, yc), (55, yc)], ls=DASH)
    cv.text(49, yc - 2.2, r"$h_1 \ldots h_{k-1}$")
    cv.arrow([(81, yc), (93, yc)])
    cv.text(87, yc - 2.2, r"$h_k$")
    cv.arrow([(119, yc), (131, yc)], ls=DASH)
    cv.text(125, yc - 2.2, r"$\ldots h_{n-1}$")
    cv.text(80, 25.8, r"$h_i$ = SHA-256(entry $i$ $\|$ $h_{i-1}$); editing entry $k$ changes $h_k$ and every later hash")

    # batch anchoring: Merkle tree, time-stamping authority, token
    cv.text(3, 31.5, "Batch anchoring", ha="left", bold=True)
    leaves = [3.0, 16.0, 29.0, 42.0]
    lwid, ly, lh = 11.0, 63.5, 5.6
    for x, s in zip(leaves, [r"$h_j$", r"$h_{j+1}$", r"$h_{j+2}$", r"$h_{j+3}$"]):
        cv.box(x, ly, lwid, lh, [s])
    l1 = [(4.0, 24.0), (30.0, 50.0)]
    l1y = 52.5
    for (x0, x1), s in zip(l1, [r"$H(h_j \| h_{j+1})$", r"$H(h_{j+2} \| h_{j+3})$"]):
        cv.box(x0, l1y, x1 - x0, lh, [s])
    cv.box(13.0, 40.5, 28.0, 6.5, [r"Merkle root $R_b$"])
    for x, tx in zip(leaves, [11.0, 17.0, 37.0, 43.0]):
        cv.arrow([(x + lwid / 2, ly), (tx, l1y + lh)])
    cv.arrow([(14.0, l1y), (23.0, 47.0)])
    cv.arrow([(40.0, l1y), (31.0, 47.0)])
    cv.box(56.0, 37.0, 23.0, 11.0, ["Time-stamping", "authority", "(RFC 3161)"])
    cv.arrow([(41.0, 43.75), (56.0, 43.75)])
    cv.text(48.5, 41.6, r"submit $R_b$")
    cv.box(56.0, 53.0, 23.0, 8.5, [r"token $\mathrm{TST}_b$", "write-once store"])
    cv.arrow([(67.5, 48.0), (67.5, 53.0)])

    # verification walk and tamper example
    cv.text(84, 31.5, "Verification", ha="left", bold=True)
    cv.text(84, 37.0, r"Pre-registration holds if $\tau_k$ (lexicon frozen)", ha="left")
    cv.text(84, 40.6, r"precedes $\tau_{k+1}$ (first test score)", ha="left")
    cv.text(84, 46.6, r"Tamper example: entry $k$ edited after scoring", ha="left")
    xm = [88 + 8.4 * i for i in range(9)]
    labs = [r"$h_1$", r"$\ldots$", r"$h_{k-1}$", r"$h_k$", r"$h_{k+1}$", r"$\ldots$", r"$h_n$", r"$R_b$", "token"]
    for i, (x, s) in enumerate(zip(xm, labs)):
        ok = i < 3
        cv.ax.plot([x], [53.0], marker="o" if ok else "x", ms=4.2, mew=1.2, mfc="white",
                   mec=OI["green"] if ok else OI["verm"], ls="none", zorder=5)
        cv.text(x, 57.6, s)
    cv.ax.plot([88], [63.2], marker="o", ms=4.2, mew=1.2, mfc="white", mec=OI["green"], ls="none", zorder=5)
    cv.text(90.4, 63.2, "recomputed value matches", ha="left")
    cv.ax.plot([128], [63.2], marker="x", ms=4.2, mew=1.2, mec=OI["verm"], ls="none", zorder=5)
    cv.text(130.4, 63.2, "fails", ha="left")
    cv.text(84, 68.6, "one edit breaks all later hashes, the root and its token", ha="left")


# =====================================================================================
# D5  Analysis models and masking audit (Fig. 3a)
# =====================================================================================
def draw_D5(cv):
    letter(cv, "a")
    cv.box(2, 6.5, 22, 15.5, [("CAPEv2 report", "b"), "behaviour calls", "only; label", "fields stripped"])
    cv.box(2, 26.5, 22, 17.5, [("Environment", "b"), (r"record $e$", "b"), "version, build,", "hook set,",
                               "timeout, route"])
    cv.box(28, 6.5, 24, 22.8, [("Class tokeniser", "b"), r"$t = (a,\,\kappa,\,r)$", "API, argument and",
                               "return classes;", r"hook set $H_c$;", "repeats merged;", r"length $L$"])
    cv.arrow([(24, 14.25), (28, 14.25)])

    # example token strip and masks
    seq = "BBPBFBPFBB"
    rows = [("tokens", None), ("mask F", {4, 7}), ("mask P", {2, 6}), ("mask F, P", {2, 4, 6, 7}),
            ("random", {0, 3, 5, 9})]
    sx0, sq, sg = 67.0, 3.4, 0.5
    cv.text(86.0, 8.6, "Frozen lexicon tags and masks (example)")
    for ri, (lab, masked) in enumerate(rows):
        yc = 13.6 + 6.0 * ri
        cv.text(65.5, yc, lab, ha="right")
        for j, c in enumerate(seq):
            x = sx0 + j * (sq + sg)
            if masked is not None and j in masked:
                cv.box(x, yc - sq / 2, sq, sq, ["M"], fc="#D9D9D9", ec="black", r=0.3)
            else:
                st = CLS[c]
                cv.box(x, yc - sq / 2, sq, sq, [c], fc=st["fc"], ec=st["ec"], ls=st["ls"], r=0.3,
                       lw=1.2 if c == "P" else 1.0)
    cv.arrow([(52, 13.6), (56.0, 13.6)])
    cv.text(86.0, 42.6, r"masked sequence $x$ of length $L$")

    # learners
    lx0, lx1 = 112.0, 131.0
    lrn = [(6.5, 8.6, [("Transformer", "b"), "encoder"]), (17.6, 8.6, [("BiLSTM", "b"), "with attention"]),
           (28.7, 8.6, [("1D-CNN", "b"), "kernels 3, 5, 7"]),
           (39.8, 11.6, [("n-gram", "b"), "LightGBM,", "TF-IDF 1–3"])]
    mids = []
    for yy, hh, lines in lrn:
        cv.box(lx0, yy, lx1 - lx0, hh, lines)
        mids.append(yy + hh / 2)
    cv.line([(105.5, 25.6), (109.0, 25.6)])
    cv.line([(109.0, mids[0]), (109.0, mids[-1])])
    for m in mids:
        cv.arrow([(109.0, m), (lx0, m)])

    # outputs
    ox0, ox1 = 140.0, 158.0
    outs = [(6.5, 10.5, [("Family", "b"), "posterior", r"$f_\theta(x)$, $K$"]),
            (21.0, 10.5, [("Attribution", "b"), "shares", r"$\varphi_B, \varphi_P, \varphi_F$"]),
            (35.5, 13.5, [("Error rate", "b"), "within or", "across,", r"chosen by $e$"])]
    omid = []
    for yy, hh, lines in outs:
        cv.box(ox0, yy, ox1 - ox0, hh, lines)
        omid.append(yy + hh / 2)
    for m in mids:
        cv.line([(lx1, m), (134.5, m)])
    cv.line([(134.5, mids[0]), (134.5, mids[-1])])
    for m in omid:
        cv.arrow([(134.5, m), (ox0, m)])
    cv.text(137.2, omid[1] - 2.0, "IG")
    cv.arrow([(13.0, 44.0), (13.0, 52.8), (149.0, 52.8), (149.0, 49.0)])
    cv.text(62.0, 50.8, "the environment record selects the matching error rate")

    # training and evaluation regimes
    cv.text(2.0, 58.6, "Training and evaluation regimes", ha="left", bold=True)
    cv.box(22.0, 61.0, 20.5, 5.2, ["test: no mask"], fc=LIGHT)
    cv.box(43.5, 61.0, 20.5, 5.2, ["test: F masked"], fc=LIGHT)
    cv.box(2.0, 67.0, 19.0, 10.5, ["train:", "no mask"], fc=LIGHT)
    cv.box(2.0, 78.5, 19.0, 10.5, ["train:", "F masked"], fc=LIGHT)
    cells = {(0, 0): [("base model", "b"), "within and", "cross tests"], (0, 1): [("reliance", "b"), "test (H2)"],
             (1, 0): [("mitigated,", "b"), "unmasked test"], (1, 1): [("mitigated", "b"), "deployed model", "(H4)"]}
    for (i, j), lines in cells.items():
        cv.box(22.0 + j * 21.5, 67.0 + i * 11.5, 20.5, 10.5, lines,
               fc=tint(OI["green"], .14) if (i, j) == (1, 1) else "white")
    cv.block(2.0, 92.5, ["Also P and F, P test masks", "and matched random masks"], ha="left")
    cv.box(68.0, 61.0, 44.0, 12.6, [("Artefact dropout", "b"), "each F token masked with",
                                    r"probability $p$ during training", "(the mitigation)"])
    cv.box(68.0, 76.5, 44.0, 12.6, [("Probes", "b"), "length-only (H1 baseline)", "artefact-only, F and P (H1)",
                                    "environment-ID (secondary)"])
    cv.box(116.0, 61.0, 42.0, 28.0, [("Estimands and decision rule", "b"), "H1: artefact-only − length-only",
                                     "H2: masked drop − random drop",
                                     r"H3: gap $G = F1_{\mathrm{within}} - F1_{\mathrm{cross}}$",
                                     r"H4: recovered share $\rho$ of $G$", r"margin $\Delta$ = 3 macro-F1 points",
                                     r"supported if lower bound > $\Delta$", r"rejected if upper bound < $\Delta$"],
           ha="left")


# =====================================================================================
# D6  Experimental-evaluation protocol (Fig. 3b)
# =====================================================================================
def draw_D6(cv):
    letter(cv, "b")
    x0c, x1c, xm0, mw = 27.0, 36.0, 38.0, 5.9

    def mx(m):
        return xm0 + mw * m          # m = months after 1 January 2024

    for lab, yc, vm in [("I1, origin 1", 9.5, 13), ("I1, origin 2", 15.5, 15), ("I1, origin 3", 21.5, 17)]:
        cv.text(25.0, yc, lab, ha="right")
        cv.ax.add_patch(Rectangle((x0c, yc - 2), mx(vm) - x0c, 4, fc=tint(OI["blue"], .30), ec=OI["blue"],
                                  lw=1.0, zorder=2))
        cv.ax.add_patch(_patch(Rectangle, (mx(vm), yc - 2), mw, 4, fc=tint(OI["orange"], .40),
                               ec=OI["orange"], lw=1.0, hatch="////", zorder=2, hatchcolor=OI["orange"]))
        cv.ax.add_patch(_patch(Rectangle, (mx(vm + 1), yc - 2), 2 * mw, 4, fc=tint(OI["green"], .30),
                               ec=OI["green"], lw=1.0, hatch="xx", zorder=2, hatchcolor=OI["green"]))
    cv.text(25.0, 27.5, "I2, WinMET", ha="right")
    cv.box(mx(0), 25.5, mx(20) - mx(0), 4.0,
           ["no submission dates: one split grouped by near-duplicate cluster"], ls=DASH, r=0.3)

    ay = 31.6
    cv.ax.add_line(Line2D([x0c, x1c], [ay, ay], color="black", lw=1.0))
    cv.ax.add_line(Line2D([xm0, mx(20)], [ay, ay], color="black", lw=1.0))
    for xb in (36.6, 37.4):
        cv.ax.add_line(Line2D([xb - 0.4, xb + 0.4], [ay + 0.8, ay - 0.8], color="black", lw=1.0))
    for t in [x0c, x1c] + [mx(m) for m in (0, 3, 6, 9, 12, 15, 18, 20)]:
        cv.ax.add_line(Line2D([t, t], [ay, ay + 1.0], color="black", lw=1.0))
    ty = ay + 2.8
    cv.text((x0c + x1c) / 2, ty, "2020–23")
    cv.text(mx(0), ty, "Jan 2024", ha="left")
    cv.text(mx(3), ty, "Apr")
    cv.text(mx(6), ty, "Jul")
    cv.text(mx(9), ty, "Oct")
    cv.text(mx(12), ty, "Jan 2025", ha="left")
    cv.text(mx(15), ty, "Apr")
    cv.text(mx(18), ty, "Jul")
    cv.text(mx(20), ty, "Aug", ha="right")
    cv.text(x0c, ay + 6.6, "Traces in I1 by first-seen year: 834 in 2020–2023, 5,146 in 2024, 20,186 in 2025",
            ha="left")

    ky = 42.8

    def swatch(x, **kw):
        cv.ax.add_patch(_patch(Rectangle, (x, ky - 1.5), 4.0, 3.0, lw=1.0, zorder=2, **kw))

    swatch(27.0, fc=tint(OI["blue"], .30), ec=OI["blue"])
    cv.text(32.0, ky, "training months", ha="left")
    swatch(60.0, fc=tint(OI["orange"], .40), ec=OI["orange"], hatch="////", hatchcolor=OI["orange"])
    cv.text(65.0, ky, "validation month", ha="left")
    swatch(94.0, fc=tint(OI["green"], .30), ec=OI["green"], hatch="xx", hatchcolor=OI["green"])
    cv.text(99.0, ky, "test window", ha="left")
    swatch(124.0, fc="white", ec="black", ls=DASH)
    cv.text(129.0, ky, "grouped split (no dates)", ha="left")

    cv.box(2.0, 47.5, 84.0, 20.0, [("Harmonisation before scoring", "b"), "1  values: argument classes",
                                   r"2  hook set: common $H_c$", "3  curation: same filter on both",
                                   f"4  labels: agreement {GE} 80% (K4)",
                                   "5  near-duplicates: MinHash, API 3-grams"], ha="left")
    cv.text(92.0, 48.6, "Transfer tests on shared families", ha="left", bold=True)
    cv.box(92.0, 51.5, 21.0, 12.0, [("I1", "b"), "MELD-DS-448", "CAPEv2, 22H2"])
    cv.box(137.0, 51.5, 21.0, 12.0, [("I2", "b"), "WinMET", "modified CAPEv2"])
    cv.arrow([(113.0, 55.0), (137.0, 55.0)])
    cv.text(125.0, 53.0, "train I1, test I2")
    cv.arrow([(137.0, 60.0), (113.0, 60.0)])
    cv.text(125.0, 62.0, "train I2, test I1")
    cv.block(92.0, 68.0, [f"paired same-binary test if {GE} 200 traces",
                          f"in {GE} 5 families share a SHA-256"], ha="left")

    cv.box(2.0, 73.5, 42.0, 16.0, [("Runs", "b"), "4 learners × 5 seeds", "× 3 origins × masking",
                                   "regimes, both directions"])
    cv.box(47.0, 73.5, 64.0, 16.0, [("Statistics", "b"), "cluster bootstrap, 9,999 replicates, 95% CI",
                                    "exact McNemar on seed-majority predictions",
                                    "Wilcoxon over families; TOST at ±3 points",
                                    "Holm over H1–H4; Hodges–Lehmann estimate"])
    cv.box(114.0, 73.5, 44.0, 16.0, [(r"Decision rule ($\Delta$ = 3 points)", "b"),
                                     r"supported: lower bound > $\Delta$", r"rejected: upper bound < $\Delta$",
                                     "otherwise inconclusive", "gates K3–K6 before scoring"])


# =====================================================================================
# geometry checks
# =====================================================================================
def _bb(cv, t, r):
    bb = t.get_window_extent(renderer=r)
    inv = cv.ax.transData.inverted()
    (xa, ya), (xb, yb) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    return (min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb))


def _overlap(a, b, m=0.0):
    return not (a[2] + m <= b[0] or b[2] + m <= a[0] or a[3] + m <= b[1] or b[3] + m <= a[1])


def _in_diamond(px, py, d, tol=0.0):
    cx, cy, w, h = d
    return abs(px - cx) / (w / 2) + abs(py - cy) / (h / 2) <= 1 - tol


def _seg_hits(p, q, r):
    x0, y0 = p
    x1, y1 = q
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, x0 - r[0]), (dx, r[2] - x0), (-dy, y0 - r[1]), (dy, r[3] - y0)):
        if pp == 0:
            if qq < 0:
                return False
        else:
            t = qq / pp
            if pp < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def check(fig, canvases):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []
    for cv in canvases:
        items = [(t, c, _bb(cv, t, r)) for t, c in cv.texts if t.get_text().strip()]
        f = lambda b: "(%.1f,%.1f,%.1f,%.1f)" % b
        for t, c, b in items:
            s = t.get_text().replace("\n", " ")[:40]
            if b[0] < 0.2 or b[1] < 0.2 or b[2] > cv.W - 0.2 or b[3] > cv.H - 0.2:
                out.append(f"{cv.name}: outside canvas '{s}' {f(b)}")
            if c is None:
                for kind, g in cv.obstacles:
                    if kind == "rect" and _overlap(b, g, 0.15):
                        out.append(f"{cv.name}: free text on box '{s}' {f(b)} vs {f(g)}")
                    if kind == "diamond":
                        corners = [(b[0], b[1]), (b[2], b[1]), (b[0], b[3]), (b[2], b[3])]
                        if any(_in_diamond(px, py, g) for px, py in corners):
                            out.append(f"{cv.name}: free text on diamond '{s}' {f(b)}")
            elif c[0] == "rect":
                g = c[1]
                if b[0] < g[0] - 0.02 or b[2] > g[2] + 0.02 or b[1] < g[1] - 0.02 or b[3] > g[3] + 0.02:
                    out.append(f"{cv.name}: overflow '{s}' text {f(b)} box {f(g)}")
            elif c[0] == "diamond":
                corners = [(b[0], b[1]), (b[2], b[1]), (b[0], b[3]), (b[2], b[3])]
                if not all(_in_diamond(px, py, c[1], 0.02) for px, py in corners):
                    out.append(f"{cv.name}: overflow diamond '{s}' {f(b)}")
            grown = (b[0] - 0.2, b[1] - 0.2, b[2] + 0.2, b[3] + 0.2)
            for pts in cv.paths:
                if any(_seg_hits(p, q, grown) for p, q in zip(pts[:-1], pts[1:])):
                    out.append(f"{cv.name}: line crosses text '{s}' {f(b)}")
                    break
        for (t1, c1, b1), (t2, c2, b2) in itertools.combinations(items, 2):
            if c1 is not None and c1 is c2:
                if min(b1[3], b2[3]) - max(b1[1], b2[1]) > 0.45 and _overlap(b1, b2):
                    out.append(f"{cv.name}: line collision '{t1.get_text()[:25]}' / '{t2.get_text()[:25]}'")
                continue
            if _overlap(b1, b2, 0.05):
                out.append(f"{cv.name}: text overlap '{t1.get_text()[:25]}' / '{t2.get_text()[:25]}'")
    return out


# =====================================================================================
# build
# =====================================================================================
SPECS = [
    ("S7_D1_conceptual_model", draw_D1, 96.0),
    ("S7_D2_layered_architecture", draw_D2, 107.0),
    ("S7_D3_evidence_workflow", draw_D3, 127.0),
    ("S7_D4_custody_ledger", draw_D4, 72.0),
    ("S7_D5_analysis_model_masking_audit", draw_D5, 97.0),
    ("S7_D6_evaluation_protocol", draw_D6, 91.0),
]
FIGS = [("S7_Fig1_D1_D2", 0, 1), ("S7_Fig2_D3_D4", 2, 3), ("S7_Fig3_D5_D6", 4, 5)]
GAP = 3.0


def main():
    report = [f"font: {FONT}, {FS} pt; width {W} mm"]
    for name, fn, h in SPECS:
        fig = plt.figure(figsize=(W * MM, h * MM))
        cv = Canvas(fig, [0, 0, 1, 1], W, h, name)
        fn(cv)
        report += check(fig, [cv])
        for ext in ("pdf", "svg"):
            fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"))
        fig.savefig(os.path.join(OUTDIR, f"{name}.png"), dpi=600)
        plt.close(fig)
    for name, ia, ib in FIGS:
        (na, fa, ha), (nb, fb, hb) = SPECS[ia], SPECS[ib]
        hf = ha + GAP + hb
        fig = plt.figure(figsize=(W * MM, hf * MM))
        ca = Canvas(fig, [0, (hb + GAP) / hf, 1, ha / hf], W, ha, name + "/a")
        cb = Canvas(fig, [0, 0, 1, hb / hf], W, hb, name + "/b")
        fa(ca)
        fb(cb)
        fig.savefig(os.path.join(OUTDIR, f"{name}.pdf"))
        fig.savefig(os.path.join(OUTDIR, f"{name}.png"), dpi=600)
        fig.savefig(os.path.join(OUTDIR, f"{name}_preview_300dpi.png"), dpi=300)
        report.append(f"{name}: {W:.0f} x {hf:.0f} mm")
        plt.close(fig)
    with open(os.path.join(OUTDIR, "check_report.txt"), "w") as fh:
        fh.write("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
