"""Synthetic two-installation corpora with planted ground truth (S10 rehearsal and protocol validation).

Nothing generated here is evidence about malware; every output that uses it is labelled SYNTHETIC.
Scenario 'planted': footprint (F) tokens carry family signal inside each installation and use disjoint
vocabularies across installations, so a model that leans on them transfers badly; behaviour (B) and probing (P)
tokens are shared. Scenario 'null': F tokens are shared and carry no family signal. The audit should support H2-H4
under 'planted' and not under 'null' (ISO/IEC 27041-style validation against known ground truth).
"""
from __future__ import annotations

import numpy as np

from .models import Trace
from .splits import grouped_split

MONTHS = ("2024-10", "2024-11", "2024-12", "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06")
REHEARSAL_ORIGINS = ((1, "2025-02", ("2025-03", "2025-04")), (2, "2025-04", ("2025-05", "2025-06")))


def make_corpus(scenario: str, K: int = 6, n_per_family=(130, 100), n_b: int = 150, n_p: int = 12, n_f: int = 12,
                L_range=(40, 64), dup_rate: float = 0.25, p_kind=(0.78, 0.07, 0.15), b_signal: float = 0.12,
                f_signal: float = 0.6, seed: int = 0) -> tuple[dict, int]:
    rng = np.random.default_rng(seed)
    b0, p0 = 4, 4 + n_b
    f0 = p0 + n_p
    V = f0 + 2 * n_f
    f_block = {"I1": np.arange(f0, f0 + n_f),
               "I2": np.arange(f0 + n_f, f0 + 2 * n_f) if scenario == "planted" else np.arange(f0, f0 + n_f)}
    fam_b = [rng.choice(np.arange(b0, p0), 20, replace=False) for _ in range(K)]
    fam_p = [rng.choice(np.arange(p0, f0), 3, replace=False) for _ in range(K)]
    corpus = {}
    for inst, n_fam in zip(("I1", "I2"), n_per_family):
        machines = 2 if inst == "I1" else 1
        perm = rng.permutation(f_block[inst])
        fam_f = [perm[(2 * k) % n_f:(2 * k) % n_f + 2] for k in range(K)]          # two F tokens per family
        mach_f = [rng.choice(f_block[inst], 2, replace=False) for _ in range(machines)]
        recs = []
        for k in range(K):
            roots = []
            for _ in range(n_fam):
                mach = int(rng.integers(machines))
                if roots and rng.random() < dup_rate:                 # near-duplicate of an earlier sample
                    src = recs[roots[int(rng.integers(len(roots)))]]
                    ids, cls = src["ids"].copy(), src["cls"].copy()
                    flip = (rng.random(len(ids)) < 0.08) & (cls == 0)
                    ids[flip] = rng.integers(b0, p0, flip.sum())
                    root = src["root"]
                else:
                    n = int(rng.integers(*L_range))
                    cls = rng.choice(3, n, p=p_kind)                   # 0 B, 1 F, 2 P
                    ids = np.empty(n, dtype=np.int64)
                    nb, npb, nf = (cls == 0).sum(), (cls == 2).sum(), (cls == 1).sum()
                    ids[cls == 0] = np.where(rng.random(nb) < b_signal, rng.choice(fam_b[k], nb),
                                             rng.integers(b0, p0, nb))
                    ids[cls == 2] = np.where(rng.random(npb) < 0.5, rng.choice(fam_p[k], npb),
                                             rng.integers(p0, f0, npb))
                    if scenario == "planted":
                        u = rng.random(nf)
                        ids[cls == 1] = np.where(u < f_signal, rng.choice(fam_f[k], nf),
                                                 np.where(u < f_signal + 0.2, rng.choice(mach_f[mach], nf),
                                                          rng.choice(f_block[inst], nf)))
                    else:
                        ids[cls == 1] = np.where(rng.random(nf) < 0.3, rng.choice(mach_f[mach], nf),
                                                 rng.choice(f_block[inst], nf))
                    root = len(recs)
                    roots.append(len(recs))
                betas = rng.integers(0, 4, len(ids))
                recs.append({"ids": ids, "betas": betas, "cls": cls, "family": f"fam{k}", "root": root,
                             "machine": f"{inst}-m{mach}", "month": MONTHS[int(rng.integers(len(MONTHS)))],
                             "length": {"n_calls": int(len(ids) * rng.integers(3, 9)), "n_apis": int(len(set(ids))),
                                        "n_procs": int(rng.integers(1, 4)), "duration_s": float(rng.uniform(5, 120)),
                                        "early_exit": 0}})
        corpus[inst] = recs
    return corpus, V


def build_cells(corpus: dict, V: int, seed: int = 0):
    """Rolling origins for I1 (by month) and a cluster-grouped split for I2, as the S9 rules prescribe."""
    from .audit import Cell, TestSet
    to_trace = lambda r, y: Trace(r["ids"], r["betas"], r["cls"], y, r["length"])
    sets, cells, monthly = {}, {}, {}
    i1, i2 = corpus["I1"], corpus["I2"]
    fam2 = np.array([r["family"] for r in i2])
    g = grouped_split(np.array([r["root"] for r in i2]), fam2, seed=seed)
    part2 = np.where(g["test"], "test", np.where(g["val"], "val", "train"))
    test2 = [r for r, p in zip(i2, part2) if p == "test"]
    sets["I2_g"] = TestSet("I2_g", np.array([r["family"] for r in test2]), np.array([r["root"] for r in test2]),
                           machine=np.array([r["machine"] for r in test2]))
    for o, val_m, test_ms in REHEARSAL_ORIGINS:
        tr1 = [r for r in i1 if r["month"] < val_m]
        va1 = [r for r in i1 if r["month"] == val_m]
        te1 = [r for r in i1 if r["month"] in test_ms]
        sets[f"I1_o{o}"] = TestSet(f"I1_o{o}", np.array([r["family"] for r in te1]),
                                   np.array([r["root"] for r in te1]), month=np.array([r["month"] for r in te1]),
                                   machine=np.array([r["machine"] for r in te1]))
        fams = sorted(set(r["family"] for r in i1))
        fi = {f: k for k, f in enumerate(fams)}
        seen = {r["root"] for r in tr1 + va1}                                   # eq. (9): drop test near-duplicates
        idx1 = np.array([i for i, r in enumerate(te1) if r["root"] not in seen], dtype=int)
        idx2 = np.arange(len(test2))                                            # grouped split: no shared roots
        tr2 = [r for r, p in zip(i2, part2) if p == "train"]
        va2 = [r for r, p in zip(i2, part2) if p == "val"]
        own1 = [to_trace(te1[i], fi[te1[i]["family"]]) for i in idx1]
        own2 = [to_trace(r, fi[r["family"]]) for r in test2]
        cells[f"I1_o{o}"] = Cell("I1", o, fams, V, [to_trace(r, fi[r["family"]]) for r in tr1],
                                 [to_trace(r, fi[r["family"]]) for r in va1], f"I1_o{o}", idx1, own1,
                                 "I2_g", idx2, own2)
        cells[f"I2_o{o}"] = Cell("I2", o, fams, V, [to_trace(r, fi[r["family"]]) for r in tr2],
                                 [to_trace(r, fi[r["family"]]) for r in va2], "I2_g", idx2, own2,
                                 f"I1_o{o}", idx1, own1)
    fams = sorted(set(r["family"] for r in i1))
    fi = {f: k for k, f in enumerate(fams)}
    seen1 = {r["root"] for r in i1 if r["month"] <= REHEARSAL_ORIGINS[0][1]}
    for m in ("2025-03", "2025-04", "2025-05", "2025-06"):                     # P9 series for the origin-1 model
        rs = [r for r in i1 if r["month"] == m and r["root"] not in seen1]
        monthly[m] = (np.array([fi[r["family"]] for r in rs]), [to_trace(r, fi[r["family"]]) for r in rs])
    cells["I1_o1"].monthly = monthly
    return sets, cells
