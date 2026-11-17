"""S10 assembly: test sets and cells from the sealed S9 records, streamed one record at a time (7 Nov 2026).

The 2 November engine loaded every record of both installations at once, which needs far more than the 7 GB of the
analysis machine (S9 amendment 2). This module reads each installation once, in the order of run_s9.iter_records,
and keeps only what S10 uses: API 3-gram sets and MinHash signatures (near-duplicates), head-plus-tail sequence
stores (any L <= L_CAP, as in the dry run), token-type sets of training traces (vocabulary) and scalar fields.

Frozen inputs (split manifests, family sets, alias table, lexicon) are read from outputs/freeze_bundle/ and accepted
only if each file's SHA-256 equals the digest in the ledger's freeze entry. Token classes come from the frozen
lexicon's token table; the code rules must agree with it on every type seen (else the run stops).

Rules carried over from the 2 November engine (S10 decisions):
  S10-1  test traces are de-duplicated against the union of both installations' training and validation traces at
         each origin (all families); tau_J is the antimode of nearest-neighbour similarities on that union
         (fallback 0.8);
  S10-3  WinMET's grouped test set is clustered at tau_J = 0.8;
  S10-5  WinMET models are refitted only when an origin's family set differs (cells share a fit group).
Rules added on 7 November 2026, after the OSF registration and before any test trace is scored:
  S10-11 every retained CAPE report is a trace (as in the S9 counts); a binary's reports share its split part (S9
         rule: a binary listed in a test part is test only); reports of one binary in a test set form one bootstrap
         cluster; a test report is also dropped when its binary has a report in that origin's training union.
  S10-12 the P9 series uses MELD-DS-448 traces of the origin-1 families first seen March-August 2025, de-duplicated
         against the origin-1 training union, scored by the origin-1 models.
Deviation D1 (8 Nov 2026, after the first assembly and before any model was trained or scored): cells hold every
shared-family test trace with its largest Jaccard index to the origin's training union and a same-binary flag;
equation (9) is applied at analysis time for each tau_J in ANALYSES (audit.restrict). The primary analysis uses
tau_J = 0.8; test sets carry their near-duplicate clusters at every analysis's tau_J (WinMET's at 0.8 throughout).
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config as C
from . import dedup, lexicon, splits, tokens
from .audit import Cell, TestSet
from .audit import _keep as audit_keep
from .models import Trace

P9_MONTHS = ("2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08")
ANALYSES = ("registered", "0.7", "0.8", "0.9")   # deviation D1: tau_J of each analysis ("registered" = the antimode)
PRIMARY = "0.8"
ASSEMBLY_FORMAT = "D1 (8 Nov 2026): all shared-family test traces with their largest Jaccard index to the training union"
N_HASH = 128                                   # MinHash length: 32 x 4 bands for tau_J, 25 x 5 at 0.8, 16 x 8 at 0.9


@dataclass
class Assembly:
    sets: dict
    cells: dict
    groups: dict                               # fit group -> cell keys that share one training set (S10-5)
    info: dict = field(default_factory=dict)
    digest: str = ""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------
# Frozen inputs, verified against the freeze entry of the ledger
# ------------------------------------------------------------------------------------------------
def frozen_inputs(P, ledger) -> dict:
    frz = [r["entry"] for r in ledger.records if r["entry"]["action"] == "freeze"]
    if len(frz) != 1:
        raise SystemExit(f"expected one freeze entry in the ledger, found {len(frz)}")
    dig = json.loads(frz[0]["payload"])
    bundle = P.out / "freeze_bundle"
    bad = {n: d for n, d in dig.items() if not (bundle / n).exists() or sha256_file(bundle / n) != d}
    if bad:
        raise SystemExit(f"frozen files differ from the freeze entry (a deviation; nothing scored): {sorted(bad)}")
    load = lambda n: json.loads((bundle / n).read_text(encoding="utf-8"))          # noqa: E731
    spl = {n[6:-5]: load(n)["ids"] for n in dig if n.startswith("split_")}
    if "I2_grouped" not in spl:
        raise SystemExit("this engine expects WinMET's grouped split (split_I2_grouped.json), as frozen at S9")
    return {"digests": dig, "freeze_entry": frz[0]["i"], "alias": load("alias_table.json"),
            "family_sets": {int(k): list(v) for k, v in load("family_sets.json").items()},
            "splits": spl, "token_table": load("lexicon.json").get("token_table", {})}


def parts_of(ids: dict) -> dict:
    """Split parts as sets of SHA-256; a binary listed in the test part is test only, then validation (S9 rule)."""
    test = set(ids["test"])
    val = set(ids["val"]) - test
    return {"train": set(ids["train"]) - test - val, "val": val, "test": test}


# ------------------------------------------------------------------------------------------------
# Near-duplicates without the set of all candidate pairs
# ------------------------------------------------------------------------------------------------
def nn_similarities_stream(sets: list, sigs: np.ndarray, stats: dict | None = None) -> np.ndarray:
    """dedup.nn_similarities(sets, sigs), computed one trace at a time.

    The candidate pairs are those of dedup.candidates_within(sigs[:, :128], 32, 4) (dedup.candidate_neighbours yields
    the same pairs without holding them all, amendment of 3 Nov 2026), and each trace's value is the largest exact
    Jaccard index over its candidates. A pair is skipped only when both traces already hold 1.0, the largest
    possible value; identical sets are recognised by their bytes. The result therefore equals nn_similarities."""
    n = len(sets)
    nn = np.zeros(n)
    visited = evaluated = identical = 0
    if n >= 2:
        keys = [hashlib.blake2b(np.ascontiguousarray(s).tobytes(), digest_size=16).digest() for s in sets]
        for i, js in dedup.candidate_neighbours(sigs[:, :128], 32, 4):
            if nn[i] >= 1.0:
                js = js[nn[js] < 1.0]
            si, ki = sets[i], keys[i]
            for j in js.tolist():
                visited += 1
                if nn[i] >= 1.0 and nn[j] >= 1.0:
                    continue
                sj = sets[j]
                if keys[j] == ki and np.array_equal(si, sj):
                    jj = 1.0
                    identical += 1
                else:
                    jj = dedup.jaccard_sorted(si, sj)
                    evaluated += 1
                if jj > nn[i]:
                    nn[i] = jj
                if jj > nn[j]:
                    nn[j] = jj
    if stats is not None:
        stats.update({"traces": n, "pairs_visited": visited, "jaccard_evaluated": evaluated,
                      "identical_pairs": identical})
    return nn


def merge_same_binary(cluster: np.ndarray, shas: list, fams: list) -> tuple[np.ndarray, int]:
    """S10-11: reports of one binary (same SHA-256 and family) join one cluster; ids renumbered by first member."""
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    first, merges = {}, 0
    for c, s, f in zip(cluster.tolist(), shas, fams):
        key = (s, f)
        if key in first:
            a, b = find(first[key]), find(c)
            if a != b:
                parent[b] = a
                merges += 1
        else:
            first[key] = c
    out, num = [], {}
    for c in cluster.tolist():
        out.append(num.setdefault(find(c), len(num)))
    return np.array(out, dtype=np.int64), merges


# ------------------------------------------------------------------------------------------------
# One streaming pass per installation
# ------------------------------------------------------------------------------------------------
class TokenTable:
    def __init__(self):
        self.ids: dict[str, int] = {}
        self.str: list[str] = []

    def tid(self, t: str) -> int:
        k = self.ids.get(t)
        if k is None:
            k = self.ids[t] = len(self.str)
            self.str.append(t)
        return k


def _scan(P, inst: str, need, fam_of, month_of, hook: set, mh, tt: TokenTable, log) -> dict:
    import run_s9 as R9
    d = {"sha": [], "fam": [], "month": [], "machine": [], "length": {}, "grams": {}, "sig": {}, "store": {},
         "types": {}, "n": 0, "records_read": 0}
    k = -1
    for r in R9.iter_records(P, inst, "s10 assemble" if log else ""):
        d["records_read"] += 1
        if r["curated_out"]:
            continue
        k += 1
        sha, fam = r["sha256"], fam_of(r)
        d["sha"].append(sha)
        d["fam"].append(fam)
        d["month"].append(month_of(sha) if month_of else "")
        d["machine"].append(str((r.get("env") or {}).get("machine")))
        need_g, need_s, need_t = need(sha, fam)
        if need_g:
            g = dedup.api_3grams([a if a in hook else C.OOS for a, *_ in r["calls"]])
            d["sig"][k] = mh.signature(g)
            d["grams"][k] = g.astype(np.uint32)                  # 3-gram ids are below 2^31
        if need_s or need_t:
            s = R9.sequences([r], hook)[0]
            if need_s:
                part = tokens.head_tail_store(s, C.L_CAP)
                d["store"][k] = (np.array([tt.tid(t) for t, _ in part], dtype=np.int32),
                                 np.array([b for _, b in part], dtype=np.int8), len(s))
                d["length"][k] = r["length"]
            if need_t:
                d["types"][k] = np.unique(np.array([tt.tid(t) for t, _ in s], dtype=np.int32))
    d["n"] = k + 1
    d["fam"] = np.array(d["fam"], dtype=object)
    return d


def _encoder(D: dict, vmap: np.ndarray, cls_by_tid: np.ndarray, L: int, fi: dict):
    h = L // 2

    def enc(inst: str, k: int) -> Trace:
        ids_g, betas, _ = D[inst]["store"][k]
        if len(ids_g) > L:                                       # head plus tail; equals trunc_head_tail(seq, L)
            ids_g = np.concatenate([ids_g[:h], ids_g[-h:]])
            betas = np.concatenate([betas[:h], betas[-h:]])
        return Trace(vmap[ids_g], betas.copy(), cls_by_tid[ids_g], fi[D[inst]["fam"][k]], D[inst]["length"][k])
    return enc


def _test_set(name: str, d: dict, idx: list, taus: dict, with_month: bool) -> tuple[TestSet, dict]:
    """A test set with near-duplicate clusters (equation 9) at the tau_J of every analysis key (deviation D1)."""
    fams = [d["fam"][k] for k in idx]
    shas = [d["sha"][k] for k in idx]
    clusters, info = {}, {"traces": len(idx), "by_analysis": {}}
    cache: dict = {}
    for key, tau in taus.items():
        if tau not in cache:
            st: dict = {}
            if idx:
                cl, cross = dedup.test_clusters([d["grams"][k] for k in idx], np.stack([d["sig"][k] for k in idx]),
                                                fams, fams, tau, stats=st)
                cl, merges = merge_same_binary(cl, shas, fams)
            else:
                cl, cross, merges = np.zeros(0, dtype=np.int64), 0, 0
            cache[tau] = (cl, {"tau_J": tau, "clusters": int(len(set(cl.tolist()))), "cross_label_pairs": int(cross),
                               "same_binary_merges": int(merges), "lsh": st})
        clusters[key], info["by_analysis"][key] = cache[tau]
    main = PRIMARY if PRIMARY in clusters else next(iter(clusters))
    ts = TestSet(name, np.array(fams, dtype=object), clusters[main],
                 month=np.array([d["month"][k] for k in idx], dtype=object) if with_month else None,
                 machine=np.array([d["machine"][k] for k in idx], dtype=object), clusters=clusters)
    return ts, info


def _maxj(u_sets, u_sigs, u_shas, d, idx, tau_min):
    """Largest exact Jaccard index of each trace to the training union, and whether its binary is in the union.

    One pass with the banding of the smallest tau_J analysed (32 x 4 for tau_J <= 0.7, as the registered rule uses
    at its antimode); at tau_J 0.8 and 0.9 this banding proposes a pair with probability above 0.9999."""
    if not idx:
        return np.zeros(0), np.zeros(0, dtype=bool)
    _, maxj = dedup.dedup_test(u_sets, [d["grams"][k] for k in idx], u_sigs, np.stack([d["sig"][k] for k in idx]),
                               tau_min)
    same = np.array([d["sha"][k] in u_shas for k in idx], dtype=bool)
    return maxj, same


def nn_profile(nn: np.ndarray) -> dict:
    """Histogram and kernel density of nearest-neighbour similarities, with every local peak and minimum, so the
    antimode the rule picked can be judged (Supplementary Information)."""
    from scipy.stats import gaussian_kde
    x = nn[nn > 0]
    hist, edges = np.histogram(x, bins=50, range=(0.0, 1.0))
    out = {"n": int(x.size), "hist_counts": hist.tolist(), "hist_edges": edges.round(3).tolist(),
           "quantiles": {str(q): float(np.quantile(x, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)} if x.size else {}}
    if x.size >= 50 and np.ptp(x) > 0:
        grid = np.linspace(0.0, 1.0, 401)
        dens = gaussian_kde(x)(grid)
        ext = lambda cmp: [{"at": float(round(grid[i], 3)), "density": float(dens[i])}            # noqa: E731
                           for i in range(1, len(grid) - 1) if cmp(i)]
        out["kde_peaks"] = ext(lambda i: dens[i] > dens[i - 1] and dens[i] >= dens[i + 1])
        out["kde_minima"] = ext(lambda i: dens[i] < dens[i - 1] and dens[i] <= dens[i + 1])
        out["kde_on_grid"] = {"at": grid[::4].round(3).tolist(), "density": dens[::4].round(4).tolist()}
    return out


def kept_count(maxj: np.ndarray, same: np.ndarray, fam_ok: np.ndarray, tau: float) -> int:
    return int(((maxj < tau) & ~same & fam_ok).sum())


# ------------------------------------------------------------------------------------------------
# Assembly
# ------------------------------------------------------------------------------------------------
def assemble(P, frozen: dict, log=print) -> Assembly:
    import pandas as pd
    import run_s9 as R9
    t0 = time.perf_counter()
    hook = set(json.loads((P.work / "hook_set.json").read_text()))
    alias, fam_sets = frozen["alias"], frozen["family_sets"]
    origins = [o for o, *_ in C.ORIGINS]
    k_all = set().union(*map(set, fam_sets.values()))
    man = pd.read_csv(P.data / "I1" / "malicious_dataset_manifest.csv")
    lab1 = dict(zip(man.sha256.str.lower(), man.family.map(splits.norm_family)))
    mon1 = dict(zip(man.sha256.str.lower(), pd.to_datetime(man.first_seen).dt.strftime("%Y-%m").fillna("")))
    p1 = {o: parts_of(frozen["splits"][f"I1_o{o}"]) for o in origins}
    p2 = parts_of(frozen["splits"]["I2_grouped"])
    tv1 = set().union(*(p1[o]["train"] | p1[o]["val"] for o in origins))
    te1 = set().union(*(p1[o]["test"] for o in origins))
    tr1 = set().union(*(p1[o]["train"] for o in origins))
    tv2, te2, tr2 = p2["train"] | p2["val"], p2["test"], p2["train"]
    need = {"I1": lambda s, f: (s in tv1 or (f in k_all and s in te1), f in k_all and (s in tv1 or s in te1),
                                f in k_all and s in tr1),
            "I2": lambda s, f: (s in tv2 or (f in k_all and s in te2), f in k_all and (s in tv2 or s in te2),
                                f in k_all and s in tr2)}
    al = lambda f: alias.get(f, f)                                                    # noqa: E731
    fam_of = {"I1": lambda r: al(lab1.get(r["sha256"], "")),
              "I2": lambda r: al(splits.norm_family(r["labels"]["avclass"]))}
    tt, mh, D = TokenTable(), dedup.MinHasher(N_HASH), {}
    for inst in ("I1", "I2"):
        D[inst] = _scan(P, inst, need[inst], fam_of[inst], (lambda s: mon1.get(s, "")) if inst == "I1" else None,
                        hook, mh, tt, log)
        log(f"{time.strftime('%H:%M:%S')} s10 assemble {inst}: {D[inst]['n']:,} retained traces; "
            f"{len(D[inst]['grams']):,} 3-gram sets, {len(D[inst]['store']):,} sequences; memory "
            f"{R9.rss_mb()[0]:,.0f} MB (peak {R9.rss_mb()[1]:,.0f} MB)")
    table = frozen["token_table"]
    cls_by_tid = np.empty(len(tt.str), dtype=np.int8)
    missing = disagree = 0
    for j, t in enumerate(tt.str):
        code = lexicon.classify(t)[0]
        if t in table:
            c = table[t][0]
            disagree += int(c != code)
        else:
            c, missing = code, missing + 1
        cls_by_tid[j] = "BFP".index(c)
    if disagree:
        raise SystemExit(f"{disagree} token types are classed differently by the code and the frozen lexicon")
    rows = lambda inst, shaset, fams=None: [k for k in range(D[inst]["n"]) if D[inst]["sha"][k] in shaset   # noqa
                                            and (fams is None or D[inst]["fam"][k] in fams)]
    info = {"records": {i: {"read": D[i]["records_read"], "retained": D[i]["n"], "grams": len(D[i]["grams"]),
                            "sequences": len(D[i]["store"])} for i in D},
            "lexicon": {"token_types_seen": len(tt.str), "not_in_frozen_table": missing, "disagreements": disagree},
            "origins": {}, "test_sets": {}, "cells": {}}
    sets, cells, groups, reuse2 = {}, {}, {}, {}
    te2_idx = rows("I2", te2, k_all)
    sets["I2_g"], info["test_sets"]["I2_g"] = _test_set("I2_g", D["I2"], te2_idx,
                                                        {k: C.TAU_J_FALLBACK for k in ANALYSES}, False)   # S10-3
    for o in origins:
        t1 = time.perf_counter()
        K = list(fam_sets[o])
        fi = {f: k for k, f in enumerate(K)}
        Ks = set(K)
        u = [("I1", k) for k in rows("I1", p1[o]["train"] | p1[o]["val"])] + [("I2", k) for k in rows("I2", tv2)]
        u_sets = [D[i]["grams"][k] for i, k in u]
        u_sigs = np.stack([D[i]["sig"][k] for i, k in u])
        u_shas = {D[i]["sha"][k] for i, k in u}
        nst: dict = {}
        nn = nn_similarities_stream(u_sets, u_sigs, stats=nst)
        tau, tau_info = dedup.antimode_tau(nn)                   # the registered rule
        taus = {k: (tau if k == "registered" else float(k)) for k in ANALYSES}
        tau_min = min(taus.values())
        n1 = f"I1_o{o}"
        te1_idx = rows("I1", p1[o]["test"], Ks)
        sets[n1], info["test_sets"][n1] = _test_set(n1, D["I1"], te1_idx, taus, True)
        set_idx, mj, kinfo = {n1: te1_idx, "I2_g": te2_idx}, {}, {}
        for name, inst in ((n1, "I1"), ("I2_g", "I2")):
            maxj, same = _maxj(u_sets, u_sigs, u_shas, D[inst], set_idx[name], tau_min)
            fam_ok = np.isin(sets[name].family, K)
            mj[name] = (maxj, same, np.flatnonzero(fam_ok))
            kinfo[name] = {"in_families": int(fam_ok.sum()), "same_binary": int((same & fam_ok).sum()),
                           "kept": {k: kept_count(maxj, same, fam_ok, t) for k, t in taus.items()},
                           "max_jaccard_quantiles": {str(q): float(np.quantile(maxj[fam_ok], q))
                                                     for q in (0.1, 0.25, 0.5, 0.75, 0.9)} if fam_ok.any() else {}}
        encs = {}
        for i, j in (("I1", "I2"), ("I2", "I1")):
            own, oth = (n1, "I2_g") if i == "I1" else ("I2_g", n1)
            ck = f"{i}_o{o}"
            rkey = (i, tuple(K))
            if i == "I2" and rkey in reuse2:                     # S10-5: identical WinMET cell, shared fit
                enc, V, tr, va, Lc, gid = reuse2[rkey]
            else:
                parts = p1[o] if i == "I1" else p2
                tr_k, va_k = rows(i, parts["train"], Ks), rows(i, parts["val"], Ks)
                vocab = tokens.Vocab([[(tt.str[t], 0) for t in D[i]["types"][k].tolist()] for k in tr_k])
                Lc = tokens.choose_L([D[i]["store"][k][2] for k in tr_k])
                if Lc < 2:
                    raise SystemExit(f"{ck}: L = {Lc}; the 95th percentile of collapsed training lengths is below 2")
                vmap = np.array([vocab.encode(t) for t in tt.str], dtype=np.int32)
                enc = _encoder(D, vmap, cls_by_tid, Lc, fi)
                tr, va, V, gid = [enc(i, k) for k in tr_k], [enc(i, k) for k in va_k], len(vocab), ck
                if i == "I2":
                    reuse2[rkey] = (enc, V, tr, va, Lc, gid)
            encs[i] = enc
            (omaxj, osame, opos), (xmaxj, xsame, xpos) = mj[own], mj[oth]
            own_tr = [enc(i, set_idx[own][q]) for q in opos.tolist()]        # every shared-family test trace
            oth_tr = [enc(j, set_idx[oth][q]) for q in xpos.tolist()]
            cells[ck] = Cell(i, o, K, V, tr, va, own, opos, own_tr, oth, xpos, oth_tr,
                             own_maxj=omaxj[opos], own_same=osame[opos], other_maxj=xmaxj[xpos],
                             other_same=xsame[xpos], tau_registered=float(tau))
            groups.setdefault(gid, []).append(ck)
            info["cells"][ck] = {"L": Lc, "V": V, "n_train": len(tr), "n_val": len(va), "n_own_all": len(own_tr),
                                 "n_other_all": len(oth_tr), "fit_group": gid,
                                 "kept_own": {k: int(len(audit_keep(omaxj[opos], osame[opos], t)))
                                              for k, t in taus.items()},
                                 "kept_other": {k: int(len(audit_keep(xmaxj[xpos], xsame[xpos], t)))
                                                for k, t in taus.items()}}
        if o == origins[0]:                                      # P9 series for the origin-1 models (S10-12)
            mon_idx = [k for k in range(D["I1"]["n"]) if D["I1"]["month"][k] in P9_MONTHS
                       and D["I1"]["fam"][k] in Ks and D["I1"]["sha"][k] in te1]
            maxj, same = _maxj(u_sets, u_sigs, u_shas, D["I1"], mon_idx, tau_min)
            monthly, mmaxj, msame = {}, {}, {}
            for m in P9_MONTHS:
                sel = [q for q, k in enumerate(mon_idx) if D["I1"]["month"][k] == m]
                ks = [mon_idx[q] for q in sel]
                monthly[m] = (np.array([fi[D["I1"]["fam"][k]] for k in ks], dtype=np.int64),
                              [encs["I1"]("I1", k) for k in ks])
                mmaxj[m], msame[m] = maxj[sel], same[sel]
            c1 = cells[f"I1_o{o}"]
            c1.monthly, c1.monthly_maxj, c1.monthly_same = monthly, mmaxj, msame
            info["P9_months"] = {m: {"all": int(len(v[0])), "kept": {k: int(len(audit_keep(mmaxj[m], msame[m], t)))
                                                                     for k, t in taus.items()}}
                                 for m, v in monthly.items()}
        info["origins"][o] = {"families": K, "tau_J_registered": tau, "tau_rule": tau_info, "taus": taus,
                              "union_traces": len(u), "nn_search": nst, "nn_profile": nn_profile(nn),
                              "test_keep": kinfo, "seconds": round(time.perf_counter() - t1, 1)}
        log(f"{time.strftime('%H:%M:%S')} s10 assemble origin {o}: registered tau_J {tau} ({tau_info.get('rule')}); "
            + "; ".join(f"{n} kept " + ", ".join(f"{k} {c:,}" for k, c in v["kept"].items())
                        + f" of {v['in_families']:,}" for n, v in kinfo.items()))
    info["seconds"] = round(time.perf_counter() - t0, 1)
    info["peak_rss_mb"] = round(R9.rss_mb()[1], 1)
    info["format"] = ASSEMBLY_FORMAT                         # an assembly of an earlier engine is rebuilt, not reused
    A = Assembly(sets, cells, groups, info)
    A.digest = assembly_digest(A)
    return A


# ------------------------------------------------------------------------------------------------
# Content digest and storage
# ------------------------------------------------------------------------------------------------
def assembly_digest(A: Assembly) -> str:
    """SHA-256 over every array and label of the test sets and cells, in a fixed order (pickle-independent)."""
    h = hashlib.sha256()

    def put(x):
        if isinstance(x, np.ndarray):
            if x.dtype == object:
                put([str(v) for v in x.tolist()])
            else:
                h.update(str(x.dtype).encode() + str(x.shape).encode())
                h.update(np.ascontiguousarray(x).tobytes())
        else:
            h.update(json.dumps(x, sort_keys=True, default=str).encode())

    for name in sorted(A.sets):
        s = A.sets[name]
        for x in (name, s.family, s.cluster, s.month if s.month is not None else "", s.machine
                  if s.machine is not None else ""):
            put(x)
        for key in sorted(s.clusters or {}):
            put(key)
            put(np.asarray(s.clusters[key]))
    for ck in sorted(A.cells):
        c = A.cells[ck]
        put([ck, c.inst, c.origin, list(c.families), c.V, c.own, c.other, c.tau_registered])
        put(np.asarray(c.own_idx))
        put(np.asarray(c.other_idx))
        for arr in (c.own_maxj, c.own_same, c.other_maxj, c.other_same):
            put(np.asarray(arr) if arr is not None else "none")
        for m in sorted(c.monthly_maxj or {}):
            put(m)
            put(np.asarray(c.monthly_maxj[m]))
            put(np.asarray(c.monthly_same[m]))
        for part in (c.train, c.val, c.own_traces, c.other_traces):
            put(len(part))
            for t in part:
                put(t.ids)
                put(t.betas)
                put(t.cls)
                put([int(t.y), t.length])
        for m, (y, trs) in sorted((c.monthly or {}).items()):
            put(m)
            put(np.asarray(y))
            for t in trs:
                put(t.ids)
                put(t.betas)
                put(t.cls)
    put(A.groups)
    put(A.info.get("format", ""))
    return h.hexdigest()


def save_assembly(A: Assembly, path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(A, f, protocol=5)
    tmp.replace(path)
    return sha256_file(path)


def load_assembly(path: Path, expect_digest: str | None = None) -> Assembly:
    with open(path, "rb") as f:
        A = pickle.load(f)
    if expect_digest is not None and A.digest != expect_digest:
        raise SystemExit(f"{path} holds assembly {A.digest[:12]}..., not the {expect_digest[:12]}... recorded in the "
                         "ledger when scoring began")
    return A
