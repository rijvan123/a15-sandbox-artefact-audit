"""Label checks for kill criterion K4 (S6, S7; amendment 3 of 4 November 2026).

K4 (S6): an installation's family labels must agree with an independent labeller, EMBER2024's ClarAVy families, on at
least 80% of shared SHA-256s; families that do not are dropped and the limitation is stated. EMBER2024 publishes no
labels-only file, so `scan_ember_archives` reads its Windows feature archives without unpacking them and keeps only the
scalar fields (hashes, label, family, confidence, dates) of SHA-256s that also occur in MELD-DS-448 or WinMET. Feature
values are never kept.

Two checks that need no EMBER2024 data are reported beside it: agreement between the installations on binaries they
share, and agreement of each installation's label with CAPE's own detections. CAPE detections come from the same
sandbox run as the trace, so that comparison checks where a label comes from (S6 charge 5); it is not independent.
"""
from __future__ import annotations

import gzip
import json
import re
import time
import zipfile
from collections import defaultdict
from pathlib import Path

from .config import K4_CLARAVY_MIN_CONF, K4_MIN_AGREEMENT, K4_MIN_SHARED
from .splits import norm_family

SHA_RE = re.compile(rb'"sha256"\s*:\s*"([0-9a-fA-F]{64})"')
NAN = float("nan")


def agreement_any(pairs) -> dict:
    """pairs: (label, [families named by a second labeller]). A pair agrees if the label is among those families.
    Only pairs where both sides name a family count; the others are reported as missing."""
    both = agree = missing = 0
    for lab, others in pairs:
        others = [o for o in others if o]
        if not lab or not others:
            missing += 1
            continue
        both += 1
        agree += lab in others
    return {"n_both_named": both, "n_missing_label": missing, "agreement": agree / both if both else NAN}


def per_family(pairs, families) -> dict:
    """Agreement among traces that carry each family label (both sides named), for the listed families."""
    d = defaultdict(lambda: [0, 0])
    for lab, others in pairs:
        others = [o for o in others if o]
        if lab in families and others:
            d[lab][0] += 1
            d[lab][1] += lab in others
    return {f: {"n": d[f][0], "agreement": d[f][1] / d[f][0] if d[f][0] else NAN} for f in sorted(families)}


def claravy_family(rec: dict) -> tuple[str, float | None]:
    """Family name and confidence of one EMBER2024 record. The field names are looked up, not assumed: 'family' if
    present, else the first field whose name contains 'family' but not 'conf'; confidence from a field with both."""
    keys = list(rec)
    fam_key = "family" if "family" in rec else next((k for k in keys if "family" in k.lower()
                                                     and "conf" not in k.lower()), None)
    conf_key = next((k for k in keys if "family" in k.lower() and "conf" in k.lower()), None)
    fam = rec.get(fam_key) if fam_key else None
    conf = rec.get(conf_key) if conf_key else None
    return ("" if fam in (None, "") else str(fam)), (float(conf) if isinstance(conf, (int, float)) else None)


def k4_claravy(ember: dict, inst_labels: dict, shared_families, aliases: dict, min_shared: int = K4_MIN_SHARED,
               threshold: float = K4_MIN_AGREEMENT, min_conf: float = K4_CLARAVY_MIN_CONF) -> dict:
    """ClarAVy agreement per installation and per shared family, and the S6 rule.

    inst_labels: {installation: {sha256: family}} with the alias table already applied; ember: {sha256: record}.
    The rule uses ClarAVy labels with family_confidence >= min_conf (all labels are reported beside them). A shared
    family is dropped if, in an installation where it has at least `min_shared` such SHA-256s, agreement is below
    `threshold`; with fewer in every installation it stays 'unverified' (kept, stated as a limitation)."""
    shared_families = sorted(set(shared_families))
    out: dict = {"ember_records_matched": len(ember), "min_shared": min_shared, "threshold": threshold,
                 "min_confidence": min_conf}
    per: dict = {}
    for inst, labs in inst_labels.items():
        rows, benign = [], 0
        for s, fam in labs.items():
            if s in ember:
                cf, conf = claravy_family(ember[s])
                cf = norm_family(cf)
                rows.append((fam, aliases.get(cf, cf), conf))
                benign += ember[s].get("label") == 0
        named = [(f, c, conf) for f, c, conf in rows if f and c]
        sure = [(f, c) for f, c, conf in named if conf is not None and conf >= min_conf]
        out[inst] = {"shared_sha256": len(rows), "benign_in_ember2024": benign, "both_named": len(named),
                     "agreement_all_labels": sum(f == c for f, c, _ in named) / len(named) if named else NAN,
                     "both_named_confident": len(sure),
                     "agreement": sum(f == c for f, c in sure) / len(sure) if sure else NAN}
        per[inst] = {}
        for fam in shared_families:
            sub_all = [(f, c) for f, c, _ in named if f == fam]
            sub = [(f, c) for f, c in sure if f == fam]
            per[inst][fam] = {"n": len(sub), "agreement": sum(f == c for f, c in sub) / len(sub) if sub else NAN,
                              "n_all_labels": len(sub_all), "agreement_all_labels":
                                  sum(f == c for f, c in sub_all) / len(sub_all) if sub_all else NAN,
                              "claravy_names_when_disagreeing": dict(sorted(
                                  _count(c for f, c in sub_all if f != c).items(), key=lambda kv: -kv[1])[:5])}
    dropped, unverified, verified = [], [], []
    for fam in shared_families:
        judged = [per[i][fam] for i in per if per[i][fam]["n"] >= min_shared]
        if not judged:
            unverified.append(fam)
        elif any(j["agreement"] < threshold for j in judged):
            dropped.append(fam)
        else:
            verified.append(fam)
    out["per_family"] = per
    out["rule"] = {"dropped": dropped, "unverified": unverified, "verified": verified,
                   "statement": "families dropped under K4 (S6); unverified families are kept and stated as a limitation"}
    return out


def _count(items) -> dict:
    d: dict = defaultdict(int)
    for x in items:
        d[x] += 1
    return d


def _lines(z: zipfile.ZipFile, info: zipfile.ZipInfo):
    raw = z.open(info)
    fh = gzip.GzipFile(fileobj=raw) if info.filename.endswith(".gz") else raw
    try:
        yield from fh
    finally:
        fh.close()
        raw.close()


def scan_ember_archives(archives, wanted: set, label: str = "", every: int = 200_000) -> tuple[dict, dict]:
    """Stream EMBER2024 zip archives (JSON-lines members, optionally gzip-compressed) without unpacking them and keep
    the scalar fields of every record whose SHA-256 is in `wanted` (lower case). Returns (records, per-archive stats)."""
    found: dict = {}
    stats: dict = {}
    t0, total = time.perf_counter(), 0
    for a in archives:
        a = Path(a)
        n = hits = dup = members = 0
        with zipfile.ZipFile(a) as z:
            for info in z.infolist():
                name = info.filename.lower()
                if info.is_dir() or not name.endswith((".jsonl", ".json", ".jsonl.gz", ".json.gz")):
                    continue
                members += 1
                for line in _lines(z, info):
                    n += 1
                    total += 1
                    m = SHA_RE.search(line[:4096]) or SHA_RE.search(line)
                    if m:
                        s = m.group(1).decode().lower()
                        if s in wanted:
                            if s in found:
                                dup += 1
                            else:
                                rec = json.loads(line)
                                found[s] = {k: v for k, v in rec.items()
                                            if v is None or isinstance(v, (str, int, float, bool))}
                                found[s]["_archive"] = a.name
                                hits += 1
                    if label and total % every == 0:
                        print(f"{time.strftime('%H:%M:%S')} {label}: {total:,} records read in "
                              f"{time.perf_counter() - t0:,.0f} s; {len(found):,} shared SHA-256 so far", flush=True)
        stats[a.name] = {"members": members, "records": n, "shared_sha256": hits, "repeats": dup}
    return found, stats
