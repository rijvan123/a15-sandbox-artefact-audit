#!/usr/bin/env python
"""S9 driver for the A15 audit: acquisition, sealing, field audit, harmonisation, lexicon, overlap (P3), counts
(K3, N_req), label check (K4), dry run (K5, K6, P1, P4), freeze preparation and freeze.

Sandbox:  python run_s9.py --root WORK selftest --reports-archive malvada/test_reports/test_reports.7z
Cowork:   python run_s9.py --root WORK acquire | ingest --inst I1 | ingest --inst I2 | audit | harmonise |
          lexicon | overlap | dryrun | freeze-prep | freeze      (one stage per Cowork task, in this order)
No stage opens a test window before 'freeze': the dry run uses the pseudo-origin inside every training period.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a15 import config as C                                                    # noqa: E402
from a15 import dedup, gates, lexicon, overlap, splits, stats, tokens          # noqa: E402
from a15.acquire import ProvenanceLog, ProvRow, acquire_all, git_pinned, probe, record_blocked   # noqa: E402
from a15.cape import parse_report, stream_7z, volumes, ConcatReader, curated_out                 # noqa: E402
from a15.ledger import Ledger, TimeStamper, enc, sha256_file, timed                              # noqa: E402

HOSTS = {"I1": "https://huggingface.co/api/datasets/MeldProject/MELD-DS-448",
         "I2": "https://zenodo.org/api/records/12647555", "I3": "https://drive.google.com/",
         "M": "https://huggingface.co/api/datasets/joyce8/EMBER2024", "TSA": "https://freetsa.org/tsr"}


def jdump(path: Path, obj) -> str:
    blob = json.dumps(obj, indent=1, sort_keys=True, default=str)
    Path(path).write_text(blob)
    return hashlib.sha256(blob.encode()).hexdigest()


def open_ledger(P: C.Paths, tsa_url: str | None) -> Ledger:
    stamper = TimeStamper(P.work / "tsa", tsa_url=tsa_url, ca_file=os.environ.get("A15_TSA_CA"),
                          untrusted=os.environ.get("A15_TSA_CERT"))
    return Ledger(P.ledger / "ledger.jsonl", C.STUDY_ID, stamper)


def custodian_key() -> bytes:
    k = os.environ.get("A15_CUSTODIAN_KEY", "")
    return k.encode() if k else hashlib.sha256(b"A15-development-custodian-key").digest()


def rss_mb() -> tuple[float, float]:
    """Current and peak resident memory of this process in MB (Linux; zeros where unavailable)."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        with open("/proc/self/statm") as f:
            cur = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 20
        return cur, peak
    except Exception:
        return 0.0, 0.0


# ------------------------------------------------------------------------------------------------
# Ingest: stream an archive, seal every report (Algorithm 1 steps 2-5), keep class-level records only
# ------------------------------------------------------------------------------------------------
def ingest(archive, inst: str, password: str | None, ledger: Ledger, out_path: Path, limit: int | None = None) -> dict:
    key, n_seen, n_cur, t0 = custodian_key(), 0, 0, time.perf_counter()
    fh = gzip.open(out_path, "wt", encoding="utf-8")

    def handle(name: str, raw: bytes):
        nonlocal n_seen, n_cur
        if limit and n_seen >= limit:
            return
        n_seen += 1
        rep = parse_report(raw, inst, key, name)
        x_canon = json.dumps([[c[3], c[4], c[5], c[6], c[7]] for c in rep.calls], separators=(",", ":"),
                             default=str).encode()
        d_x = hashlib.sha256(x_canon).hexdigest()                 # digest of the raw trace record (never stored)
        d_e = hashlib.sha256(enc({k: str(v) for k, v in rep.env.items()})).hexdigest()
        pre = []                                                   # class-level calls: (api|kappa|ret, n)
        for _, _, _, api, args, ok, ret, n in rep.calls:
            kap = ";".join(f"{s}:{c}" for s, c in tokens.arg_classes(api, args, rep.self_paths))
            pre.append([api, kap, tokens.ret_class(ok, ret), n])
        cur = curated_out(rep)
        n_cur += cur
        ledger.append("seal", evidence_id=f"{inst}:{rep.sha256}:{rep.report_sha256[:16]}", d_f=rep.report_sha256,
                      d_x=d_x, d_e=d_e, curated_out=int(cur))
        fh.write(json.dumps({"inst": inst, "sha256": rep.sha256, "report_sha256": rep.report_sha256, "d_x": d_x,
                             "d_e": d_e, "env": rep.env, "env_prov": rep.env_provenance, "labels": rep.labels,
                             "time_keys": rep.time_keys, "length": rep.length, "curation": rep.curation,
                             "curated_out": int(cur), "calls": pre}) + "\n")
        if n_seen % 500 == 0:                                      # progress and memory, visible through tee
            now, peak = rss_mb()
            print(f"{time.strftime('%H:%M:%S')} {inst}: {n_seen:,} reports sealed; memory {now:,.0f} MB "
                  f"(peak {peak:,.0f} MB)", flush=True)

    if isinstance(archive, list):
        n = stream_7z(ConcatReader(archive), handle, password)
    else:
        n = stream_7z(archive, handle, password)
    fh.close()
    return {"members": n, "parsed": n_seen, "curated_out": n_cur, "seconds": time.perf_counter() - t0,
            "peak_rss_mb": rss_mb()[1]}


def load_records(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def field_audit(recs: list[dict]) -> dict:
    n = len(recs)
    env = {k: sum(r["env_prov"].get(k) == "report" for r in recs) / n for k in recs[0]["env_prov"]} if n else {}
    tk = {k: sum(bool(r["time_keys"].get(k)) for r in recs) / n for k in ("vt_scan_date", "analysis_date")}
    lab = {"cape": sum(bool(r["labels"]["cape"]) for r in recs) / n,
           "avclass": sum(bool(r["labels"]["avclass"]) for r in recs) / n}
    cur = Counter(k for r in recs for k, v in r["curation"].items() if v)
    calls = np.array([r["length"]["n_calls"] for r in recs])
    return {"n": n, "env_field_share": env, "time_key_share": tk, "label_share": lab, "curation_counts": dict(cur),
            "curated_out": sum(r["curated_out"] for r in recs), "early_exit_share": float(np.mean(
                [r["length"]["early_exit"] for r in recs])), "calls_quantiles": np.percentile(calls, [5, 25, 50, 75,
                                                                                                    95]).tolist()}


def sequences(recs: list[dict], hook_set: set | None) -> list[list[tuple[str, int]]]:
    out = []
    for r in recs:
        toks = [(f"{a}|{k}|{t}" if (hook_set is None or a in hook_set) else C.OOS, n) for a, k, t, n in r["calls"]]
        out.append(tokens.collapse(toks))
    return out


# ------------------------------------------------------------------------------------------------
# Self-test on the 200 MALVADA reports (installation code T: same laboratory as WinMET)
# ------------------------------------------------------------------------------------------------
def selftest(args, P: C.Paths) -> None:
    import torch
    from a15 import models as M
    rep: dict = {"stage": "S9 self-test", "measured_on": "sandbox (1 CPU, 3 GB RAM)", "python": platform.python_version(),
                 "torch": torch.__version__}
    ledger = open_ledger(P, None)
    prov = ProvenanceLog(P.out / "data_provenance_log.csv")

    # 1. reachability of the primary hosts from this environment; blocked hosts become substitution entries
    probes = {k: probe(u) for k, u in HOSTS.items()}
    rep["host_probes"] = probes
    for code in ("I1", "I2", "I3", "M"):
        if not probes[code]["reachable"]:
            record_blocked(C.SOURCES[code], prov, ledger, probes[code], "claude.ai sandbox")
    if not probes["TSA"]["reachable"]:
        ledger.append("pending_anchor_policy", reason="TSA unreachable from sandbox: local development TSA")

    # 2. MALVADA pinned and its test archive verified against the published MD5 (test_reports/README.md)
    src = C.SOURCES["MALVADA"]
    mal = Path(args.malvada_dir)
    head = os.popen(f"git -C {mal} rev-parse HEAD").read().strip()
    prov.add(ProvRow("MALVADA", src.name, "(git tree)", src.locator, head, src.licence, status="ok",
                     digest_match=str(head == src.revision), note="git commit is the content digest"))
    ledger.append("acquire", evidence_id="MALVADA:git", d_f=head, source="MALVADA")
    arc = Path(args.reports_archive)
    sha = sha256_file(arc)
    md5 = hashlib.md5(arc.read_bytes()).hexdigest()
    ok = md5 == "6daf269f52c9192a878f0840ac89df5e"
    prov.add(ProvRow("T", "MALVADA test reports (200 raw CAPE JSON)", "test_reports/test_reports.7z", src.locator,
                     head, src.licence, arc.stat().st_size, sha, "6daf269f52c9192a878f0840ac89df5e", "md5", str(ok),
                     status="ok" if ok else "quarantine", note="published digest in test_reports/README.md"))
    ledger.append("acquire", evidence_id="T:test_reports.7z", d_f=sha, published_match=int(ok))

    # 3. stream, parse, seal: the archive is never unpacked to disk
    recs_path = P.work / "T_records.jsonl.gz"
    rep["ingest"] = ingest(arc, "T", None, ledger, recs_path)
    recs = [r for r in load_records(recs_path)]
    rep["field_audit"] = field_audit(recs)
    kept = [r for r in recs if not r["curated_out"]]

    # 4. tokens, sequences, length rule, vocabulary and lexicon classes (single installation: H_c = observed APIs)
    hook = {a for r in kept for a, _, _, _ in r["calls"]}
    seqs = sequences(kept, hook)
    L = tokens.choose_L([len(s) for s in seqs])
    vocab = tokens.Vocab(seqs)
    occ = Counter()
    rule_occ = Counter()
    for s in seqs:
        for t, _ in s:
            c, rid = lexicon.classify(t)
            occ[c] += 1
            rule_occ[rid] += 1
    tot = sum(occ.values())
    alt = Counter(lexicon.classify_alt(t) for s in seqs for t, _ in s)
    rep["tokens"] = {"traces_kept": len(kept), "hook_set_size": len(hook), "L": L, "vocab_size": len(vocab),
                     "collapsed_length_quantiles": np.percentile([len(s) for s in seqs], [5, 50, 95]).tolist(),
                     "class_share_primary": {k: occ[k] / tot for k in "BFP"},
                     "class_share_boundary_alt": {k: alt[k] / tot for k in "BFP"},
                     "rule_firing_share": {k: v / tot for k, v in rule_occ.most_common()}}
    lex_doc = lexicon.lexicon_document([t for s in seqs for t, _ in s])
    lex_sha = jdump(P.work / "lexicon_T_selftest.json", lex_doc)
    ledger.append("lexicon_seal", evidence_id="lexicon:selftest", d_f=lex_sha, version=lexicon.LEXICON_VERSION,
                  note="sealed before any overlap statistic is computed")

    # 5. privacy release gate on everything that would leave the project (vocabulary and token table)
    rep["privacy_scan"] = gates.residual_data_scan(vocab.itos + list(lex_doc["token_table"]), len(kept))

    # 6. near-duplicates: API 3-gram sets, MinHash, nearest-neighbour similarities, tau_J rule
    apis = [[a if a in hook else C.OOS for a, _, _, _ in r["calls"]] for r in kept]
    sets_ = [dedup.api_3grams(a) for a in apis]
    mh = dedup.MinHasher(128)
    sigs = np.array([mh.signature(s) for s in sets_])
    nn = dedup.nn_similarities(sets_, sigs)
    tau, tau_info = dedup.antimode_tau(nn)
    exact_pairs = sum(1 for i, j in dedup.candidates_within(sigs, 32, 4) if dedup.jaccard(sets_[i], sets_[j]) == 1.0)
    rep["dedup"] = {"tau_J": tau, "rule": tau_info, "nn_similarity_quantiles": np.percentile(nn, [25, 50, 75, 95]).tolist(),
                    "share_nn_ge_0.8": float(np.mean(nn >= 0.8)), "exact_duplicate_pairs": exact_pairs}
    np.save(P.work / "nn_similarities_T.npy", nn)

    # 7. learners: smoke training (not an evaluation), P1 and completeness of IG, timing for K6
    fam = [splits.norm_family(r["labels"]["avclass"]) for r in kept]
    cnt = Counter(f for f in fam if f)
    top = [f for f, n in cnt.most_common() if n >= 8][:6]
    if len(top) < 2:
        top = [f for f, _ in cnt.most_common(2)]
    fidx = {f: i for i, f in enumerate(top)}
    data = []
    for s, f, r in zip(seqs, fam, kept):
        if f in fidx:
            s = tokens.trunc_head_tail(s, L)
            data.append(M.Trace(np.array([vocab.encode(t) for t, _ in s], dtype=np.int64),
                                np.array([b for _, b in s], dtype=np.int64),
                                np.array(["BFP".index(lexicon.classify(t)[0]) for t, _ in s], dtype=np.int64),
                                fidx[f], r["length"]))
    rng = random.Random(7)
    rng.shuffle(data)
    cut = int(0.8 * len(data))
    tr, va = data[:cut], data[cut:]
    K, V = len(top), len(vocab)
    smoke, p1, ig_delta, timing = {}, {}, {}, {}
    small = {"transformer": {"layers": 2, "ff": 256, "dropout": 0.1, "lr": 1e-3, "wd": 0.01, "gamma": 0.5, "batch": 16},
             "bilstm": {"layers": 1, "hidden": 64, "dropout": 0.1, "lr": 3e-3, "wd": 0.01, "gamma": 0.5, "batch": 16},
             "cnn": {"channels": 64, "dropout": 0.1, "lr": 3e-3, "wd": 0.01, "gamma": 0.5, "batch": 16}}
    for kind, cfg in small.items():
        model, info = M.train_deep(kind, V, K, tr, va, cfg, seed=C.SEEDS[0], p_drop=0.5, val_mask=("F",), epochs=2,
                                   patience=2)
        smoke[kind] = {"train_seconds": info["seconds"], "n_train": len(tr), "n_val": len(va), "K": K}
        zeros, maxabs, deltas = 0, 0.0, []
        for t in va[:10]:
            g = M.integrated_gradients(model, t, m=50, test_mask=("F",))
            masked = t.cls == 1
            maxabs = max(maxabs, float(np.abs(g["ig"][:len(t.ids)][masked]).max()) if masked.any() else 0.0)
            zeros += int(masked.any())
            g2 = M.integrated_gradients(model, t, m=50)
            deltas.append(g2["delta"])
        p1[kind] = {"traces_with_F_positions": zeros, "max_abs_IG_at_masked_positions": maxabs, "P1_holds": maxabs == 0.0}
        pop = M.ig_population(model, va[:20])
        ig_delta[kind + "_escalation"] = {"m_final": pop["m"], "share_delta_ok": pop["share_ok"],
                                          "criterion_met": pop["criterion_met"]}
        ig_delta[kind] = {"median_delta": float(np.median(deltas)), "share_delta_le_0.05": float(np.mean(
            np.array(deltas) <= C.IG_DELTA_MAX))}
        timing[kind] = M.time_per_trace(kind, V, K, L, cfg, n=16, reps=2)
    lg = M.NgramLGBM(K, {"num_leaves": 15, "n_estimators": 100, "learning_rate": 0.1, "min_child_samples": 5,
                         "gamma": 0.5}, C.SEEDS[0])
    (_, t_lgb) = timed(lg.fit, tr, [])
    timing["lgbm_fit_seconds_100_rounds"] = t_lgb
    rep["learners"] = {"smoke_training": smoke, "P1": p1, "IG_completeness": ig_delta,
                       "seconds_per_trace_fwd_bwd_at_L": timing, "families_used": top,
                       "note": "random split of 200 reports; training only exercises code paths, it is not an evaluation"}
    sec = {k: v for k, v in timing.items() if k in ("transformer", "bilstm", "cnn")}
    rep["K6_projection_estimated"] = gates.k6_projection(sec, {"I1": args.plan_n_train_i1, "I2": args.plan_n_train_i2},
                                                         origins={"I1": 3, "I2": 1})

    # 8. ledger: anchor, verify, tamper evidence (P10), freeze ordering (P11), cost per item (P12)
    anc = ledger.anchor()
    ok, msg = ledger.verify()
    rep["ledger"] = {"entries": len(ledger.records), "verify": ok, "message": msg, "anchor_tsa": anc["entry"]["tsa"],
                     "anchor_trusted": bool(anc["entry"]["trusted"])}
    rep["ledger_tests"] = ledger_tests(P)
    jdump(P.out / "s9_selftest_report.json", rep)
    print(json.dumps({k: rep[k] for k in ("ingest", "tokens", "dedup", "privacy_scan", "ledger")}, indent=1,
                     default=str)[:4000])


def ledger_tests(P: C.Paths, n_edits: int = 1000) -> dict:
    """P10-P12 on a scratch ledger (the study ledger is never edited)."""
    scratch = P.work / "scratch_ledger"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    led = Ledger(scratch / "l.jsonl", C.STUDY_ID, TimeStamper(P.work / "tsa"))
    t0 = time.perf_counter()
    for i in range(2000):
        led.append("seal", evidence_id=f"T:{i}", d_f=hashlib.sha256(str(i).encode()).hexdigest())
        if i == 999:
            led.append("freeze", payload={"lexicon.json": "x"})
            led.anchor()
    seal_us = 1e6 * (time.perf_counter() - t0) / 2000
    led.append("test_score", note="first scoring entry")
    led.anchor()
    t0 = time.perf_counter()
    ok, _ = led.verify(check_tokens=False)
    verify_us = 1e6 * (time.perf_counter() - t0) / len(led.records)
    vfrz = led.v_frz()
    rng = random.Random(3)
    recs0 = [json.loads(x) for x in (scratch / "l.jsonl").read_text().splitlines()]
    detected = 0
    for _ in range(n_edits):                                   # single-field edits inside anchored batches
        k = rng.randrange(0, 1999)
        recs = list(recs0)
        recs[k] = json.loads(json.dumps(recs0[k]))
        e = recs[k]["entry"]
        field = rng.choice([f for f in e if f not in ("i",)])
        e[field] = str(e[field]) + "x"
        tmp = Ledger.__new__(Ledger)
        tmp.records, tmp.h0, tmp.stamper = recs, led.h0, led.stamper
        detected += not tmp.verify(check_tokens=False)[0]
    neg = Ledger(scratch / "neg.jsonl", C.STUDY_ID, TimeStamper(P.work / "tsa"))
    neg.append("freeze", payload={"lexicon.json": "x"})
    neg.append("test_score")
    neg.anchor()
    return {"P10_edits": n_edits, "P10_detected": detected, "P11_vfrz_study": vfrz, "P11_vfrz_negative": neg.v_frz(),
            "P12_seal_us_per_entry": seal_us, "P12_verify_us_per_entry": verify_us, "chain_ok": ok}


# ------------------------------------------------------------------------------------------------
# Cowork stages (same functions; inputs are the real installations)
# ------------------------------------------------------------------------------------------------
def stage_acquire(args, P):
    ledger, prov = open_ledger(P, args.tsa), ProvenanceLog(P.out / "data_provenance_log.csv")
    git_pinned(C.SOURCES["MALVADA"], P.data, prov, ledger)
    status = acquire_all(P.data, prov, ledger, "Cowork", include=("I1", "I2"))
    jdump(P.out / "acquire_status.json", status)
    ledger.anchor()
    print(status)


def stage_ingest(args, P):
    ledger = open_ledger(P, args.tsa)
    if args.inst == "I1":
        parts = volumes(P.data / "I1" / "Dataset", "cape_reports.7z")
        res = ingest(parts, "I1", None, ledger, P.work / "I1_records.jsonl.gz")
    else:
        res = {}
        for vol in sorted((P.data / "I2").glob("WinMET_volume_*.7z")):
            res[vol.name] = ingest(vol, "I2", C.SOURCES["I2"].archive_password, ledger,
                                   P.work / f"I2_records_{vol.stem}.jsonl.gz")
    ledger.anchor()
    jdump(P.out / f"ingest_{args.inst}.json", res)


def _records(P, inst):
    """Every record of an installation at once (self-test scale only; the stages below stream)."""
    out = []
    for f in record_files(P, inst):
        out += load_records(f)
    return out


# ------------------------------------------------------------------------------------------------
# Streaming access to the sealed records (amendment of 3 November 2026). Loading every record at once needs about
# 50 GB for MELD-DS-448 and 175 GB for WinMET, so each stage reads the records one at a time and keeps only what it
# uses. Checked against the earlier, all-in-memory stages on two fixtures (run_notes/streaming_equivalence_2026-11-03)
# and by the amendment tests in tests/test_harness.py.
# ------------------------------------------------------------------------------------------------
PROGRESS_EVERY = 2_000


def record_files(P, inst: str) -> list[Path]:
    return sorted(P.work.glob(f"{inst}_records*.jsonl.gz"))


def iter_records(P, inst: str, label: str = ""):
    """The records of one installation, in the order of _records(), parsed one line at a time."""
    n, t0 = 0, time.perf_counter()
    for path in record_files(P, inst):
        with gzip.open(path, "rb") as f:
            for line in f:
                yield json.loads(line)
                n += 1
                if label and n % PROGRESS_EVERY == 0:
                    now, peak = rss_mb()
                    print(f"{time.strftime('%H:%M:%S')} {label} {inst}: {n:,} records read in "
                          f"{time.perf_counter() - t0:,.0f} s; memory {now:,.0f} MB (peak {peak:,.0f} MB)", flush=True)
    if label:
        print(f"{time.strftime('%H:%M:%S')} {label} {inst}: {n:,} records read in {time.perf_counter() - t0:,.0f} s "
              f"(all)", flush=True)


def without_calls(r: dict) -> dict:
    r.pop("calls", None)
    return r


def run_info(P, stage: str, t0: float, **extra) -> dict:
    """Wall-clock time and peak memory of a stage, written beside its outputs (not part of any frozen file)."""
    info = {"stage": stage, "seconds": round(time.perf_counter() - t0, 1), "peak_rss_mb": round(rss_mb()[1], 1),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
    jdump(P.out / f"runinfo_{stage}.json", info)
    print(f"{time.strftime('%H:%M:%S')} {stage} finished in {info['seconds']:,.0f} s; peak memory "
          f"{info['peak_rss_mb']:,.0f} MB", flush=True)
    return info


def stage_audit(args, P):
    t0 = time.perf_counter()
    res = {inst: field_audit([without_calls(r) for r in iter_records(P, inst, "audit")]) for inst in ("I1", "I2")}
    jdump(P.out / "field_audit.json", res)
    run_info(P, "audit", t0)
    print(json.dumps(res, indent=1, default=str))


def stage_harmonise(args, P):
    """Labels, aliases, time keys and splits; the common hook set from training periods; K3, K4 and N_req."""
    import pandas as pd
    from a15 import labels as LB
    t0 = time.perf_counter()
    man = pd.read_csv(P.data / "I1" / "malicious_dataset_manifest.csv")
    lab1 = dict(zip(man.sha256.str.lower(), man.family.map(splits.norm_family)))
    date1 = dict(zip(man.sha256.str.lower(), pd.to_datetime(man.first_seen)))
    api_id: dict[str, int] = {}

    def api_ids(r) -> np.ndarray:                              # the API names of a trace, as sorted unique ids
        return np.unique(np.fromiter((api_id.setdefault(c[0], len(api_id)) for c in r["calls"]), dtype=np.int64))

    r1 = [{"sha256": r["sha256"], "apis": api_ids(r), "cape": r["labels"]["cape"]}
          for r in iter_records(P, "I1", "harmonise") if not r["curated_out"]]
    mh = dedup.MinHasher(125)
    r2, sets2, sig2 = [], [], []
    for r in iter_records(P, "I2", "harmonise"):
        if r["curated_out"]:
            continue
        g = dedup.api_3grams([a for a, _, _, _ in r["calls"]])   # used only by the grouped rule (decided below)
        sig2.append(mh.signature(g))
        sets2.append(g.astype(np.uint32))                      # 3-gram ids are below 2^31
        r2.append({"sha256": r["sha256"], "labels": r["labels"], "time_keys": r["time_keys"], "apis": api_ids(r)})
    names = np.array(sorted(api_id, key=api_id.get), dtype=object)

    def api_union(recs, mask=None) -> set:
        ids = [r["apis"] for k, r in enumerate(recs) if mask is None or mask[k]]
        return set(names[np.unique(np.concatenate(ids))].tolist()) if ids else set()

    fam1 = np.array([lab1.get(r["sha256"], "") for r in r1])
    fam2 = np.array([splits.norm_family(r["labels"]["avclass"]) for r in r2])
    cand = splits.alias_candidates(set(fam1) - {""}, set(fam2) - {""})
    alias_path = P.out / "alias_table.json"
    alias = json.loads(alias_path.read_text()) if alias_path.exists() else {}
    al = lambda n: alias.get(n, n)                             # noqa: E731
    fam1, fam2 = np.array(splits.apply_aliases(fam1, alias)), np.array(splits.apply_aliases(fam2, alias))
    d1 = pd.Series([date1.get(r["sha256"]) for r in r1])
    om1 = splits.origin_masks(d1)
    scan = pd.to_datetime(pd.Series([r["time_keys"]["vt_scan_date"] for r in r2]), errors="coerce")
    share = float(scan.notna().mean())
    om2 = splits.origin_masks(scan)
    per_origin = {o: int((splits.family_counts(fam2, om2[o]).n_test >= C.K_MIN_TEST).sum()) for o in om2}
    rule = splits.winmet_split_rule(share, per_origin)
    out = {"I2_time_key_share": share, "I2_split_rule": rule, "alias_candidates_for_review": cand,
           "alias_table_entries": len(alias)}
    fam_sets, c1s, c2s = {}, {}, {}
    for o in om1:
        c1 = splits.family_counts(fam1, om1[o])
        if rule == "rolling":
            c2 = splits.family_counts(fam2, om2[o])
        else:
            c2 = pd.DataFrame({"n_train_period": pd.Series(fam2).value_counts(),
                               "n_test": (pd.Series(fam2).value_counts() * 0.2).round()}).fillna(0)
        fam_sets[o], c1s[o], c2s[o] = splits.shared_families(c1, c2), c1, c2
    out["N_req_planning"] = stats.n_required()
    shared_sha = set(r["sha256"] for r in r1) & set(r["sha256"] for r in r2)
    by_fam = Counter(lab1.get(s, "") for s in shared_sha)
    out["paired_same_binary"] = gates.paired_test_rule(dict(by_fam))
    first_av: dict[str, str] = {}
    for r in r2:                                               # label of the first retained WinMET trace of a binary
        first_av.setdefault(r["sha256"], r["labels"]["avclass"])
    pairs = [(lab1.get(s, ""), splits.norm_family(first_av[s])) for s in shared_sha]
    out["K4_cross_installation"] = gates.k4_label_agreement(pairs)
    cape_av = [(splits.norm_family((r["labels"]["cape"] or [""])[0]), splits.norm_family(r["labels"]["avclass"]))
               for r in r2 if r["labels"]["cape"]]
    out["K4_I2_cape_vs_avclass"] = gates.k4_label_agreement(cape_av)
    # Amendment 3 (4 Nov 2026): K4 with the alias table applied; CAPE comparisons that count a match among all CAPE
    # families where both sides name one; MELD-DS-448 labels against their own CAPE detections (label source, S6
    # charge 5); and the pre-registered ClarAVy check when `ember-labels` has written work/ember2024_labels.json.
    fam_union = sorted(set().union(*map(set, fam_sets.values()))) if fam_sets else []
    out["K4_cross_installation_aliased"] = gates.k4_label_agreement(
        [(al(lab1.get(s, "")), al(splits.norm_family(first_av[s]))) for s in shared_sha])
    p2 = [(f, [al(splits.norm_family(c)) for c in r["labels"]["cape"]]) for f, r in zip(fam2, r2)]
    out["K4_I2_avclass_in_cape"] = {**LB.agreement_any(p2), "per_shared_family": LB.per_family(p2, fam_union)}
    p1 = [(f, [al(splits.norm_family(c)) for c in r["cape"]]) for f, r in zip(fam1, r1)]
    out["K4_I1_meld_in_cape"] = {**LB.agreement_any(p1), "per_shared_family": LB.per_family(p1, fam_union),
                                 "note": "CAPE detections come from the same sandbox run; agreement near 100% would "
                                         "suggest the MELD-DS-448 labels are sandbox-derived (S6 charge 5)"}
    emb_path = P.work / "ember2024_labels.json"
    dropped: set = set()
    if emb_path.exists():
        ember = json.loads(emb_path.read_text())
        lab_by_inst = {"I1": {r["sha256"]: f for r, f in zip(r1, fam1)}, "I2": {}}
        for r, f in zip(r2, fam2):
            lab_by_inst["I2"].setdefault(r["sha256"], f)
        k4c = LB.k4_claravy(ember, lab_by_inst, fam_union, alias)
        out["K4_claravy"] = k4c
        dropped = set(k4c["rule"]["dropped"])
        out["K4_triggered"] = bool(dropped)
    else:
        out["K4_claravy"] = {"status": "not run: work/ember2024_labels.json is absent (run the ember-labels stage)"}
    for o in om1:
        before = fam_sets[o]
        fam_sets[o] = [f for f in before if f not in dropped]  # S6 K4 action: keep only agreeing families
        out[f"origin{o}"] = {**splits.k3_status(fam_sets[o]),
                             "I1_test_traces_shared": int(c1s[o].loc[fam_sets[o], "n_test"].sum()) if fam_sets[o]
                             else 0}
        if dropped:
            out[f"origin{o}"]["n_shared_before_K4"] = len(before)
    train1 = api_union(r1, om1[1]["train"] | om1[1]["val"])
    train2 = api_union(r2) if rule == "grouped" else api_union(r2, om2[1]["train"] | om2[1]["val"])
    hook = tokens.common_hook_set({"I1": train1, "I2": train2})
    jdump(P.work / "hook_set.json", sorted(hook))
    for o, m in om1.items():                                   # split manifests (frozen at Algorithm 2)
        jdump(P.work / f"split_I1_o{o}.json", splits.manifest(f"I1_o{o}", {
            part: [r["sha256"] for r, keep in zip(r1, m[part]) if keep] for part in ("train", "val", "test")}))
    diag = {"records_retained": {"I1": len(r1), "I2": len(r2)}, "api_names": len(api_id),
            "hook_set_size": len(hook), "I2_split_rule": rule}
    if rule == "grouped":                                      # clusters at the fallback tau_J = 0.8, documented
        sig = np.array(sig2) if sig2 else np.zeros((0, 125), dtype=np.uint64)
        cstats: dict = {}
        cl, _ = dedup.test_clusters(sets2, sig, list(fam2), list(fam2), C.TAU_J_FALLBACK, stats=cstats)
        g = splits.grouped_split(cl, fam2, seed=C.SEEDS[0])
        jdump(P.work / "split_I2_grouped.json", splits.manifest("I2_grouped", {
            part: [r["sha256"] for r, keep in zip(r2, g[part]) if keep] for part in ("train", "val", "test")}))
        diag["I2_near_duplicate_clusters"] = cstats
        for o in om1:                                          # WinMET test traces of each origin's shared families
            out[f"origin{o}"]["I2_test_traces_shared"] = int(np.isin(fam2[g["test"]], fam_sets[o]).sum())
    else:
        for o, m in om2.items():
            jdump(P.work / f"split_I2_o{o}.json", splits.manifest(f"I2_o{o}", {
                part: [r["sha256"] for r, keep in zip(r2, m[part]) if keep] for part in ("train", "val", "test")}))
            out[f"origin{o}"]["I2_test_traces_shared"] = int(c2s[o].loc[fam_sets[o], "n_test"].sum()) if fam_sets[o] \
                else 0
    jdump(P.work / "family_sets.json", fam_sets)
    jdump(P.out / "harmonise_counts.json", out)
    run_info(P, "harmonise", t0, **diag)
    print(json.dumps({k: v for k, v in out.items() if k != "alias_candidates_for_review"}, indent=1, default=str))
    print(f"alias candidates for the authors' review: {len(cand)}; alias table entries applied: {len(alias)}")


def stage_lexicon(args, P):
    t0 = time.perf_counter()
    hook = set(json.loads((P.work / "hook_set.json").read_text()))
    types: set[str] = set()                                    # every curated and retained trace, as before
    for inst in ("I1", "I2"):
        for r in iter_records(P, inst, "lexicon"):
            types.update(t for t, _ in sequences([r], hook)[0])
    doc = lexicon.lexicon_document(sorted(types))
    sha = jdump(P.work / "lexicon.json", doc)
    open_ledger(P, args.tsa).append("lexicon_seal", evidence_id="lexicon", d_f=sha, version=lexicon.LEXICON_VERSION)
    run_info(P, "lexicon", t0, token_types=len(types), lexicon_sha256=sha)


def stage_overlap(args, P):
    t0 = time.perf_counter()
    led = open_ledger(P, args.tsa)
    if not any(r["entry"]["action"] == "lexicon_seal" for r in led.records):
        raise SystemExit("seal the lexicon first: P3 must not shape the lexicon")
    hook = set(json.loads((P.work / "hook_set.json").read_text()))
    counts = overlap.CountBuilder(("I1", "I2"))
    for inst in ("I1", "I2"):
        for r in iter_records(P, inst, "overlap"):
            if not r["curated_out"]:
                counts.add(inst, [t for t, _ in sequences([r], hook)[0]])
    X, vocab = counts.matrices()
    del counts
    res = overlap.class_overlap_counts(X["I1"], X["I2"], vocab, lambda t: lexicon.classify(t)[0], n_boot=args.n_boot)
    jdump(P.out / "P3_overlap.json", res)
    led.append("gate", evidence_id="P3", payload=res)
    run_info(P, "overlap", t0, token_types=len(vocab), nonzero_counts={k: int(m.nnz) for k, m in X.items()})
    print(json.dumps(res, indent=1, default=str))


def i2_parts(P) -> dict | None:
    """WinMET split parts by SHA-256 from the grouped manifest; a binary listed in the test part is test only."""
    path = P.work / "split_I2_grouped.json"
    if not path.exists():
        return None
    ids = json.loads(path.read_text())["ids"]
    test = set(ids["test"])
    val = set(ids["val"]) - test
    return {"train": set(ids["train"]) - test - val, "val": val, "test": test}


def stage_dryrun(args, P):
    """K5, K6, P1 and P4 on the pseudo-origin (Dec 2024 validation, Jan-Feb 2025 test): no test window is opened.

    Amendment 3 (4 Nov 2026): the dry run uses the shared families of harmonise (family_sets.json) and every MELD-DS-448
    trace of them in the pseudo-origin windows (no 1% thinning unless --fraction < 1), and WinMET traces only from the
    training and validation parts of its split, so no WinMET test trace is used before the freeze. K5 projects the
    half-width with both test sets of each origin (I1 and I2 shared-family test traces); K6 uses the full-design
    training sizes.

    Amendment 4 (5 Nov 2026, before the freeze): the model trains for all requested epochs (10 by default, the
    epochs the K6 projection assumes; the 4 Nov run stopped after 3 of 5 because early stopping fired without a
    validation set), and the K6 input is the mean cost over the search configurations, with the machine state."""
    import pandas as pd
    from a15 import models as M
    t0 = time.perf_counter()
    led = open_ledger(P, args.tsa)
    hook = set(json.loads((P.work / "hook_set.json").read_text()))
    man = pd.read_csv(P.data / "I1" / "malicious_dataset_manifest.csv")
    lab1 = dict(zip(man.sha256.str.lower(), man.family.map(splits.norm_family)))
    date1 = dict(zip(man.sha256.str.lower(), pd.to_datetime(man.first_seen)))
    alias = json.loads((P.out / "alias_table.json").read_text()) if (P.out / "alias_table.json").exists() else {}
    fam_sets = json.loads((P.work / "family_sets.json").read_text())
    cand = set().union(*map(set, fam_sets.values())) if fam_sets else set()
    # pass 1: identities, labels and (for a dated WinMET) time keys of the retained traces
    r1 = [{"sha256": r["sha256"]} for r in iter_records(P, "I1", "dryrun pass 1") if not r["curated_out"]]
    r2 = [{"sha256": r["sha256"], "avclass": r["labels"]["avclass"], "scan": r["time_keys"]["vt_scan_date"]}
          for r in iter_records(P, "I2", "dryrun pass 1") if not r["curated_out"]]
    f1 = np.array(splits.apply_aliases([lab1.get(r["sha256"], "") for r in r1], alias))
    f2 = np.array(splits.apply_aliases([splits.norm_family(r["avclass"]) for r in r2], alias))
    d1 = pd.Series([date1.get(r["sha256"]) for r in r1])
    m1 = splits.dryrun_masks(d1)
    if args.fraction < 1.0:                                    # documented month x family sub-sample (S7)
        keys = pd.DataFrame({"month": d1.dt.strftime("%Y-%m"), "family": f1})
        sub = splits.temporal_subsample(keys, args.fraction, seed=C.SEEDS[0])
    else:
        sub = np.ones(len(r1), dtype=bool)
    tr1 = [i for i in range(len(r1)) if sub[i] and f1[i] in cand and (m1["train"][i] or m1["val"][i])]
    te1 = [i for i in range(len(r1)) if sub[i] and f1[i] in cand and m1["test"][i]]
    parts = i2_parts(P)
    if parts is not None:                                      # grouped WinMET split: training and validation parts
        tr2 = [i for i, r in enumerate(r2) if f2[i] in cand and r["sha256"] in parts["train"]]
        te2 = [i for i, r in enumerate(r2) if f2[i] in cand and r["sha256"] in parts["val"]]
    else:                                                      # dated WinMET: the same pseudo-origin windows
        m2 = splits.dryrun_masks(pd.to_datetime(pd.Series([r["scan"] for r in r2]), errors="coerce"))
        tr2 = [i for i in range(len(r2)) if f2[i] in cand and (m2["train"][i] or m2["val"][i])]
        te2 = [i for i in range(len(r2)) if f2[i] in cand and m2["test"][i]]
    fams = sorted(cand & set(f1[tr1]) & set(f2[tr2]) & set(f1[te1]) & set(f2[te2]))
    fidx = {f: k for k, f in enumerate(fams)}
    # pass 2: collapsed sequences of the selected traces only. L <= L_CAP, so a sequence longer than L_CAP is kept as
    # its first and last L_CAP / 2 elements: trunc_head_tail(stored, L) equals trunc_head_tail(sequence, L).
    ttab: dict[str, int] = {}
    tstr: list[str] = []

    def tid(t: str) -> int:
        k = ttab.get(t)
        if k is None:
            k = ttab[t] = len(tstr)
            tstr.append(t)
        return k

    kept: dict[str, dict] = {}
    for inst, tr, te in (("I1", tr1, te1), ("I2", tr2, te2)):
        tr_set, all_set = set(tr), set(tr) | set(te)
        store, k = {}, -1
        for r in iter_records(P, inst, "dryrun pass 2"):
            if r["curated_out"]:
                continue
            k += 1
            if k not in all_set:
                continue
            s = sequences([r], hook)[0]
            part = tokens.head_tail_store(s, C.L_CAP)
            store[k] = (len(s), frozenset(tid(t) for t, _ in s) if k in tr_set else None,
                        np.array([tid(t) for t, _ in part], dtype=np.int64), np.array([b for _, b in part],
                                                                                     dtype=np.int64), r["length"])
        kept[inst] = store
    L = tokens.choose_L([kept["I1"][i][0] for i in tr1] + [kept["I2"][i][0] for i in tr2])
    if L < 2:
        raise SystemExit(f"L = {L}: the 95th percentile of collapsed training lengths is below 2; check the records")
    vocab = tokens.Vocab([[(tstr[j], 0) for j in kept["I1"][i][1]] for i in tr1] +
                         [[(tstr[j], 0) for j in kept["I2"][i][1]] for i in tr2])

    def mk(inst, fam, idx):
        out = []
        for i in idx:
            if fam[i] in fidx:
                _, _, ids, betas, length = kept[inst][i]
                s = tokens.trunc_head_tail([(tstr[j], b) for j, b in zip(ids.tolist(), betas.tolist())], L)
                out.append(M.Trace(np.array([vocab.encode(t) for t, _ in s]), np.array([b for _, b in s]),
                                   np.array(["BFP".index(lexicon.classify(t)[0]) for t, _ in s]), fidx[fam[i]],
                                   length))
        return out
    A_tr, A_te, B_te = mk("I1", f1, tr1), mk("I1", f1, te1), mk("I2", f2, te2)
    del kept
    cfg = M.sample_configs("transformer", 1, C.SEEDS[0])[0]
    model, info = M.train_deep("transformer", len(vocab), len(fams), A_tr, [], cfg, C.SEEDS[0], epochs=args.epochs)
    yw, pw = np.array([t.y for t in A_te]), M.predict(model, A_te)
    yc, pc = np.array([t.y for t in B_te]), M.predict(model, B_te)
    y = np.concatenate([yw, yc])
    p = np.concatenate([pw, pc])
    which = np.array([0] * len(yw) + [1] * len(yc))
    boot = stats.ClusterBootstrap([(w, k) for w, k in zip(which, y)], list(range(len(y))), n_boot=999)
    K = len(fams)
    gap = lambda idx: (stats.macro_f1(y[idx][which[idx] == 0], p[idx][which[idx] == 0], K)
                       - stats.macro_f1(y[idx][which[idx] == 1], p[idx][which[idx] == 1], K))
    reps = stats.bootstrap_replicates(gap, boot)
    lo, hi = stats.percentile_ci(reps)
    rec = float(np.mean(stats.recall_far(yw, pw, K)[0]))
    n_dry = (len(yw) + len(yc)) / 2
    counts = json.loads((P.out / "harmonise_counts.json").read_text())

    def n_test(o: int) -> float:                               # two test sets of unequal size: 2 / (1/n1 + 1/n2)
        c = counts[f"origin{o}"]
        n1, n2 = c["I1_test_traces_shared"], c.get("I2_test_traces_shared")
        if n2 is None:
            return float(n1)
        return 2.0 / (1.0 / n1 + 1.0 / n2) if n1 > 0 and n2 > 0 else 0.0
    n_full = max(min(n_test(o) for o in (1, 2, 3)), 1e-9)
    k5 = gates.k5_p4((hi - lo) / 2, n_dry, rec, boot.design_effect(), n_full)
    # K6 input. Amendment 4 (5 Nov 2026): the mean cost over the N_SEARCH configurations the search will draw (seed
    # C.SEEDS[0]; the first one is the configuration trained above), each the best of 3 passes over 32 traces at L,
    # with the machine state recorded. The earlier input (first draw of seed 0, 16 traces, best of 2 passes) is kept
    # beside it for comparison with the 3 and 4 November runs.
    import torch
    load_before = os.getloadavg() if hasattr(os, "getloadavg") else None
    timing = {k: M.time_per_trace_configs(k, len(vocab), K, L, M.sample_configs(k, C.N_SEARCH, C.SEEDS[0]))
              for k in ("transformer", "bilstm", "cnn")}
    sec = {k: v["mean"] for k, v in timing.items()}
    sec_single = {k: M.time_per_trace(k, len(vocab), K, L, M.sample_configs(k, 1, 0)[0], n=16, reps=2)
                  for k in ("transformer", "bilstm", "cnn")}
    machine = {"cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "loadavg_before": load_before,
               "loadavg_after": os.getloadavg() if hasattr(os, "getloadavg") else None}
    n_tr1 = []                                                 # full-design training sizes (train + validation)
    for o, fs in fam_sets.items():
        ids = json.loads((P.work / f"split_I1_o{o}.json").read_text())["ids"]
        fs = set(fs)
        n_tr1.append(sum(alias.get(lab1.get(s, ""), lab1.get(s, "")) in fs for s in ids["train"] + ids["val"]))
    if parts is not None:
        n_tr2 = sum(1 for i, r in enumerate(r2) if f2[i] in cand and (r["sha256"] in parts["train"]
                                                                       or r["sha256"] in parts["val"]))
    else:
        n_tr2 = len(tr2)
    n_train = {"I1": float(np.mean(n_tr1)) if n_tr1 else 0.0, "I2": float(n_tr2)}
    # runs per learner, installation and origin: search + dropout grid + 2 final models per seed + 1 probe per seed
    runs_full = C.N_SEARCH + len(C.DROPOUT_GRID) + 3 * len(C.SEEDS)   # 20 + 4 + 15 = 39 (amendment 5; 21 under 3)
    k6 = gates.k6_projection(sec, n_train, origins={"I1": 3, "I2": 1}, runs_per_cell=runs_full)
    tri = {k: 2 * len(C.SEEDS) if k == "bilstm" else runs_full for k in sec}       # plan of amendments 3 and 5
    tri_h = {k: gates.k6_projection({k: sec[k]}, n_train, origins={"I1": 3, "I2": 1},
                                    runs_per_cell=tri[k])["total_hours"] for k in sec}
    k6["triaged_plan"] = {"runs_per_cell": tri, "hours_by_learner": tri_h, "total_hours": sum(tri_h.values()),
                          "K6_triggered": bool(sum(tri_h.values()) > C.K6_MAX_CPU_HOURS),
                          "rule": f"CPU plan (amendments 3 and 5): Transformer and 1D-CNN full factorial with "
                                  f"{C.N_SEARCH} search configurations and {len(C.SEEDS)} seeds; BiLSTM unmasked and "
                                  f"F-masked only; LightGBM full factorial (minutes per run, not projected)"}
    k6["n_train_full_design"] = {"I1_by_origin": n_tr1, "I1_mean": n_train["I1"], "I2": n_tr2}
    k6["seconds_per_trace_at_L"] = sec
    k6["timing_by_configuration"] = timing
    k6["seconds_per_trace_single_config_as_before"] = sec_single
    k6["machine"] = machine
    k6["dryrun_training_seconds_per_trace_epoch"] = (info["seconds"] / (len(A_tr) * info["epochs_run"])
                                                     if A_tr and info["epochs_run"] else float("nan"))
    masked_max = max((float(np.abs(M.integrated_gradients(model, t, test_mask=("F",))["ig"][t.cls == 1]).max())
                      for t in A_te[:20] if (t.cls == 1).any()), default=0.0)
    res = {"families": fams, "n": {"train": len(A_tr), "within": len(yw), "cross": len(yc)}, "L": L,
           "gap_dry": float(gap(np.arange(len(y)))), "gap_ci95": [lo, hi], "K5_P4": k5, "K6": k6,
           "P1_max_abs_IG_masked": masked_max, "P1_holds": masked_max == 0.0, "train_seconds": info["seconds"],
           "design": "amendment 3: shared families, all pseudo-origin traces, WinMET training and validation parts; "
                     "amendment 4: all requested epochs, K6 timed over the search configurations",
           "epochs_requested": args.epochs, "epochs_run": info["epochs_run"],
           "mean_recall_within": rec, "n_full_effective": n_full,
           "distinct_predictions": {"within": int(np.unique(pw).size), "cross": int(np.unique(pc).size)}}
    jdump(P.out / "dryrun_K5_K6_P1_P4.json", res)
    led.append("gate", evidence_id="dryrun", payload={k: res[k] for k in ("K5_P4", "K6", "P1_holds")})
    run_info(P, "dryrun", t0, selected={"I1_train": len(tr1), "I1_test": len(te1), "I2_train": len(tr2),
                                         "I2_test": len(te2)}, families=len(fams))
    print(json.dumps(res, indent=1, default=str))


def stage_ember_labels(args, P):
    """K4 label check (S6; source M): verify the EMBER2024 Windows archives against their published SHA-256 (they
    are downloaded if missing), read them without unpacking and keep the ClarAVy labels of SHA-256s that also occur in
    MELD-DS-448 or WinMET (work/ember2024_labels.json). harmonise uses that file for K4 when it is present."""
    import dataclasses
    import pandas as pd
    from a15 import labels as LB
    from a15.acquire import acquire_all
    t0 = time.perf_counter()
    led = open_ledger(P, args.tsa)
    prov = ProvenanceLog(P.out / "data_provenance_log.csv")
    d = P.data / "M"
    d.mkdir(parents=True, exist_ok=True)
    rev = (d / "REVISION.txt").read_text().strip() if (d / "REVISION.txt").exists() else ""
    C.SOURCES["M"] = dataclasses.replace(C.SOURCES["M"], revision=rev)
    if args.offline:                                           # test fixtures only: no listing, digests unverified
        status = {"M": "offline: archives not verified against published digests"}
        for f in C.SOURCES["M"].files:
            if (d / f).exists():
                prov.add(ProvRow("M", C.SOURCES["M"].name, f, "huggingface:joyce8/EMBER2024", rev, "Apache-2.0",
                                 (d / f).stat().st_size, sha256_file(d / f), status="unverified_offline"))
    else:
        status = acquire_all(P.data, prov, led, "authors' PC (WSL)", include=("M",))
    archives = [d / f for f in C.SOURCES["M"].files if (d / f).exists()]
    wanted = set(pd.read_csv(P.data / "I1" / "malicious_dataset_manifest.csv").sha256.str.lower())
    n1 = len(wanted)
    parts = i2_parts(P)
    if parts is not None:
        wanted |= parts["train"] | parts["val"] | parts["test"]
    else:
        wanted |= {r["sha256"] for r in iter_records(P, "I2", "ember-labels (WinMET hashes)")}
    found, scan = LB.scan_ember_archives(archives, wanted, label="ember-labels")
    sha = jdump(P.work / "ember2024_labels.json", found)
    in1 = sum(s in found for s in set(pd.read_csv(P.data / "I1" / "malicious_dataset_manifest.csv").sha256.str.lower()))
    led.append("label_check", evidence_id="M:ClarAVy", d_f=sha, shared_sha256=len(found), payload=scan,
               note="ClarAVy family labels of SHA-256s shared with MELD-DS-448 or WinMET; no feature value kept")
    anc = led.anchor()
    info = run_info(P, "ember-labels", t0, acquire=status, revision=rev, archives=scan, hashes_wanted=len(wanted),
                    shared_sha256=len(found), shared_with_I1_manifest=in1, I1_manifest_hashes=n1,
                    anchor={"i": anc["entry"]["i"], "tsa": anc["entry"]["tsa"], "trusted": anc["entry"]["trusted"]}
                    if anc else None)
    print(json.dumps(info, indent=1, default=str))


def stage_freeze_prep(args, P):
    from a15 import freeze as FZ
    lex = json.loads((P.work / "lexicon.json").read_text())
    alias = json.loads((P.out / "alias_table.json").read_text()) if (P.out / "alias_table.json").exists() else {}
    fam = json.loads((P.work / "family_sets.json").read_text())
    measured = {k: json.loads((P.out / f).read_text()) for k, f in (("P3", "P3_overlap.json"),
                                                                   ("dryrun", "dryrun_K5_K6_P1_P4.json"))
                if (P.out / f).exists()}
    split_manifests = {}
    for f in sorted(P.work.glob("split_*.json")):
        split_manifests[f.stem[6:]] = json.loads(f.read_text())
    dig = FZ.build_bundle(P.out / "freeze_bundle", lex, alias, fam, split_manifests, measured)
    (P.out / "freeze_bundle" / "OSF_registration.md").write_text(FZ.osf_registration(dig, dig["lexicon.json"]))
    open_ledger(P, args.tsa).append("freeze_prepared", payload=dig)
    print(json.dumps(dig, indent=1))


def stage_freeze(args, P):
    from a15 import freeze as FZ
    if not args.confirm_reviewed:
        raise SystemExit("the authors review freeze_bundle/ and the alias table first; then pass --confirm-reviewed")
    dig = {f.name: sha256_file(f) for f in sorted((P.out / "freeze_bundle").glob("*.json"))}
    led = open_ledger(P, args.tsa)
    anc = FZ.freeze(led, dig)
    print({"anchor": anc["entry"]["i"], "tsa": anc["entry"]["tsa"], "trusted": anc["entry"]["trusted"],
           "verify": led.verify()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tsa", default=None, help="RFC 3161 TSA URL (Cowork: https://freetsa.org/tsr)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--reports-archive", required=True)
    s.add_argument("--malvada-dir", required=True)
    s.add_argument("--plan-n-train-i1", type=int, default=20_000)
    s.add_argument("--plan-n-train-i2", type=int, default=25_000)
    sub.add_parser("acquire")
    g = sub.add_parser("ingest")
    g.add_argument("--inst", choices=["I1", "I2"], required=True)
    sub.add_parser("audit")
    sub.add_parser("harmonise")
    sub.add_parser("lexicon")
    o = sub.add_parser("overlap")
    o.add_argument("--n-boot", type=int, default=1999)
    d = sub.add_parser("dryrun")
    d.add_argument("--fraction", type=float, default=C.DRYRUN_FRACTION)
    d.add_argument("--epochs", type=int, default=10)         # amendment 4: 10 (was 5), the K6 projection's epochs
    e = sub.add_parser("ember-labels")
    e.add_argument("--offline", action="store_true", help="test fixtures only: skip digest verification")
    sub.add_parser("freeze-prep")
    fz = sub.add_parser("freeze")
    fz.add_argument("--confirm-reviewed", action="store_true")
    a = ap.parse_args()
    P = C.Paths(Path(a.root))
    {"selftest": selftest, "acquire": stage_acquire, "ingest": stage_ingest, "audit": stage_audit,
     "harmonise": stage_harmonise, "lexicon": stage_lexicon, "overlap": stage_overlap, "dryrun": stage_dryrun,
     "ember-labels": stage_ember_labels, "freeze-prep": stage_freeze_prep, "freeze": stage_freeze}[a.cmd](a, P)


if __name__ == "__main__":
    main()
