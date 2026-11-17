"""Unit and property tests for the S9 harness. Synthetic inputs here test code only; they are never results."""
import hashlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a15 import config as C, dedup, gates, lexicon, masks, overlap, splits, stats, tokens   # noqa: E402
from a15.cape import ConcatReader, stream_7z                                              # noqa: E402
from a15.ledger import Ledger, TimeStamper, audit_path, mth, verify_inclusion, H          # noqa: E402


# ------------------------------------------------------------------------------------------------ ledger
@pytest.fixture()
def led(tmp_path):
    return Ledger(tmp_path / "l.jsonl", C.STUDY_ID, TimeStamper(tmp_path / "tsa"))


def test_rfc6962_vectors():
    leaves = [b"", b"\x00", b"\x10", b"\x20\x21", b"\x30\x31", b"\x40\x41\x42\x43",
              bytes(range(0x50, 0x58)), bytes(range(0x60, 0x70))]
    assert mth(leaves[:1]).hex() == "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"
    assert mth(leaves).hex() == "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328"
    for n in range(1, 40):
        ds = [H(bytes([i])) for i in range(n)]
        root = mth(ds)
        assert all(verify_inclusion(m, n, ds[m], audit_path(m, ds), root) for m in range(n))


def test_chain_anchor_verify_and_tamper(led):
    for i in range(50):
        led.append("seal", evidence_id=f"x{i}", d_f=hashlib.sha256(bytes([i])).hexdigest())
    anc = led.anchor()
    assert anc["entry"]["tsa"] == "LOCAL-DEV-TSA" and anc["entry"]["trusted"] == 0
    assert led.verify() == (True, "ok")
    led.records[10]["entry"]["d_f"] = "0" * 64                    # edit without re-hashing
    ok, msg = led.verify()
    assert not ok and "entry 11" in msg


def test_tamper_with_recomputation_caught_by_token(led):
    for i in range(20):
        led.append("seal", evidence_id=f"x{i}")
    led.anchor()
    led.records[5]["entry"]["evidence_id"] = "forged"            # recompute every later hash: token still binds root
    prev = led.h0
    from a15.ledger import enc
    for r in led.records:
        if r["entry"]["action"] == "anchor":
            break
        r["entry"]["prev"] = prev.hex()
        r["h"] = H(enc(r["entry"]) + prev).hex()
        prev = bytes.fromhex(r["h"])
    assert not led.verify(check_tokens=False)[0]                   # anchor entry's prev no longer matches


def test_vfrz(tmp_path):
    good = Ledger(tmp_path / "g.jsonl", C.STUDY_ID, TimeStamper(tmp_path / "tsa"))
    good.append("freeze", payload={"a": 1})
    good.anchor()
    good.append("test_score")
    assert good.v_frz() == 1
    bad = Ledger(tmp_path / "b.jsonl", C.STUDY_ID, TimeStamper(tmp_path / "tsa"))
    bad.append("freeze", payload={"a": 1})
    bad.append("test_score")
    bad.anchor()
    assert bad.v_frz() == 0


# ------------------------------------------------------------------------------------------------ tokeniser
def test_anonymisation_invariance_and_classes():
    raw = r"C:\Users\alice\AppData\Local\Temp\abc.exe"
    anon = r"C:\Users\[REDACTED]\AppData\Local\Temp\abc.exe"
    assert tokens.path_class(raw) == tokens.path_class(anon) == "USER_TEMP"
    assert tokens.ext_class(raw) == "EXE"
    assert tokens.path_class(r"\??\C:\Windows\System32\kernel32.dll") == "SYS32"
    assert tokens.reg_class(r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Run") == "HKLM:RUN"
    assert tokens.reg_class(r"\REGISTRY\MACHINE\HARDWARE\DESCRIPTION\System") == "HKLM:HARDWARE_DESC"
    assert tokens.reg_class(r"HKLM\SOFTWARE\Oracle\VirtualBox Guest Additions") == "HKLM:VM_ARTEFACT"
    assert tokens.ip_class("192.168.56.1") == "PRIVATE" and tokens.ip_class("8.8.8.8") == "PUBLIC"
    assert tokens.sleep_class("120000") == "LT10MIN" and tokens.sleep_class("0xffffffff") == "INFINITE"
    assert tokens.proc_class(r"C:\Python38\pythonw.exe agent.py") == "AGENT"
    assert tokens.ret_class(False, "0xc0000034") == "NOT_FOUND"


def test_token_has_no_raw_values():
    args = (("FileName", r"C:\Users\bob\Desktop\secret_report.docx"), ("Handle", "0x00000a4c"),
            ("Buffer", "password=hunter2"), ("Milliseconds", "30000"), ("ProcessName", "evil.exe"))
    t = tokens.token("NtCreateFile", args, True, "0x0", None)
    assert "bob" not in t and "secret" not in t and "hunter2" not in t and "0x00000a4c" not in t
    assert re.fullmatch(r"[A-Za-z0-9_]+\|[A-Za-z0-9_#:;=]*\|[A-Z_]+", t), t


def test_collapse_beta_trunc_L():
    seq = tokens.collapse([("a", 1), ("a", 3), ("b", 1), ("a", 200)])
    assert seq == [("a", 2), ("b", 0), ("a", 7)]                  # 4 calls -> bucket 2; 200 -> capped at 7
    assert tokens.trunc_head_tail(list(range(10)), 4) == [0, 1, 8, 9]
    assert tokens.choose_L([100] * 95 + [900] * 5) == 128 and tokens.choose_L([5000] * 20) == C.L_CAP


# ------------------------------------------------------------------------------------------------ lexicon
def test_lexicon_rules():
    assert lexicon.classify(C.OOS) == ("F", "F0")
    assert lexicon.classify_alt(C.OOS) == "B"
    assert lexicon.classify("NtDelayExecution|Status:SLEEP_SKIPPED;Milliseconds:sleep=LT60S|OK")[0] == "F"
    assert lexicon.classify("NtOpenKey|FullName:reg=HKLM:VM_ARTEFACT|NOT_FOUND")[0] == "P"
    assert lexicon.classify("NtDelayExecution|Milliseconds:sleep=LT60S|OK")[0] == "P"
    assert lexicon.classify("GetCursorPos||OK")[0] == "P"
    assert lexicon.classify("NtWriteFile|FileName:path=USER_TEMP;FileName#:ext=EXE|OK") == ("B", "B0")
    assert lexicon.classify("GetTickCount||OK") == ("B", "P8") and lexicon.classify_alt("GetTickCount||OK") == "P"


# ------------------------------------------------------------------------------------------------ masks
def test_matched_mask_constraints():
    gen = np.random.default_rng(0)
    cls = np.array([1] * 20 + [0] * 200 + [2] * 10)
    band = gen.integers(0, 5, size=cls.size)
    chosen, borrowed, m = masks.matched_random_mask(cls, band, ("F", "P"), gen, 5)
    assert m == 30 and len(chosen) == 30
    assert not set(chosen) & set(np.flatnonzero(np.isin(cls, [1, 2])))


def test_apply_mask_dropout():
    ids, betas, cls = np.arange(10, 20), np.ones(10, int), np.array([1, 0] * 5)
    i2, b2, sel = masks.apply_mask(ids, betas, cls, ("F",), 1, C.BETA_NEUTRAL)
    assert (i2[cls == 1] == 1).all() and (b2[cls == 1] == C.BETA_NEUTRAL).all() and (i2[cls == 0] == ids[cls == 0]).all()


# ------------------------------------------------------------------------------------------------ overlap (synthetic fixture)
def _fixture(shared_f: bool, seed=0):
    rng = np.random.default_rng(seed)
    B = [f"api{i}||OK" for i in range(40)]
    F1 = [f"NtDelayExecution|Status:SLEEP_SKIPPED;Milliseconds:sleep=LT{x}|OK" for x in ("1S", "10S")]
    F2 = F1 if shared_f else [f"NtDelayExecution|Status:SLEEP_SKIPPED;Milliseconds:sleep={x}|OK" for x in ("LT60S", "ZERO")]
    mk = lambda F: [list(rng.choice(B, 30)) + list(rng.choice(F, 5)) for _ in range(150)]
    return mk(F1), mk(F2)


def test_jw_identity_and_p3_direction():
    a, b = np.array([3., 1, 0, 2]), np.array([1., 1, 4, 0])
    qa, qb = a / a.sum(), b / b.sum()
    tv = 0.5 * np.abs(qa - qb).sum()
    assert abs(overlap.jw(a, b) - (1 - tv) / (1 + tv)) < 1e-12
    s1, s2 = _fixture(shared_f=False)
    r = overlap.class_overlap(s1, s2, lambda t: lexicon.classify(t)[0], n_boot=199)
    assert r["P3_holds"] and r["J_w"]["F"] < r["J_w"]["B"]
    s1, s2 = _fixture(shared_f=True)
    r = overlap.class_overlap(s1, s2, lambda t: lexicon.classify(t)[0], n_boot=199)
    assert not r["P3_holds"]


# ------------------------------------------------------------------------------------------------ splits
def test_origin_windows_never_open_test_in_dry_run():
    dates = pd.Series(pd.date_range("2024-06-01", "2025-08-31", freq="D"))
    dm = splits.dryrun_masks(dates)
    assert dates[dm["test"]].max() < pd.Timestamp("2025-03-01")
    for oid, m in splits.origin_masks(dates).items():
        assert not (m["train"] & m["test"]).any() and not (m["val"] & m["test"]).any()
        assert dates[m["train"]].max() < dates[m["val"]].min() < dates[m["test"]].min()


def test_family_names_and_subsample():
    assert splits.norm_family("Cobalt Strike") == splits.norm_family("CobaltStrike") == "cobaltstrike"
    assert splits.norm_family("unknown") == "" and splits.norm_family("(n/a)") == ""
    keys = pd.DataFrame({"month": ["2025-01"] * 300 + ["2025-02"] * 100, "family": ["a", "b"] * 200})
    keep = splits.temporal_subsample(keys, 0.1, 0)
    assert keep.sum() == 40


# ------------------------------------------------------------------------------------------------ statistics
def test_balanced_f1_matches_equation_12():
    rng = np.random.default_rng(1)
    K = 4
    y = rng.choice(K, 4000, p=[0.55, 0.25, 0.15, 0.05])
    yhat = np.where(rng.random(4000) < 0.8, y, rng.integers(0, K, 4000))
    C_ = stats.confusion(y, yhat, K).astype(float)
    G = C_ / C_.sum(1, keepdims=True)
    f1bal = 2 * np.diag(G) / (1 + G.sum(0))
    assert abs(stats.macro_f1(y, yhat, K, True) - 100 * f1bal.mean()) < 1e-9


def test_planning_and_tests():
    assert stats.n_required(0.8, 1.0, 1.0, 2.0) == 3074
    assert abs(stats.half_width(3000, 0.8, 1.0, 1.01) - 2.04) < 0.01
    from scipy.stats import binomtest
    assert abs(stats.mcnemar_exact(12, 4) - binomtest(4, 16, 0.5).pvalue) < 1e-12
    assert stats.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    reps = np.random.default_rng(2).normal(3, 1, 9999)
    srt = np.sort(reps)
    assert (stats.p_greater(reps, srt[249] - 1e-9) <= 0.025) and (stats.p_greater(reps, srt[250]) > 0.025)
    assert abs(stats.aut([1, 1, 1]) - 1) < 1e-12 and stats.hodges_lehmann([1, 2, 4]) == 2.25


def test_cluster_bootstrap_keeps_cluster_counts():
    strata = ["a"] * 6 + ["b"] * 4
    clusters = [0, 0, 1, 2, 2, 2, 5, 6, 6, 7]
    boot = stats.ClusterBootstrap(strata, clusters, n_boot=50, seed=0)
    for idx in boot:
        assert set(np.array(strata)[idx]) <= {"a", "b"}
    assert boot.design_effect() > 1


def test_verdicts_and_k5():
    v = stats.verdicts({"H1": 0.001, "H2": 0.2, "H3": 0.004, "H4": 0.5}, {"H1": 1, "H2": 0.001, "H3": 1, "H4": 0.9})
    assert v["verdict"] == {"H1": "supported", "H2": "rejected", "H3": "supported", "H4": "inconclusive"}
    k = gates.k5_p4(measured_hw=10.0, n_dry=120, rec=0.8, deff=1.2, n_full=3000)
    assert k["K5_triggered"] is (k["projected_hw_full"] > 2)


def test_privacy_scan():
    clean = ["NtCreateFile|FileName:path=USER_TEMP;FileName#:ext=EXE|OK"]
    assert not gates.residual_data_scan(clean, 1)["release_blocked"]
    dirty = clean + [r"C:\Users\carol\file", "10.0.0.5", "x@example.org"]
    r = gates.residual_data_scan(dirty, 1)
    assert r["release_blocked"] and r["hits"]["user_path"] == 1 and r["hits"]["ipv4"] == 1


# ------------------------------------------------------------------------------------------------ dedup
def test_dedup_exact_and_clusters():
    a = dedup.api_3grams(["A", "B", "C", "D", "E"])
    b = dedup.api_3grams(["A", "B", "C", "D", "E"])
    c = dedup.api_3grams(["X", "Y", "Z", "W", "V"])
    mh = dedup.MinHasher(125)
    sig = np.array([mh.signature(s) for s in (a, b, c)])
    keep, maxj = dedup.dedup_test([a], [b, c], sig[:1], sig[1:], 0.8)
    assert list(keep) == [False, True] and maxj[0] == 1.0
    ids, cross = dedup.test_clusters([a, b, c], sig, ["f", "g", "f"], ["s", "s", "s"], 0.8)
    assert ids[0] != ids[1] and cross == 1                          # different labels: flagged, never merged


def test_minhash_unbiased():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2 ** 31 - 1, 600).astype(np.uint64)
    s1, s2 = np.unique(base[:500]), np.unique(base[100:])
    j = dedup.jaccard(s1, s2)
    est = np.mean([np.mean(dedup.MinHasher(128, seed=k).signature(s1) == dedup.MinHasher(128, seed=k).signature(s2))
                   for k in range(20)])
    assert abs(est - j) < 0.03


# ------------------------------------------------------------------------------------------------ archive streaming
def test_stream_split_and_encrypted_7z(tmp_path):
    import py7zr
    members = {f"r{i}.json": json.dumps({"i": i, "pad": "x" * 5000}).encode() for i in range(6)}
    for pw in (None, "infected"):
        arc = tmp_path / f"a_{pw}.7z"
        with py7zr.SevenZipFile(arc, "w", password=pw) as z:
            for k, v in members.items():
                z.writestr(v, k)
        blob = arc.read_bytes()
        parts = []
        for k, off in enumerate(range(0, len(blob), 4000)):          # byte-split volumes, as 7-Zip writes them
            p = tmp_path / f"a_{pw}.7z.{k + 1:03d}"
            p.write_bytes(blob[off:off + 4000])
            parts.append(p)
        got = {}
        n = stream_7z(ConcatReader(parts), lambda name, data: got.__setitem__(name, data), password=pw)
        assert n == 6 and got == members


# ------------------------------------------------------------------------------------------------ learners: P1, padding, completeness
@pytest.mark.parametrize("kind", ["transformer", "bilstm", "cnn"])
def test_P1_ig_zero_at_masked_positions_and_completeness(kind):
    import torch
    from a15 import models as M
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    V, K = 60, 4
    model = M.BUILDERS[kind](V, K, **({"layers": 2} if kind == "transformer" else {}))
    for _ in range(5):
        n = int(rng.integers(20, 80))
        t = M.Trace(rng.integers(4, V, n), rng.integers(0, 8, n), rng.choice(3, n, p=[0.6, 0.3, 0.1]), 0)
        g = M.integrated_gradients(model, t, m=50, test_mask=("F",))
        assert np.all(g["ig"][t.cls == 1] == 0.0)                  # P1: exactly zero, not approximately
        if kind != "cnn":                                           # smooth networks: completeness at m = 50
            assert M.integrated_gradients(model, t, m=50)["delta"] < 0.05
    if kind == "cnn":                                               # max pooling is piecewise: escalation rule (P2)
        pop = M.ig_population(model, [M.Trace(rng.integers(4, V, 40), rng.integers(0, 8, 40), np.zeros(40, int), 0)
                                      for _ in range(10)])
        assert pop["m"] in (50, 100, 200)
    short = M.Trace(rng.integers(4, V, 15), rng.integers(0, 8, 15), np.zeros(15, int), 0)
    long = M.Trace(rng.integers(4, V, 70), rng.integers(0, 8, 70), np.zeros(70, int), 0)
    alone = M.predict(model, [short], logits=True)
    batched = M.predict(model, [short, long], logits=True)[:1]
    assert np.allclose(alone, batched, atol=1e-5)                   # padding never changes a real position


# ------------------------------------------------------------------------------------------------ amendment of 3 Nov 2026
def _test_clusters_before_amendment(test_sets, test_sigs, labels, strata, tau):
    """dedup.test_clusters as it stood before 3 November 2026 (one set of all candidate pairs); reference only."""
    parent = list(range(len(test_sets)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    bands, rows = dedup.config_for(tau)
    cross = 0
    for i, j in dedup.candidates_within(test_sigs[:, :bands * rows], bands, rows):
        if strata[i] != strata[j] or dedup.jaccard(test_sets[i], test_sets[j]) < tau:
            continue
        if labels[i] != labels[j]:
            cross += 1
            continue
        parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(len(test_sets))])
    _, ids = np.unique(roots, return_inverse=True)
    return ids, cross


def _partition(ids):
    groups = {}
    for i, c in enumerate(np.asarray(ids).tolist()):
        groups.setdefault(c, []).append(i)
    return sorted(tuple(g) for g in groups.values())


def _near_duplicate_sets(seed, n=700, n_base=45):
    rng = np.random.default_rng(seed)
    base = [rng.integers(0, 2 ** 31 - 1, int(rng.integers(3, 220))).astype(np.uint64) for _ in range(n_base)]
    sets = []
    for _ in range(n):
        b = base[int(rng.integers(n_base))]
        keep = rng.random(b.size) >= rng.choice([0.0, 0.0, 0.05, 0.12, 0.3, 0.6])
        extra = rng.integers(0, 2 ** 31 - 1, int(rng.integers(0, 8))).astype(np.uint64)
        sets.append(np.unique(np.concatenate([b[keep], extra])))
    return sets, rng


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_clusters_amendment_equivalence(seed):
    """Same partition and cross-label count as the earlier implementation; ids numbered by first member."""
    sets, rng = _near_duplicate_sets(seed)
    mh = dedup.MinHasher(125)
    sig = np.array([mh.signature(s) for s in sets])
    fams = [f"f{int(rng.integers(5))}" for _ in sets]
    cases = [(fams, fams), ([f"l{int(rng.integers(3))}" for _ in sets], [f"s{int(rng.integers(2))}" for _ in sets])]
    for labels, strata in cases:
        for tau in (0.7, 0.8, 0.9):
            old_ids, old_cross = _test_clusters_before_amendment(sets, sig, labels, strata, tau)
            st = {}
            new_ids, new_cross = dedup.test_clusters(sets, sig, labels, strata, tau, stats=st)
            assert _partition(old_ids) == _partition(new_ids)
            assert old_cross == new_cross == st["cross_label_pairs"]
            first = [int(np.flatnonzero(new_ids == c)[0]) for c in range(int(new_ids.max()) + 1)]
            assert first == sorted(first) and new_ids.dtype == np.int64
            bands, rows = dedup.config_for(tau)
            assert st["pairs_visited"] == len(dedup.candidates_within(sig[:, :bands * rows], bands, rows))


def test_candidate_neighbours_equal_candidates_within():
    sets, _ = _near_duplicate_sets(4, n=400)
    mh = dedup.MinHasher(128)
    sig = np.array([mh.signature(s) for s in sets])
    for bands, rows in ((32, 4), (25, 5), (16, 8)):
        sub = sig[:, :bands * rows]
        got = {(i, int(j)) for i, js in dedup.candidate_neighbours(sub, bands, rows) for j in js}
        assert got == dedup.candidates_within(sub, bands, rows)


def test_clusters_bucket_cap_and_empty():
    one = np.array([5, 9, 11], dtype=np.uint64)
    sets = [one] * (dedup.MAX_BUCKET + 1) + [np.array([1, 2, 3, 4], dtype=np.uint64)] * 2
    mh = dedup.MinHasher(125)
    sig = np.array([mh.signature(s) for s in sets])
    lab = ["f"] * len(sets)
    old_ids, _ = _test_clusters_before_amendment(sets, sig, lab, lab, 0.8)
    new_ids, _ = dedup.test_clusters(sets, sig, lab, lab, 0.8)
    assert _partition(old_ids) == _partition(new_ids)                # capped bucket: singletons in both
    assert new_ids[-1] == new_ids[-2] and len(set(new_ids[:-2].tolist())) == dedup.MAX_BUCKET + 1
    ids, cross = dedup.test_clusters([], np.zeros((0, 125), dtype=np.uint64), [], [], 0.8)
    assert ids.size == 0 and cross == 0


def _class_overlap_before_amendment(seqs1, seqs2, classify, n_boot=1999, seed=0, chunk=50):
    """overlap.class_overlap as it stood before 3 November 2026; reference only."""
    vocab = sorted({t for s in seqs1 for t in s} | {t for s in seqs2 for t in s})
    index = {t: i for i, t in enumerate(vocab)}
    cls = np.array([classify(t) for t in vocab])
    X1, X2 = overlap.count_matrix(seqs1, index), overlap.count_matrix(seqs2, index)
    cols = {c: np.flatnonzero(cls == c) for c in "BFP"}
    point = {c: overlap.jw(np.asarray(X1[:, k].sum(0)).ravel(), np.asarray(X2[:, k].sum(0)).ravel()) if k.size else
             float("nan") for c, k in cols.items()}
    rng = np.random.default_rng(seed)
    n1, n2 = X1.shape[0], X2.shape[0]
    reps = {c: [] for c in "BFP"}
    done = 0
    import scipy.sparse as sp
    while done < n_boot:
        r = min(chunk, n_boot - done)
        W1 = rng.multinomial(n1, np.full(n1, 1 / n1), size=r).astype(np.float64)
        W2 = rng.multinomial(n2, np.full(n2, 1 / n2), size=r).astype(np.float64)
        for c, k in cols.items():
            if not k.size:
                continue
            A, B = (sp.csr_matrix(W1) @ X1[:, k]).toarray(), (sp.csr_matrix(W2) @ X2[:, k]).toarray()
            reps[c].extend(overlap.jw(A[q], B[q]) for q in range(r))
        done += r
    d = np.array(reps["B"]) - np.array(reps["F"]) if reps["F"] else np.array([])
    lo, hi = (np.nanpercentile(d, [2.5, 97.5]) if d.size else (np.nan, np.nan))
    return {"J_w": point, "diff_B_minus_F": point["B"] - point["F"], "ci95": [float(lo), float(hi)],
            "P3_holds": bool(lo > 0), "n_boot": n_boot, "n_traces": [n1, n2],
            "n_token_types": {c: int(k.size) for c, k in cols.items()},
            "implication": "" if lo > 0 else "P3 fails: S8 predicts an upper bound of rho below 0.10 (H4 rejected)"}


def test_class_overlap_streaming_equivalence():
    rng = np.random.default_rng(11)
    toks = ([f"NtApi{k}|FileName:path=USER_TEMP|OK" for k in range(25)] + ["GetCursorPos||OK", C.OOS,
            "NtDelayExecution|Milliseconds:sleep=GE10MIN|OK", "Sleep|Status:SLEEP_SKIPPED|OK"])
    def draw(n):
        return [[toks[j] for j in rng.integers(0, len(toks), int(rng.integers(0, 60)))] for _ in range(n)]
    seqs1, seqs2 = draw(70), draw(55)
    classify = lambda t: lexicon.classify(t)[0]
    ref = _class_overlap_before_amendment(seqs1, seqs2, classify, n_boot=230, chunk=50)
    assert json.dumps(overlap.class_overlap(seqs1, seqs2, classify, n_boot=230), sort_keys=True) == \
        json.dumps(ref, sort_keys=True)
    cb = overlap.CountBuilder(("I1", "I2"))
    for s in seqs1:
        cb.add("I1", s)
    for s in seqs2:
        cb.add("I2", s)
    X, vocab = cb.matrices()
    got = overlap.class_overlap_counts(X["I1"], X["I2"], vocab, classify, n_boot=230)
    assert json.dumps(got, sort_keys=True) == json.dumps(ref, sort_keys=True)


def test_head_tail_store_equivalence():
    """Keeping the first and last 256 elements is enough for trunc_head_tail at every L the dry run can choose."""
    rng = np.random.default_rng(3)
    for n in (0, 1, 2, 3, 255, 256, 257, 511, 512, 513, 700, 2000):
        seq = [(f"t{k}", int(rng.integers(8))) for k in range(n)]
        stored = tokens.head_tail_store(seq, C.L_CAP)
        assert len(stored) == min(n, C.L_CAP)
        for L in (2 ** k for k in range(1, C.L_CAP.bit_length())):        # every power of two up to L_CAP
            assert tokens.trunc_head_tail(stored, L) == tokens.trunc_head_tail(seq, L)


def test_gate_dictionaries_serialise():
    """Gate results go into the ledger with json.dumps: no NumPy scalars may remain (dry-run fix, 3 Nov 2026)."""
    k5 = gates.k5_p4(1.7, 400.0, 0.8, 1.3, 2500)
    k6 = gates.k6_projection({"transformer": 0.05}, {"I1": 1000.0, "I2": 900.0}, origins={"I1": 3, "I2": 1})
    for d in (k5, k6):
        json.dumps(d, sort_keys=True)
    assert type(k5["P4_holds"]) is bool and type(k5["K5_triggered"]) is bool and type(k6["K6_triggered"]) is bool


def test_jaccard_sorted_equals_jaccard():
    rng = np.random.default_rng(17)
    cases = [(np.zeros(0, np.uint64), np.zeros(0, np.uint64)), (np.zeros(0, np.uint64), np.array([3], np.uint64))]
    for _ in range(400):
        pool = rng.integers(0, int(rng.choice([20, 500, 2 ** 31 - 1])), 3000).astype(np.uint64)
        a = np.unique(pool[: int(rng.integers(1, 1500))])
        b = np.unique(np.concatenate([a[rng.random(a.size) < rng.random()], pool[-int(rng.integers(0, 1500)):]]))
        cases.append((a, b))
    for a, b in cases:
        for x, y in ((a, b), (b, a), (a.astype(np.uint32), b.astype(np.uint32))):
            assert dedup.jaccard_sorted(x, y) == dedup.jaccard(x, y)


# ------------------------------------------------------------------------------------------------ amendment 3 (4 Nov 2026)
def test_label_agreement_helpers():
    from a15 import labels as LB
    pairs = [("redline", ["redline", "metastealer"]), ("redline", ["crifi"]), ("amadey", []), ("", ["loki"]),
             ("loki", ["loki"])]
    a = LB.agreement_any(pairs)
    assert a["n_both_named"] == 3 and a["n_missing_label"] == 2 and abs(a["agreement"] - 2 / 3) < 1e-12
    pf = LB.per_family(pairs, ["redline", "loki", "amadey"])
    assert pf["redline"] == {"n": 2, "agreement": 0.5} and pf["loki"]["agreement"] == 1.0 and pf["amadey"]["n"] == 0
    assert LB.claravy_family({"family": "RedLine", "family_confidence": 0.83}) == ("RedLine", 0.83)
    assert LB.claravy_family({"clarAVy_family": "x", "family_conf": 1}) == ("x", 1.0)
    assert LB.claravy_family({"sha256": "a"}) == ("", None)


def test_k4_claravy_rule():
    from a15 import labels as LB
    ember, i1, i2 = {}, {}, {}
    for k in range(30):                                            # agreeing family, 30 shared in I1
        ember[f"a{k}"] = {"family": "Remcos", "family_confidence": 0.9}
        i1[f"a{k}"] = "remcos"
    for k in range(12):                                            # 12 shared in I2, only 6 agree -> dropped
        ember[f"b{k}"] = {"family": "agenttesla" if k < 6 else "tedy", "family_confidence": 0.5}
        i2[f"b{k}"] = "agenttesla"
    for k in range(4):                                             # too few shared -> unverified
        ember[f"c{k}"] = {"family": "loki"}
        i1[f"c{k}"] = "loki"
    ember["d"] = {"family": "SnakeKeylogger"}                      # alias applied to the ClarAVy name
    i2["d"] = "snakelogger"
    fams = ["remcos", "agenttesla", "loki", "snakelogger", "gcleaner"]
    out = LB.k4_claravy(ember, {"I1": i1, "I2": i2}, fams, {"snakekeylogger": "snakelogger"}, min_conf=0.0)
    r = out["rule"]
    assert r["dropped"] == ["agenttesla"] and r["verified"] == ["remcos"]
    assert set(r["unverified"]) == {"loki", "snakelogger", "gcleaner"}
    assert out["per_family"]["I2"]["agenttesla"]["claravy_names_when_disagreeing"] == {"tedy": 6}
    assert out["I1"]["shared_sha256"] == 34 and abs(out["I2"]["agreement_all_labels"] - 7 / 13) < 1e-12
    sure = LB.k4_claravy(ember, {"I1": i1, "I2": i2}, fams, {"snakekeylogger": "snakelogger"})   # confidence >= 0.7
    assert sure["min_confidence"] == 0.7 and sure["rule"]["verified"] == ["remcos"]
    assert "agenttesla" in sure["rule"]["unverified"] and sure["per_family"]["I2"]["agenttesla"]["n"] == 0
    assert sure["per_family"]["I2"]["agenttesla"]["n_all_labels"] == 12


def test_scan_ember_archives(tmp_path):
    import gzip as gz
    import zipfile
    from a15 import labels as LB
    shas = [hashlib.sha256(str(k).encode()).hexdigest() for k in range(6)]
    line = lambda s, fam: json.dumps({"sha256": s.upper() if s == shas[1] else s, "md5": "m", "label": 1,
                                      "family": fam, "family_confidence": 0.8,
                                      "histogram": list(range(300)), "imports": {"k": ["a"]}}) + "\n"
    z1 = tmp_path / "Win32_train.zip"
    with zipfile.ZipFile(z1, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("Win32_train/week1.jsonl", line(shas[0], "redline") + line(shas[1], "amadey") + line(shas[2], "x"))
        z.writestr("Win32_train/week2.jsonl.gz", gz.compress((line(shas[3], "loki") + line(shas[0], "dup")).encode()))
        z.writestr("Win32_train/readme.txt", "not a record")
    found, st = LB.scan_ember_archives([z1], {shas[0], shas[1], shas[3], shas[5]})
    assert set(found) == {shas[0], shas[1], shas[3]} and found[shas[0]]["family"] == "redline"
    assert "histogram" not in found[shas[0]] and "imports" not in found[shas[0]]       # features are never kept
    assert st["Win32_train.zip"] == {"members": 2, "records": 5, "shared_sha256": 3, "repeats": 1}


def test_train_deep_runs_all_epochs_without_validation():
    """Amendment 4: without a validation set, train_deep trains for every requested epoch (it stopped after
    `patience` epochs before, because every validation score was NaN). With one, early stopping still applies."""
    from a15 import models as M
    rng = np.random.default_rng(0)
    V, K = 60, 3
    tr = [M.Trace(rng.integers(4, V, n), rng.integers(0, 3, n), np.zeros(n, dtype=np.int64), int(rng.integers(0, K)))
          for n in rng.integers(5, 20, 24)]
    cfg = M.sample_configs("cnn", 1, C.SEEDS[0])[0]
    _, info = M.train_deep("cnn", V, K, tr, [], cfg, C.SEEDS[0], epochs=5, patience=3)
    assert info["epochs_run"] == 5 and len(info["history"]) == 5
    _, info2 = M.train_deep("cnn", V, K, tr, tr[:6], cfg, C.SEEDS[0], epochs=12, patience=1)
    assert 1 <= info2["epochs_run"] <= 12 and info2["epochs_run"] == len(info2["history"])


def test_time_per_trace_configs():
    """Amendment 4: the K6 input is the mean over the search configurations; the first of them is the configuration
    the dry run trains (same seed)."""
    from a15 import models as M
    cfgs = M.sample_configs("cnn", 3, C.SEEDS[0])
    assert cfgs[0] == M.sample_configs("cnn", 1, C.SEEDS[0])[0]
    t = M.time_per_trace_configs("cnn", 50, 3, 16, cfgs, n=2, reps=1)
    assert len(t["per_config"]) == 3 and t["min"] <= t["mean"] <= t["max"] and t["mean"] > 0
    json.dumps(t)


def test_bench_threads_runs_and_picks_fastest(capsys):
    """Amendment 4: the thread check times every learner at each thread count and names the fastest."""
    import bench_threads
    out = bench_threads.main(["--threads", "1,2", "--L", "8", "--V", "50", "--K", "3", "--n", "2", "--reps", "1"])
    assert [r["threads"] for r in out["rows"]] == [1, 2]
    assert out["best"]["hours"]["total"] == min(r["hours"]["total"] for r in out["rows"])
    assert "export A15_THREADS=" in capsys.readouterr().out


def test_amendment5_margin_and_plan_recorded():
    """Amendment 5: margin 3 points (registered K5 action) in every hypothesis text and gate record; five seeds and
    twenty search configurations; K5 keeps its registered 2-point limit."""
    from a15 import freeze as FZ
    assert C.DELTA == 3.0 and C.SEEDS == (11, 23, 37, 41, 53) and C.N_SEARCH == 20 and C.K5_MAX_HALF_WIDTH == 2.0
    blob = json.dumps(FZ.HYPOTHESES)
    assert "2 points" not in blob and "+/-2" not in blob and "+/-3" in blob and "above 3 points" in blob
    assert FZ.SETTINGS["k5_applied"]["margin_points"] == 3.0 and FZ.SETTINGS["seeds"] == [11, 23, 37, 41, 53]
    k5 = gates.k5_p4(7.12, 523.5, 0.459, 1.0, 789.3)
    assert k5["K5_triggered"] and k5["margin_in_force"] == 3.0 and k5["projected_exceeds_margin"]
    assert gates.k5_p4(1.0, 523.5, 0.8, 1.0, 5000.0)["projected_exceeds_margin"] is False
