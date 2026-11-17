"""S10 job engine (7 Nov 2026): the restartable, parallel path gives exactly the in-process results; deviation D1
(8 Nov 2026): every test trace is scored and each tau_J analysis restricts traces and predictions consistently.

Synthetic inputs test code only. Each test trains tiny models on one thread, so it runs in a few minutes on a laptop.
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a15 import audit, s10data, s10jobs, synthetic                            # noqa: E402
from a15 import config as C                                                    # noqa: E402

PLAN = audit.Plan(label="SYNTHETIC TEST", learners=("transformer", "bilstm", "cnn", "lgbm"), n_search=2,
                  seeds=C.SEEDS[:2], dropout_grid=(0.5, 1.0), epochs=2, patience=2, n_boot=49, n_ig=4, n_perm=5)


@pytest.fixture(scope="module")
def world():
    import torch
    torch.set_num_threads(1)
    audit.LGBM_THREADS = 1
    corpus, V = synthetic.make_corpus("planted", n_per_family=(22, 18), seed=3)
    sets, cells = synthetic.build_cells(corpus, V, seed=3)
    return sets, cells


def _with_maxj(sets, cells, seed=11, tau_registered=0.62):
    """The synthetic world with the near-duplicate data s10data.assemble attaches (deviation D1)."""
    rng = np.random.default_rng(seed)
    out = {}
    for ck, c in cells.items():
        no, nx = len(c.own_traces), len(c.other_traces)
        kw = dict(own_maxj=rng.uniform(0.0, 1.0, no), own_same=rng.random(no) < 0.05,
                  other_maxj=rng.uniform(0.0, 1.0, nx), other_same=rng.random(nx) < 0.05,
                  tau_registered=tau_registered)
        if c.monthly:
            kw["monthly_maxj"] = {m: rng.uniform(0.0, 1.0, len(y)) for m, (y, _) in c.monthly.items()}
            kw["monthly_same"] = {m: np.zeros(len(y), dtype=bool) for m, (y, _) in c.monthly.items()}
        out[ck] = replace(c, **kw)
    new_sets = {n: replace(x, clusters={k: x.cluster for k in s10data.ANALYSES}) for n, x in sets.items()}
    return new_sets, out


def _run_jobs(tmp, sets, cells, workers, max_jobs=None, plan=PLAN):
    A = s10data.Assembly(sets, cells, audit.fit_groups(cells))
    A.digest = s10data.assembly_digest(A)
    apath = tmp / "assembly.pkl"
    if not apath.exists():
        s10data.save_assembly(A, apath)
    r = s10jobs.run_all(A if workers <= 1 else None, plan, tmp / "jobs", workers=workers, threads=1,
                        assembly_path=apath, progress_path=tmp / "progress.jsonl", max_jobs=max_jobs,
                        groups=A.groups, n_train={g: len(cells[c[0]].train) for g, c in A.groups.items()},
                        digest=A.digest, log=lambda *_: None)
    if r["remaining"]:
        return None, r
    preds, fits, checks, rand, monthly = s10jobs.collect(A, plan, tmp / "jobs")
    return audit.analyse_all(sets, cells, plan, preds, fits, checks, rand, monthly, 0.0), (preds, fits, checks)


def _same(a: dict, b: dict):
    for kind in PLAN.learners:
        assert a["decisions"][kind]["verdict"] == b["decisions"][kind]["verdict"]
        assert a["psi"][kind].keys() == b["psi"][kind].keys()
        assert np.array_equal([a["psi"][kind][k] for k in a["psi"][kind]], [b["psi"][kind][k] for k in a["psi"][kind]],
                              equal_nan=True)
    fa = {(r["learner"], r["inst"], r["role"], r["where"], r["mask"]): r for r in a["f1"]}
    fb = {(r["learner"], r["inst"], r["role"], r["where"], r["mask"]): r for r in b["f1"]}
    assert fa.keys() == fb.keys()
    for k in fa:
        va = np.array([fa[k]["f1"], fa[k]["ci_lo"], fa[k]["ci_hi"]], dtype=float)
        vb = np.array([fb[k]["f1"], fb[k]["ci_lo"], fb[k]["ci_hi"]], dtype=float)
        assert np.array_equal(va, vb, equal_nan=True), k
    for ck in a["fits"]:
        assert a["fits"][ck]["cfg"] == b["fits"][ck]["cfg"] and a["fits"][ck]["p_star"] == b["fits"][ck]["p_star"]
    for ck in a["checks"]:
        assert a["checks"][ck]["phi_base"] == pytest.approx(b["checks"][ck]["phi_base"])
        assert a["checks"][ck]["P1_holds"] == b["checks"][ck]["P1_holds"]
        assert a["checks"][ck]["n_explained"] == b["checks"][ck]["n_explained"]
    assert a.get("primary_analysis") == b.get("primary_analysis") and a.get("analysis") == b.get("analysis")
    assert (a.get("sensitivity") or {}).keys() == (b.get("sensitivity") or {}).keys()
    for k in a.get("sensitivity") or {}:
        sa, sb = a["sensitivity"][k], b["sensitivity"][k]
        assert sa["analysis"] == sb["analysis"]
        for kind in PLAN.learners:
            assert sa["decisions"][kind]["verdict"] == sb["decisions"][kind]["verdict"]
            assert np.array_equal([sa["psi"][kind][c] for c in sa["psi"][kind]],
                                  [sb["psi"][kind][c] for c in sa["psi"][kind]], equal_nan=True)


def test_jobs_equal_inprocess_restart_and_parallel(world, tmp_path):
    sets, cells = _with_maxj(*world)                           # deviation D1: four tau_J analyses, 0.8 primary
    ref = audit.run(sets, cells, PLAN, log=lambda *_: None)
    assert ref["primary_analysis"] == "0.8" and set(ref["sensitivity"]) == {"registered", "0.7", "0.9"}
    for ck, c in cells.items():                                 # kept counts follow equation (9) at each tau_J
        for key, tau in (("0.8", 0.8),):
            assert ref["analysis"]["kept"][ck]["own"] == int(((c.own_maxj < tau) & ~c.own_same).sum())
        assert ref["sensitivity"]["registered"]["analysis"]["kept"][ck]["other"] == int(
            ((c.other_maxj < 0.62) & ~c.other_same).sum())
        n_kept = int(((c.own_maxj < 0.8) & ~c.own_same).sum())
        assert ref["checks"][f"{ck}|transformer"]["n_explained"] == min(PLAN.n_ig, n_kept)
        assert ref["cells"][ck]["n_test_own"] == n_kept
    # BiLSTM triage: no search, p = 1, no artefact-only probe, H1 not tested
    assert ref["fits"]["I1_o1|bilstm"]["triage"] and ref["fits"]["I1_o1|bilstm"]["p_star"] == 1.0
    assert ref["fits"]["I1_o1|bilstm"]["cfg"] == audit.triage_config("bilstm", 1)
    assert ref["decisions"]["bilstm"]["verdict"]["H1"].startswith("not tested")
    assert not any(r["component"].startswith("psi1") for r in ref["decisions"]["bilstm"]["rows"])
    assert "P9" in ref["checks"]["I1_o1|transformer"]
    # job engine in this process, interrupted after 7 jobs and restarted: identical results
    res, info = _run_jobs(tmp_path / "inline", sets, cells, workers=1, max_jobs=7)
    assert res is None and info["done"] == 7
    res, _ = _run_jobs(tmp_path / "inline", sets, cells, workers=1)
    _same(ref, res)
    # two spawned worker processes on one thread each: identical results
    res2, _ = _run_jobs(tmp_path / "parallel", sets, cells, workers=2)
    _same(ref, res2)


def test_job_dependencies_and_counts():
    groups = {"I1_o1": ["I1_o1"], "I2_o1": ["I2_o1", "I2_o2"]}
    plan = audit.Plan(n_search=20, seeds=C.SEEDS, dropout_grid=C.DROPOUT_GRID)
    jobs = s10jobs.job_list(groups, plan)
    per = {k: sum(1 for j in jobs if j.kind == k and j.group == "I1_o1" and j.stage != "explain")
           for k in plan.learners}
    assert per == {"transformer": 39, "bilstm": 10, "cnn": 39, "lgbm": 39}      # runs per cell of the K6 plan
    keys = {j.key for j in jobs}
    for j in jobs:
        assert all(d.key in keys for d in s10jobs.deps(j, plan))
    mit = s10jobs.Job("I1_o1", "transformer", "mit", 0)
    assert {d.stage for d in s10jobs.deps(mit, plan)} == {"dropout"}
    assert s10jobs.deps(s10jobs.Job("I1_o1", "bilstm", "mit", 0), plan) == []


def test_nn_similarities_stream_equals_reference():
    from a15 import dedup
    rng = np.random.default_rng(5)
    base = [np.unique(rng.integers(0, 3000, rng.integers(20, 120))) for _ in range(40)]
    sets = []
    for k in range(400):                                     # near-duplicates, exact copies and unrelated sets
        b = base[k % 40]
        if k % 3 == 0:
            s = b.copy()
        else:
            keep = b[rng.random(b.size) > rng.uniform(0.0, 0.4)]
            s = np.unique(np.concatenate([keep, rng.integers(0, 3000, rng.integers(0, 15))]))
        sets.append(np.sort(s).astype(np.uint32))
    mh = dedup.MinHasher(128)
    sigs = np.array([mh.signature(s.astype(np.uint64)) for s in sets])
    ref = dedup.nn_similarities([s.astype(np.uint64) for s in sets], sigs)
    st = {}
    new = s10data.nn_similarities_stream(sets, sigs, stats=st)
    assert np.array_equal(ref, new)
    assert st["identical_pairs"] > 0 and st["jaccard_evaluated"] > 0


def test_merge_same_binary():
    cl = np.array([0, 1, 2, 3, 4])
    out, m = s10data.merge_same_binary(cl, ["a", "b", "a", "c", "b"], ["x", "x", "x", "x", "y"])
    assert m == 1 and out[0] == out[2] and len(set(out.tolist())) == 4      # b/x and b/y stay apart


def test_compact_dtypes_do_not_change_predictions(world):
    import torch
    from a15.models import Trace, predict, train_deep
    sets, cells = world
    c = cells["I1_o1"]
    cfg = audit.triage_config("cnn", 1)
    small = lambda t: Trace(t.ids.astype(np.int32), t.betas.astype(np.int8), t.cls.astype(np.int8), t.y, t.length)
    m64, _ = train_deep("cnn", c.V, len(c.families), c.train, c.val, cfg, 11, p_drop=0.5, epochs=2)
    m8, _ = train_deep("cnn", c.V, len(c.families), [small(t) for t in c.train], [small(t) for t in c.val], cfg, 11,
                       p_drop=0.5, epochs=2)
    for a, b in zip(m64.state_dict().values(), m8.state_dict().values()):
        assert torch.equal(a, b)
    assert np.array_equal(predict(m64, c.own_traces, ("F",)), predict(m8, [small(t) for t in c.own_traces], ("F",)))


def test_kendall_exact_matches_scipy_without_ties_and_handles_ties():
    from a15 import stats as S
    for series in ([70.1, 66.0, 64.2, 61.9, 63.0, 58.4], [50.0, 52.0, 49.0, 55.0, 47.0, 51.0]):
        a, b = audit.kendall_decline_exact(series), S.kendall_decline(series)
        assert a["tau"] == pytest.approx(b["tau"]) and a["p_one_sided"] == pytest.approx(b["p_one_sided"])
    r = audit.kendall_decline_exact([60.0, 60.0, 55.0, 55.0, 50.0, 50.0])
    assert r["ties"] and r["tau"] < 0 and 0 < r["p_one_sided"] < 0.05
    assert audit.kendall_decline_exact([60.0] * 6)["p_one_sided"] == 1.0


def test_restrict_keeps_traces_and_predictions_aligned(world):
    sets, cells = _with_maxj(*world)
    preds, monthly, rand = {}, {}, {}
    for ck, c in cells.items():
        for where, n in (("own", len(c.own_traces)), ("other", len(c.other_traces))):
            preds[(ck, "cnn", "base", where, "none")] = np.stack([np.arange(n), np.arange(n) + 1])  # position codes
        if c.monthly:
            monthly[(ck, "cnn")] = {m: np.arange(len(y))[None, :] for m, (y, _) in c.monthly.items()}
        n = len(c.own_traces)
        rand[ck] = {"test_borrowed_share": 0.5, "_borrowed": np.ones(n, dtype=int), "_masked": np.arange(n) + 1}
    for key in s10data.ANALYSES:
        tau = 0.62 if key == "registered" else float(key)
        s2, c2, p2, m2, r2 = audit.restrict(sets, cells, preds, monthly, rand, key)
        for ck, c in cells.items():
            ko = p2[(ck, "cnn", "base", "own", "none")][0]
            kx = p2[(ck, "cnn", "base", "other", "none")][0]
            assert np.array_equal(ko, np.flatnonzero((c.own_maxj < tau) & ~c.own_same))
            assert np.array_equal(kx, np.flatnonzero((c.other_maxj < tau) & ~c.other_same))
            assert [id(t) for t in c2[ck].own_traces] == [id(c.own_traces[i]) for i in ko]
            assert [id(t) for t in c2[ck].other_traces] == [id(c.other_traces[i]) for i in kx]
            assert np.array_equal(c2[ck].own_idx, np.asarray(c.own_idx)[ko])
            assert np.array_equal(c2[ck].other_idx, np.asarray(c.other_idx)[kx])
            assert "_borrowed" not in r2[ck] and "_masked" not in r2[ck]
            assert r2[ck]["test_borrowed_share"] == pytest.approx(len(ko) / max(float((ko + 1).sum()), 1.0))
            for m, (y, trs) in (c2[ck].monthly or {}).items():
                km = m2[(ck, "cnn")][m][0]
                assert [id(t) for t in trs] == [id(c.monthly[m][1][i]) for i in km]
                assert np.array_equal(y, np.asarray(c.monthly[m][0])[km])
            again, _ = audit.restrict_cell(c2[ck], key)            # restricting twice changes nothing
            assert [id(t) for t in again.own_traces] == [id(t) for t in c2[ck].own_traces]
        assert all(s2[n].cluster is sets[n].clusters[key] for n in sets)
    s3, c3, p3, _, r3 = audit.restrict(*world, preds, monthly, rand, "as_built")   # no near-duplicate data
    assert all(c3[ck] is world[1][ck] for ck in c3) and all(p3[k] is preds[k] for k in preds)
    assert all(r3[ck]["test_borrowed_share"] == 0.5 for ck in r3)


def test_deviation_recorded_once_before_scoring(tmp_path, monkeypatch):
    import run_s10
    from a15.ledger import Ledger, TimeStamper
    led = Ledger(tmp_path / "l.jsonl", C.STUDY_ID, TimeStamper(tmp_path / "tsa"))
    led.append("freeze", evidence_id="F")
    assert run_s10.record_deviations(led) == ["D1"]
    dev = [r["entry"] for r in led.records if r["entry"]["action"] == "deviation"]
    assert len(dev) == 1 and dev[0]["evidence_id"] == "D1" and led.records[-1]["entry"]["action"] == "anchor"
    assert run_s10.record_deviations(led) == []                     # idempotent
    led.append("test_score", evidence_id="S10")
    monkeypatch.setitem(run_s10.DEVIATIONS, "D9", {"title": "late"})
    with pytest.raises(SystemExit):                                  # refused once scoring has begun
        run_s10.record_deviations(led)
    assert led.verify()[0]
