#!/usr/bin/env python
"""S10 driver for the A15 audit.

Rehearsal (any machine; synthetic data with planted ground truth, never results):
    python run_s10.py --root RUN rehearsal --scenario planted|null [--engine jobs --workers 2]
Evaluation (authors' PC, after the S9 stages, the freeze and the OSF registration):
    python run_s10.py --root WORK --tsa https://freetsa.org/tsr evaluate --workers 4 --threads 3
    python run_s10.py --root WORK status                       # progress and time estimate, from another terminal
'evaluate' refuses to score any test trace unless the ledger verifies and the freeze entry sits in an anchored batch
(V_frz = 1, S8 equation 28); it also refuses an untrusted (local) anchor unless --allow-untrusted-anchor is given.
It is restartable: run the same command again after any interruption and finished jobs are kept (7 Nov 2026).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a15 import audit, synthetic                                          # noqa: E402
from a15 import config as C                                               # noqa: E402
from a15.figures import LEARN, fmt_p, make_all                            # noqa: E402

# The registered protocol with the CPU plan of amendments 3 and 5 (frozen in settings.json): 20 search configurations
# and 5 seeds for the Transformer, 1D-CNN and n-gram LightGBM; BiLSTM unmasked and F-masked only (Plan.triage).
PLANS = {"protocol": dict(label="MEASURED (authors' PC; protocol with the CPU plan of amendments 3 and 5)"),
         # code tests on the S9 fixtures only; every output carries the label and the ledger records the plan
         "fixture-test": dict(label="FIXTURE TEST (tiny plan; synthetic fixture, not results)",
                              learners=("transformer", "bilstm", "lgbm"), n_search=2, seeds=C.SEEDS[:2],
                              dropout_grid=(0.5, 1.0), epochs=1, patience=1, n_boot=99, n_ig=5, n_perm=9)}
OSF = {"doi": "10.17605/OSF.IO/K5EPU", "project": "https://osf.io/f5gtb", "registered": "2026-11-07",
       "archive": "https://archive.org/details/osf-registrations-k5epu-v1"}
# Deviations from the registered protocol: each is written to the ledger (and time-stamped) before the first test score
DEVIATIONS = {"D1": {
    "decided": "2026-11-08 (India Standard Time, UTC+05:30)",
    "title": "primary near-duplicate threshold tau_J = 0.8; the registered antimode rule becomes a sensitivity analysis",
    "registered": "a test trace is removed when its largest Jaccard index (API 3-gram sets) to the training and "
                  "validation union of its origin is at least tau_J, the antimode of the kernel density of "
                  "nearest-neighbour similarities on [0.5, 0.95], fallback 0.8 (S8 equation 9)",
    "seen_before_the_decision": "only the counts of the first assembly (7 Nov 2026; no model trained or scored): "
                                "the rule gave tau_J = 0.535, 0.532 and 0.528 at origins 1-3; it kept 167 of 1,001, "
                                "127 of 848 and 157 of 649 shared-family MELD-DS-448 test traces (76-85% removed) "
                                "and 32 of 1,001, 34 of 1,018 and 31 of 1,007 WinMET test traces (97% removed)",
    "change": "every model scores every shared-family test trace; the primary analysis applies equation 9 with tau_J "
              "= 0.8, the registered fallback, which the registration also uses for WinMET's grouped split and for "
              "clustering WinMET test traces (S10-3); the registered antimode thresholds and tau_J = 0.7 and 0.9 "
              "(the sensitivity values of S8) are sensitivity analyses of the same models and predictions; P1, P2, "
              "P6, P7 and the environment-identity probe use the primary analysis's test traces, and P5, P8 and P9 "
              "are computed for every analysis",
    "rationale": "test sets of 31-34 WinMET and 127-167 MELD-DS-448 traces per origin cannot estimate "
                 "family-balanced macro-F1 over the shared families with useful precision; the rule exists to remove "
                 "near-copies of training traces, and at about 0.53 it removed most of each test set",
    "unchanged": "training and validation sets, family sets, models, searches, seeds, masks, estimands, cluster "
                 "bootstrap, decision rules, Holm adjustment, the H4 gate and the same-binary rule (S10-11)"}}


def jdump(path: Path, obj) -> str:
    blob = json.dumps(obj, indent=1, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    path.write_text(blob)
    return hashlib.sha256(blob.encode()).hexdigest()


def code_digests() -> dict:
    """SHA-256 of every source file of this engine, recorded in the ledger before scoring."""
    files = sorted(p for p in HERE.rglob("*") if p.is_file() and p.suffix in (".py", ".md", ".txt", ".yml")
                   and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts and "rehearsal" not in str(p))
    return {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def versions() -> dict:
    import lightgbm
    import sklearn
    import torch
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "lightgbm": lightgbm.__version__, "scikit_learn": sklearn.__version__, "platform": platform.platform(),
            "cpu_count": os.cpu_count()}


# ------------------------------------------------------------------------------------------------
# Results draft: every number is inserted by code from results.json
# ------------------------------------------------------------------------------------------------
def _kept_text(an: dict, names: dict) -> str:
    """Kept within-installation test traces per origin, e.g. 'MELD-DS-448 412 of 1,001, ...; WinMET ...'."""
    parts = []
    for inst in ("I1", "I2"):
        own = [(v["own"], v["own_of"]) for ck, v in an["kept"].items() if ck.startswith(inst + "_")]
        parts.append(f"{names[inst]} {', '.join(f'{a:,} of {b:,}' for a, b in own)}")
    return "; ".join(parts) + " shared-family test traces at origins " + ", ".join(
        sorted({ck.split("_o")[-1] for ck in an["kept"]}))


def results_draft(res: dict, names: dict) -> str:
    L, nb = res["confirmatory"], res["plan"]["n_boot"]
    rows = {r["component"]: r for r in res["decisions"][L]["rows"]}
    ver = res["decisions"][L]["verdict"]
    holm = res["decisions"][L]["holm_support"]
    f = lambda inst, role, where, mask: next(r for r in res["f1"] if (r["learner"], r["inst"], r["role"], r["where"],
                                                                      r["mask"]) == (L, inst, role, where, mask))
    ci = lambda c, d=1: f"{rows[c]['estimate']:.{d}f} (95% CI {rows[c]['ci_lo']:.{d}f} to {rows[c]['ci_hi']:.{d}f})"
    P = lambda h: f"P = {fmt_p(res['decisions'][L]['p_support'][h], nb)}, Holm-adjusted P = {holm[h]:.4f}"
    o = ["# Results draft (numbers inserted by run_s10.py; wording to be finalised at S12)", ""]
    if "SYNTHETIC" in res["label"]:
        o += [f"> **{res['label']}. Synthetic data with planted ground truth: these numbers are not results and "
              "must never enter the manuscript.**", ""]
    o += [f"Confirmatory learner: {LEARN[L]}. Estimates are family-balanced macro-F1 points averaged over origins and "
          f"{len(res['plan']['seeds'])} seeds; intervals are 95% cluster-bootstrap percentile intervals "
          f"({nb:,} replicates); P values are one-sided bootstrap P values, combined by the intersection-union rule "
          "and adjusted by Holm's method over H1-H4."]
    if res.get("primary_analysis", "as_built") != "as_built":
        o += ["", f"Test traces were kept when their largest Jaccard index to the training union of their origin was "
                  f"below tau_J = {res['primary_analysis']} and their binary had no report in that union (deviation "
                  f"D1 from the registered antimode rule, recorded before any test trace was scored; Methods): "
                  f"{_kept_text(res['analysis'], names)}."]
    o += ["", "## Artefact tokens alone identify families", ""]
    for i in ("I1", "I2"):
        o.append(f"In {names[i]}, the artefact-only probe reached {f(i, 'art', 'own', 'B')['f1']:.1f} points with "
                 f"behaviour tokens masked, against {f(i, 'len', 'own', 'none')['f1']:.1f} for the length-only probe: "
                 f"a difference of {ci(f'psi1|{i}')} points.")
    o += [f"H1 was {ver['H1']} ({P('H1')}).", "", "## Masking footprint and probing tokens", ""]
    for i in ("I1", "I2"):
        o.append(f"In {names[i]}, masking F and P tokens at test lowered macro-F1 from "
                 f"{f(i, 'base', 'own', 'none')['f1']:.1f} to {f(i, 'base', 'own', 'FP')['f1']:.1f} points, whereas a "
                 f"matched random mask of the same size left {f(i, 'base', 'own', 'rand_FP')['f1']:.1f}; the excess "
                 f"drop was {ci(f'psi2|{i}')} points.")
    o += [f"H2 was {ver['H2']} ({P('H2')}).", "", "## Transfer between installations", ""]
    for i, j in (("I1", "I2"), ("I2", "I1")):
        o.append(f"Models trained on {names[i]} scored {f(i, 'base', 'own', 'none')['f1']:.1f} points within and "
                 f"{f(i, 'base', 'other', 'none')['f1']:.1f} on {names[j]}, a gap of {ci(f'psi3|{i}')} points.")
    o += [f"H3 was {ver['H3']} ({P('H3')}).", "", "## Training without footprint tokens", ""]
    if ver["H4"] == "not tested":
        o.append("H4 was not tested, because H3 was not supported (the pre-registered gate).")
    else:
        o.append(f"F-masked training recovered a share rho = {ci('rho', 3)} of the pooled gap and changed "
                 f"within-installation macro-F1 by {ci('psi4b|I1')} points in {names['I1']} and "
                 f"{ci('psi4b|I2')} in {names['I2']}. H4 was {ver['H4']} ({P('H4')}).")
    ch = res["checks"]
    deep = [k for k in ch if k.endswith("|" + L)]
    p1 = all(ch[k]["P1_holds"] for k in deep)
    m_final = sorted({ch[k]["P2"]["m_final"] for k in deep})
    ref = [ch[k]["referral"]["rate"] for k in deep]
    o += ["", "## Explanations, referral and environment identity", "",
          f"Integrated Gradients gave exactly zero attribution at masked positions in every cell (P1 "
          f"{'held' if p1 else 'failed'}); the completeness criterion was met with m = {', '.join(map(str, m_final))} "
          f"nodes. The base model's mean footprint share of attribution was "
          f"{100 * np.mean([ch[k]['phi_base']['F'] for k in deep]):.1f}%. The referral rate on within-installation "
          f"test traces ranged from {100 * min(ref):.1f}% to {100 * max(ref):.1f}% against the 5% budget."]
    env = [(k, ch[k]["env_id"]) for k in deep if ch[k]["env_id"].get("tested")]
    for k, e in env:
        o.append(f"In {k.split('|')[0]}, a probe on frozen embeddings identified the analysis machine with a balanced "
                 f"accuracy of {100 * e['balanced_accuracy']:.1f}% against {100 * e['chance']:.0f}% chance "
                 f"(permutation P = {e['p_permutation']:.3f}).")
    o += ["", "## Secondary learners and checks", ""]
    for ln in res["decisions"]:
        if ln == L:
            continue
        r2 = {r["component"]: r for r in res["decisions"][ln]["rows"]}
        o.append(f"{LEARN[ln]} (secondary, unadjusted): psi2 = {r2['psi2|I1']['estimate']:.1f} and "
                 f"{r2['psi2|I2']['estimate']:.1f} points; transfer gaps {r2['psi3|I1']['estimate']:.1f} and "
                 f"{r2['psi3|I2']['estimate']:.1f}; rho = {r2['rho']['estimate']:.3f}.")
    de = res["design_effect"]
    o.append("Design effects of the test sets: " + ", ".join(f"{k} {v:.2f}" for k, v in de.items()) +
             f" (P5 bound 1.2). Matched random masks borrowed at most "
             f"{100 * max(c['rand_mask']['test_borrowed_share'] for c in res['cells'].values()):.1f}% of positions "
             "(P8 bound 10%).")
    p9 = [ch[k]["P9"] for k in deep if "P9" in ch[k]]
    if p9:
        o.append(f"Over the monthly windows after its origin ({', '.join(p9[0]['series'])}), the origin-1 model's "
                 f"macro-F1 had Kendall tau = {p9[0].get('tau', float('nan')):.2f} against time (one-sided P for a "
                 f"decline = {p9[0].get('p_one_sided', float('nan')):.4f}; P9 predicts tau < 0); "
                 f"AUT = {p9[0]['AUT']:.1f} points.")
    if res.get("sensitivity"):
        o += ["", "## Sensitivity to the near-duplicate threshold (deviation D1)", ""]
        H = ("H1", "H2", "H3", "H4")
        vp = res["decisions"][L]["verdict"]
        for key, sr in res["sensitivity"].items():
            an, vs, psi = sr["analysis"], sr["decisions"][L]["verdict"], sr["psi"][L]
            taus = [an["tau_J_by_origin"][k] for k in sorted(an["tau_J_by_origin"])]
            where = (f"the registered thresholds (tau_J = {', '.join(f'{t:.3f}' for t in taus)} at origins "
                     f"{', '.join(sorted(an['tau_J_by_origin']))})" if key == "registered" else f"tau_J = {key}")
            diff = [h for h in H if vs[h] != vp[h]]
            o.append(f"At {where} the rule kept {_kept_text(an, names)}. " + "; ".join(f"{h} was {vs[h]}" for h in H)
                     + (" (as in the primary analysis)" if not diff else
                        f" (unlike the primary analysis for {', '.join(diff)})")
                     + f"; psi2 = {psi['psi2|I1']:.1f} and {psi['psi2|I2']:.1f} points, transfer gaps "
                       f"{psi['psi3|I1']:.1f} and {psi['psi3|I2']:.1f} points, rho = {psi['rho']:.3f}.")
        o.append("Supplementary Table S_tau lists every estimate with its 95% interval for each threshold.")
    o += ["", "## Discussion skeleton", "",
          "[Authors: interpret the four verdicts above against S6's predictions, compare with Table 1, then state the "
          "threats to validity (single family taxonomy per installation, label noise, sequence cap, lexicon "
          "boundary choices) and the limitations recorded at S9.]"]
    return "\n".join(o) + "\n"


def write_outputs(res: dict, out: Path, names: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    jdump(out / "results.json", res)
    make_all(res, out, names)
    (out / "results_draft.md").write_text(results_draft(res, names))


# ------------------------------------------------------------------------------------------------
# Rehearsal on synthetic corpora
# ------------------------------------------------------------------------------------------------
def rehearsal_plan(a) -> audit.Plan:
    return audit.Plan(label=f"SYNTHETIC REHEARSAL ({a.scenario})", learners=tuple(a.learners.split(",")),
                      confirmatory="transformer", n_search=a.n_search, seeds=tuple(C.SEEDS[:a.n_seeds]),
                      dropout_grid=(0.5, 1.0), epochs=a.epochs, patience=2, n_boot=a.n_boot, n_ig=a.n_ig,
                      n_perm=a.n_perm)


def rehearsal(a) -> dict:
    from a15 import s10data, s10jobs
    corpus, V = synthetic.make_corpus(a.scenario, n_per_family=(a.n1, a.n2), seed=a.seed)
    sets, cells = synthetic.build_cells(corpus, V, seed=a.seed)
    plan = rehearsal_plan(a)
    out = Path(a.root) / "outputs" / f"s10_rehearsal_{a.scenario}"
    if a.engine == "inprocess":
        res = audit.run(sets, cells, plan)
    else:                                                      # the evaluation's job engine on synthetic cells
        t0 = time.perf_counter()
        A = s10data.Assembly(sets, cells, audit.fit_groups(cells))
        A.digest = s10data.assembly_digest(A)
        wd = Path(a.root) / "work" / f"s10_rehearsal_{a.scenario}"
        apath = wd / "assembly.pkl"
        s10data.save_assembly(A, apath)
        s10jobs.run_all(A, plan, wd / "jobs", workers=a.workers, threads=a.threads, assembly_path=apath,
                        progress_path=wd / "progress.jsonl", max_jobs=a.max_jobs)
        if a.max_jobs is not None and s10jobs.status(wd / "jobs", A.groups, {g: 1 for g in A.groups}, plan,
                                                     1)["done"] < len(s10jobs.job_list(A.groups, plan)):
            print(json.dumps({"stopped_after_jobs": a.max_jobs}))
            return {}
        preds, fits_info, checks, rand_info, monthly = s10jobs.collect(A, plan, wd / "jobs")
        res = audit.analyse_all(sets, cells, plan, preds, fits_info, checks, rand_info, monthly, t0)
    write_outputs(res, out, {"I1": "synthetic I1", "I2": "synthetic I2"})
    print(json.dumps({"verdicts": res["decisions"]["transformer"]["verdict"], "seconds": res["seconds"]}))
    return res


# ------------------------------------------------------------------------------------------------
# Evaluation: guard, assembly from the sealed S9 records, jobs, results, ledger entries
# ------------------------------------------------------------------------------------------------
def guard(ledger, allow_untrusted: bool) -> dict:
    if ledger.v_frz() != 1:
        raise SystemExit("V_frz = 0: the freeze is missing, unanchored or the ledger fails to verify; no scoring")
    frz = next(r["entry"] for r in ledger.records if r["entry"]["action"] == "freeze")
    anc = next(r["entry"] for r in ledger.records if r["entry"]["action"] == "anchor"
               and r["entry"]["batch_start"] <= frz["i"] <= r["entry"]["batch_end"])
    if not anc["trusted"] and not allow_untrusted:
        raise SystemExit("the freeze anchor comes from the local development TSA; anchor with an external TSA first")
    return {"freeze_entry": frz["i"], "anchor_entry": anc["i"], "tsa": anc["tsa"], "tsa_time": anc["tsa_time"]}


def _plan_record(plan: audit.Plan) -> dict:
    return json.loads(json.dumps(asdict(plan), default=list))


def _tau_label(key: str, taus: dict) -> str:
    return (f"{taus[key]:.3f} (registered)" if key == "registered" else
            f"{key} (primary)" if key == audit.Plan.primary else key)


def assembly_summary(A) -> dict:
    """What the frozen rules keep at every tau_J analysed (printed by `assemble`, stored in its ledger entry)."""
    out = {}
    for o, v in A.info["origins"].items():
        out[f"origin {o}"] = {"registered tau_J": v["tau_J_registered"], "rule": v["tau_rule"].get("rule"),
                              **{n: f"{k['in_families']:,} shared-family test traces ({k['same_binary']:,} same "
                                    "binary); kept at tau_J " + ", ".join(f"{_tau_label(key, v['taus'])}: {c:,}"
                                                                         for key, c in k["kept"].items())
                                 for n, k in v["test_keep"].items()}}
    out["cells"] = {ck: f"train {c['n_train']:,}, validation {c['n_val']:,}, L {c['L']}, vocabulary {c['V']:,}, "
                        f"fit group {c['fit_group']}; scored test traces: own {c['n_own_all']:,}, other "
                        f"{c['n_other_all']:,}" for ck, c in A.info["cells"].items()}
    o1 = A.info["origins"][next(iter(A.info["origins"]))]["taus"]
    out["P9_months"] = {m: f"{v['all']:,} traces; kept at tau_J " + ", ".join(f"{_tau_label(k, o1)}: {c:,}"
                                                                              for k, c in v["kept"].items())
                        for m, v in (A.info.get("P9_months") or {}).items()}
    out["lexicon"] = A.info["lexicon"]
    out["format"] = A.info.get("format")
    return out


def record_deviations(ledger) -> list:
    """Write each deviation not yet in the ledger, then anchor it with the time-stamp authority. Refused once
    scoring has begun: a deviation must be on record before the first test score."""
    have = {r["entry"].get("evidence_id") for r in ledger.records if r["entry"]["action"] == "deviation"}
    scored = any(r["entry"]["action"] == "test_score" for r in ledger.records)
    new = []
    for did, d in DEVIATIONS.items():
        if did in have:
            continue
        if scored:
            raise SystemExit(f"deviation {did} is not in the ledger and scoring has begun; nothing recorded")
        ledger.append("deviation", evidence_id=did, d_f=hashlib.sha256(json.dumps(d, sort_keys=True).encode())
                      .hexdigest(), payload=d, note=f"deviation {did}: {d['title']}")
        new.append(did)
    if new:
        anc = ledger.anchor()
        e = anc["entry"] if anc else {}
        print(json.dumps({"deviations_recorded": new, "anchor_entry": e.get("i"), "trusted": e.get("trusted"),
                          "tsa_time": e.get("tsa_time")}), flush=True)
        if anc and not e.get("trusted"):
            print("WARNING: the time-stamp authority was not reached; the deviation is anchored locally only "
                  f"({e.get('pending')}). It still precedes every test score in the hash chain.", flush=True)
    return new


def _assembly(P, frozen: dict, ledger, apath: Path, wd: Path):
    """The assembly of `run_s10.py assemble` if it is on disk and matches its ledger entry; else built now."""
    from a15 import s10data
    rec = [r["entry"] for r in ledger.records if r["entry"]["action"] == "s10_assembly"]
    if apath.exists() and rec:
        A = s10data.load_assembly(apath, rec[-1]["d_f"])
        if A.info.get("format") == s10data.ASSEMBLY_FORMAT:
            return A, s10data.sha256_file(apath)
        print("the assembly on disk was built by an earlier engine (before deviation D1): assembling again", flush=True)
    A = s10data.assemble(P, frozen)
    fsha = s10data.save_assembly(A, apath)
    jdump(wd / "assembly_info.json", {**A.info, "digest": A.digest, "file_sha256": fsha, "groups": A.groups})
    ledger.append("s10_assembly", evidence_id="S10", d_f=A.digest, payload=assembly_summary(A),
                  note="test sets and cells assembled from the sealed records; no model trained or scored")
    return A, fsha


def assemble(a) -> None:
    """Build the test sets and cells only (no model is trained or scored) and print what the frozen rules kept."""
    import run_s9 as R9
    from a15 import s10data
    P = C.Paths(Path(a.root))
    wd = P.work / "s10"
    wd.mkdir(parents=True, exist_ok=True)
    ledger = R9.open_ledger(P, a.tsa)
    guard(ledger, a.allow_untrusted_anchor)
    if any(r["entry"]["action"] == "test_score" for r in ledger.records):
        raise SystemExit("scoring has begun; the assembly is fixed (run evaluate to continue)")
    A, _ = _assembly(P, s10data.frozen_inputs(P, ledger), ledger, wd / "assembly.pkl", wd)
    record_deviations(ledger)
    print(json.dumps(assembly_summary(A), indent=1), flush=True)


def evaluate(a) -> None:
    import run_s9 as R9
    from a15 import s10data, s10jobs
    t0 = time.perf_counter()
    P = C.Paths(Path(a.root))
    wd, out = P.work / "s10", P.out / "s10"
    wd.mkdir(parents=True, exist_ok=True)
    apath, jobdir = wd / "assembly.pkl", wd / "jobs"
    plan = audit.Plan(**PLANS[a.plan])
    ledger = R9.open_ledger(P, a.tsa)
    g = guard(ledger, a.allow_untrusted_anchor)
    frozen = s10data.frozen_inputs(P, ledger)
    first = [r["entry"] for r in ledger.records if r["entry"]["action"] == "test_score"
             and r["entry"].get("evidence_id") == "S10"]
    if first:                                                  # restart: same plan, same assembly, finished jobs kept
        rec = json.loads(first[0]["payload"])
        if rec["plan"] != _plan_record(plan):
            raise SystemExit("the analysis plan differs from the one recorded when scoring began; nothing run")
        if apath.exists():
            A = s10data.load_assembly(apath, rec["assembly_digest"])
        else:
            print("assembly file missing: rebuilding it and checking it against the ledger", flush=True)
            A = s10data.assemble(P, frozen)
            if A.digest != rec["assembly_digest"]:
                raise SystemExit("the rebuilt assembly differs from the one recorded when scoring began")
            s10data.save_assembly(A, apath)
        ledger.append("s10_resume", evidence_id="S10", note="evaluation restarted; finished jobs are kept",
                      jobs_done=sum(1 for j in s10jobs.job_list(A.groups, plan) if s10jobs.job_path(jobdir, j).exists()),
                      workers=a.workers, threads=a.threads)
    else:
        A, fsha = _assembly(P, frozen, ledger, apath, wd)
        record_deviations(ledger)
        dev = {r["entry"]["evidence_id"]: {"entry": r["entry"]["i"], "d_f": r["entry"]["d_f"]}
               for r in ledger.records if r["entry"]["action"] == "deviation"}
        payload = {"plan": _plan_record(plan), "guard": g, "osf_registration": OSF if a.osf else None,
                   "deviations": dev,
                   "frozen_files_verified": frozen["digests"], "assembly_digest": A.digest,
                   "assembly_file_sha256": fsha,
                   "hook_set_sha256": s10data.sha256_file(P.work / "hook_set.json"),
                   "meld_manifest_sha256": s10data.sha256_file(P.data / "I1" / "malicious_dataset_manifest.csv"),
                   "code_sha256": code_digests(), "software": versions(), "workers": a.workers,
                   "threads": a.threads}
        ledger.append("test_score", evidence_id="S10", payload=payload, note="first scoring of test traces")
    groups, n_train, digest = A.groups, {gid: len(A.cells[cks[0]].train) for gid, cks in A.groups.items()}, A.digest
    jdump(wd / "job_plan.json", {"groups": groups, "n_train": n_train, "plan": _plan_record(plan), "digest": digest})
    del ledger                                                 # 130,000 entries: freed while the jobs run
    if a.workers > 1:                                          # each worker loads its own copy of the assembly
        A = None
        import gc
        gc.collect()
    r = s10jobs.run_all(A, plan, jobdir, workers=a.workers, threads=a.threads, assembly_path=apath,
                        progress_path=wd / "progress.jsonl", max_jobs=a.max_jobs, groups=groups, n_train=n_train,
                        digest=digest)
    if r["remaining"]:
        print(json.dumps({"stopped": r}), flush=True)
        return
    A = s10data.load_assembly(apath, digest)
    preds, fits_info, checks, rand_info, monthly = s10jobs.collect(A, plan, jobdir)
    res = audit.analyse_all(A.sets, A.cells, plan, preds, fits_info, checks, rand_info, monthly, t0)
    res["assembly"] = {**A.info, "digest": A.digest}
    res["registration"] = OSF if a.osf else None
    write_outputs(res, out, {"I1": "MELD-DS-448", "I2": "WinMET"})
    digests = {p.name: s10data.sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()}
    ledger = R9.open_ledger(P, a.tsa)
    ledger.append("results", evidence_id="S10", d_f=digests["results.json"], payload=digests,
                  note="S10 results, tables and figures")
    anc = ledger.anchor()
    print(json.dumps({"verdicts": res["decisions"][res["confirmatory"]]["verdict"],
                      "anchor": anc["entry"]["i"] if anc else None, "trusted": anc["entry"]["trusted"] if anc else None,
                      "verify": ledger.verify()}, indent=1))


def status(a) -> None:
    from a15 import s10jobs
    P = C.Paths(Path(a.root))
    wd = P.work / "s10"
    jp = json.loads((wd / "job_plan.json").read_text())
    plan = audit.Plan(**{k: tuple(v) if isinstance(v, list) else v for k, v in jp["plan"].items()})
    st = s10jobs.status(wd / "jobs", jp["groups"], jp["n_train"], plan, a.workers)
    last = None
    if (wd / "progress.jsonl").exists():
        lines = (wd / "progress.jsonl").read_text().splitlines()
        last = json.loads(lines[-1]) if lines else None
    print(json.dumps({**st, "last_finished": last}, indent=1))


def main():
    sys.stdout.reconfigure(line_buffering=True)   # progress lines reach `tee` and the log at once (7 Nov 2026)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tsa", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rehearsal")
    r.add_argument("--scenario", choices=["planted", "null"], required=True)
    r.add_argument("--learners", default="transformer,bilstm,cnn,lgbm")
    r.add_argument("--engine", choices=["inprocess", "jobs"], default="inprocess")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--max-jobs", type=int, default=None)
    r.add_argument("--n1", type=int, default=130)
    r.add_argument("--n2", type=int, default=100)
    r.add_argument("--n-search", type=int, default=2)
    r.add_argument("--n-seeds", type=int, default=2)
    r.add_argument("--epochs", type=int, default=6)
    r.add_argument("--n-boot", type=int, default=999)
    r.add_argument("--n-ig", type=int, default=30)
    r.add_argument("--n-perm", type=int, default=99)
    r.add_argument("--seed", type=int, default=7)
    e = sub.add_parser("evaluate")
    e.add_argument("--plan", choices=list(PLANS), default="protocol", help="'fixture-test' for code tests only")
    e.add_argument("--workers", type=int, default=1, help="parallel training processes")
    e.add_argument("--threads", type=int, default=int(os.environ.get("A15_THREADS") or 8),
                   help="PyTorch and LightGBM threads per process")
    e.add_argument("--allow-untrusted-anchor", action="store_true", help="test fixtures only")
    e.add_argument("--no-osf", dest="osf", action="store_false", help="test fixtures only")
    e.add_argument("--max-jobs", type=int, default=None, help="stop after this many jobs (tests of the restart)")
    m = sub.add_parser("assemble", help="test sets and cells only; prints the kept test traces (no scoring)")
    m.add_argument("--allow-untrusted-anchor", action="store_true", help="test fixtures only")
    s = sub.add_parser("status")
    s.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    {"rehearsal": rehearsal, "assemble": assemble, "evaluate": evaluate, "status": status}[a.cmd](a)


if __name__ == "__main__":
    main()
