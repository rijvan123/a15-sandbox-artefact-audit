"""S10 training and scoring as restartable jobs, run in this process or by parallel worker processes (7 Nov 2026).

A job trains one model of Algorithm 4 (a search configuration, a dropout rate, or a final model for one seed) or
runs the explanation checks of one learner, and writes its result to jobs/<group>/<learner>/<stage>_<index>.pkl in
one atomic step. A restart skips every job whose file exists, so an interruption (power cut, sleep, Ctrl-C) loses
only the jobs that were running. Models are not stored, except the seed-1 base and F-masked deep models that the
explanation job needs (.pt beside the job files).

Jobs reproduce audit.fit_cell exactly: the same configurations, seeds, selection rules (first maximum of validation
macro-F1) and predictions; tests/test_s10.py compares the two paths. The number of threads per process changes speed
and, through floating-point summation order, can change the last digits of a model's weights; it is recorded for
every job.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import audit
from . import stats as S

STAGE_RANK = {"search": 0, "dropout": 1, "base": 1, "art": 1, "mit": 2, "explain": 3}
# Relative cost per training trace and epoch (ms, dry run of 5 Nov 2026, 8 threads); only orders jobs and seeds the
# time estimate until measured job times replace it.
PRIOR_MS = {"transformer": 20.4, "bilstm": 39.3, "cnn": 2.3, "lgbm": 1.0}
PRIOR_EPOCHS = 10


@dataclass(frozen=True)
class Job:
    group: str
    kind: str
    stage: str
    index: int

    @property
    def key(self) -> str:
        return f"{self.group}/{self.kind}/{self.stage}_{self.index:02d}"


def job_path(jobdir: Path, j: Job) -> Path:
    return Path(jobdir) / j.group / j.kind / f"{j.stage}_{j.index:02d}.pkl"


def model_path(jobdir: Path, j: Job) -> Path:
    return job_path(jobdir, j).with_suffix(".pt")


def job_list(groups: dict, plan: audit.Plan) -> list[Job]:
    jobs = []
    for gid in groups:
        for kind in plan.learners:
            if kind not in plan.triage:
                jobs += [Job(gid, kind, "search", i) for i in range(plan.n_search)]
                jobs += [Job(gid, kind, "dropout", i) for i in range(len(plan.dropout_grid))]
            for role in audit.roles(kind, plan):
                jobs += [Job(gid, kind, role, i) for i in range(len(plan.seeds))]
            if kind in audit.DEEP:
                jobs.append(Job(gid, kind, "explain", 0))
    return jobs


def deps(j: Job, plan: audit.Plan) -> list[Job]:
    g, k = j.group, j.kind
    if k not in plan.triage and j.stage in ("dropout", "base", "art"):
        return [Job(g, k, "search", i) for i in range(plan.n_search)]
    if k not in plan.triage and j.stage == "mit":
        return [Job(g, k, "dropout", i) for i in range(len(plan.dropout_grid))]
    if j.stage == "explain":
        return [Job(g, k, "base", 0), Job(g, k, "mit", 0)]
    return []


def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=5)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def chosen_cfg(jobdir: Path, group: str, kind: str, plan: audit.Plan, origin: int) -> tuple[dict, list]:
    if kind in plan.triage:
        return audit.triage_config(kind, origin), []
    cfgs = audit.search_configs(kind, plan, origin)
    scores = [_load(job_path(jobdir, Job(group, kind, "search", i)))["score"] for i in range(plan.n_search)]
    return cfgs[int(np.argmax(scores))], scores


def chosen_p(jobdir: Path, group: str, kind: str, plan: audit.Plan) -> tuple[float, list]:
    if kind in plan.triage:
        return audit.TRIAGE_P, []
    dscores = [_load(job_path(jobdir, Job(group, kind, "dropout", i)))["score"] for i in range(len(plan.dropout_grid))]
    return plan.dropout_grid[int(np.argmax(dscores))], dscores


# ------------------------------------------------------------------------------------------------
# Worker state and one job
# ------------------------------------------------------------------------------------------------
_W: dict = {}


def init_state(A, plan: audit.Plan, jobdir: Path, threads: int) -> None:
    import torch
    torch.set_num_threads(max(1, threads))
    audit.LGBM_THREADS = max(1, threads)
    _W.update({"A": A, "plan": plan, "jobdir": Path(jobdir), "threads": threads, "rand": {}})


def init_worker(assembly_path: str, digest: str, plan_fields: dict, jobdir: str, threads: int) -> None:
    from . import s10data
    init_state(s10data.load_assembly(Path(assembly_path), digest), audit.Plan(**plan_fields), Path(jobdir), threads)


def _rand(ck: str):
    if ck not in _W["rand"]:
        _W["rand"][ck] = audit.rand_fp_traces(_W["A"].cells[ck], _W["plan"].boot_seed)[0]
    return _W["rand"][ck]


def _rebuild(kind: str, cell, cfg: dict, path: Path):
    import torch
    from .models import BUILDERS
    m = BUILDERS[kind](cell.V, len(cell.families), **{k: v for k, v in cfg.items() if k not in ("lr", "wd", "gamma",
                                                                                                "batch")})
    m.load_state_dict(torch.load(path, weights_only=True))
    m.eval()
    return m


def run_job(j: Job) -> dict:
    import torch
    A, plan, jobdir = _W["A"], _W["plan"], _W["jobdir"]
    t0 = time.perf_counter()
    cks = A.groups[j.group]
    cell = A.cells[cks[0]]
    K = len(cell.families)
    yv = np.array([t.y for t in cell.val])
    info: dict = {}
    out = {"job": j.key, "pid": os.getpid(), "threads": torch.get_num_threads(),
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if j.stage == "search":
        cfg = audit.search_configs(j.kind, plan, cell.origin)[j.index]
        model = audit._train(j.kind, cell, cfg, plan.seeds[0], plan, train=audit.search_train(cell, plan), info=info)
        out.update(cfg=cfg, score=S.macro_f1(yv, audit._pred(j.kind, model, cell.val), K))
    elif j.stage == "dropout":
        cfg, _ = chosen_cfg(jobdir, j.group, j.kind, plan, cell.origin)
        p = plan.dropout_grid[j.index]
        model = audit._train(j.kind, cell, cfg, plan.seeds[0], plan, p_drop=p, val_mask=("F",), info=info)
        out.update(p=p, score=S.macro_f1(yv, audit._pred(j.kind, model, cell.val, "F"), K))
    elif j.stage in ("base", "mit", "art"):
        cfg, _ = chosen_cfg(jobdir, j.group, j.kind, plan, cell.origin)
        seed = plan.seeds[j.index]
        if j.stage == "base":
            model = audit._train(j.kind, cell, cfg, seed, plan, info=info)
        elif j.stage == "mit":
            p, _ = chosen_p(jobdir, j.group, j.kind, plan)
            model = audit._train(j.kind, cell, cfg, seed, plan, p_drop=p, val_mask=("F",), info=info)
        else:
            trB = [audit.class_masked(t, ("B",)) for t in cell.train]
            vaB = [audit.class_masked(t, ("B",)) for t in cell.val]
            model = audit._train(j.kind, cell, cfg, seed, plan, train=trB, val=vaB, info=info)
        preds, monthly = {}, {}
        for ck in cks:
            c = A.cells[ck]
            rtr = _rand(ck) if j.stage == "base" else None
            for key, v in audit.model_predictions(j.kind, j.stage, model, c, rtr).items():
                preds[(ck,) + key] = np.asarray(v).astype(np.int16)
            if j.stage == "base" and c.monthly:
                monthly[ck] = {m: np.asarray(v).astype(np.int16)
                               for m, v in audit.monthly_predictions(j.kind, model, c).items()}
        out.update(seed=seed, preds=preds, monthly=monthly)
        if j.kind in audit.DEEP and j.index == 0 and j.stage in ("base", "mit"):
            mp = model_path(jobdir, j)
            mp.parent.mkdir(parents=True, exist_ok=True)
            tmp = mp.with_name(mp.name + f".tmp{os.getpid()}")
            torch.save(model.state_dict(), tmp)
            os.replace(tmp, mp)
    elif j.stage == "explain":
        cfg, _ = chosen_cfg(jobdir, j.group, j.kind, plan, cell.origin)
        p, _ = chosen_p(jobdir, j.group, j.kind, plan)
        base0 = _rebuild(j.kind, cell, cfg, model_path(jobdir, Job(j.group, j.kind, "base", 0)))
        mit0 = _rebuild(j.kind, cell, cfg, model_path(jobdir, Job(j.group, j.kind, "mit", 0)))
        fit = audit.Fitted(j.kind, cfg, p, [base0], [mit0], [])
        out["checks"] = {}
        for ck in cks:                            # checks on the primary analysis's test traces (deviation D1)
            cp = audit.restrict_cell(A.cells[ck], plan.primary)[0]
            out["checks"][ck] = {**audit.explain_cell(cp, fit, plan), "env_id": audit.env_probe(cp, fit, A.sets, plan)}
    else:
        raise ValueError(j.stage)
    out["train"] = info
    out["seconds"] = time.perf_counter() - t0
    _dump(job_path(jobdir, j), out)
    return {"job": j.key, "seconds": out["seconds"], "epochs_run": info.get("epochs_run"),
            "n_train": info.get("n_train"), "pid": out["pid"], "threads": out["threads"]}


# ------------------------------------------------------------------------------------------------
# Scheduling
# ------------------------------------------------------------------------------------------------
def n_train_of(A, group: str) -> int:
    return len(A.cells[A.groups[group][0]].train)


class Estimator:
    """Remaining wall time: measured seconds per training trace for each (learner, stage type), else the prior."""

    def __init__(self, n_train: dict, workers: int, n_ig: int = 200, n_perm: int = 999):
        self.n_train, self.workers = n_train, max(1, workers)
        self.n_ig, self.n_perm = n_ig, n_perm
        self.obs: dict = {}

    @staticmethod
    def kind_of(j: Job) -> tuple:
        return (j.kind, "explain" if j.stage == "explain" else "train")

    def add(self, j: Job, seconds: float) -> None:
        self.obs.setdefault(self.kind_of(j), []).append(seconds / max(self.n_train[j.group], 1))

    def seconds(self, j: Job) -> float:
        rate = self.obs.get(self.kind_of(j))
        n = max(self.n_train[j.group], 1)
        if rate:
            return float(np.mean(rate)) * n
        if j.stage == "explain":                  # about 3 n_ig traces x 50 nodes of IG, plus the permutation probe
            return PRIOR_MS[j.kind] / 1000 * self.n_ig * 75 + 0.1 * self.n_perm
        return PRIOR_MS[j.kind] / 1000 * n * PRIOR_EPOCHS * 1.35

    def remaining_hours(self, todo) -> float:
        return sum(self.seconds(j) for j in todo) / self.workers / 3600


def run_all(A, plan: audit.Plan, jobdir: Path, workers: int = 1, threads: int = 1, assembly_path: Path | None = None,
            log=print, progress_path: Path | None = None, max_jobs: int | None = None, groups: dict | None = None,
            n_train: dict | None = None, digest: str | None = None) -> dict:
    """Run every job not yet on disk; returns counts. max_jobs stops early (tests of the restart).

    With one worker the jobs run in this process on the assembly A. With several, A may be None: each worker loads
    the assembly file (checked against `digest`), and this process needs only the fit groups and training sizes."""
    jobdir = Path(jobdir)
    groups = groups if groups is not None else A.groups
    n_train = n_train if n_train is not None else {g: n_train_of(A, g) for g in groups}
    digest = digest if digest is not None else A.digest
    jobs = job_list(groups, plan)
    done = {j for j in jobs if job_path(jobdir, j).exists()}
    todo = [j for j in jobs if j not in done]
    est = Estimator(n_train, workers, plan.n_ig, plan.n_perm)
    prio = lambda j: (STAGE_RANK[j.stage], -est.seconds(j), j.key)                    # noqa: E731
    log(f"{time.strftime('%H:%M:%S')} S10 jobs: {len(jobs):,} in all, {len(done):,} already done, {len(todo):,} to "
        f"run on {workers} worker(s) x {threads} thread(s); first estimate {est.remaining_hours(todo):,.1f} h")
    t0, n_run = time.perf_counter(), 0

    def finished(j: Job, s: dict) -> None:
        nonlocal n_run
        done.add(j)
        n_run += 1
        est.add(j, s["seconds"])
        line = {**s, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if progress_path is not None:
            with open(progress_path, "a") as f:
                f.write(json.dumps(line) + "\n")
        log(f"{time.strftime('%H:%M:%S')} [{len(done):,}/{len(jobs):,}] {j.key} {s['seconds']:,.0f} s"
            + (f", {s['epochs_run']} epochs" if s.get("epochs_run") else "")
            + f"; about {est.remaining_hours(todo):,.1f} h left")

    def ready() -> list[Job]:
        return sorted((j for j in todo if all(d in done for d in deps(j, plan))), key=prio)

    if workers <= 1:
        if A is None:
            from . import s10data
            A = s10data.load_assembly(Path(assembly_path), digest)
        init_state(A, plan, jobdir, threads)
        while todo and (max_jobs is None or n_run < max_jobs):
            r = ready()
            if not r:
                raise RuntimeError("no job is ready although jobs remain (dependency error)")
            j = r[0]
            todo.remove(j)
            finished(j, run_job(j))
    else:
        import multiprocessing as mp
        if assembly_path is None:
            raise ValueError("parallel workers load the assembly from its file")
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "A15_THREADS"):
            os.environ[var] = str(threads)                     # inherited by the spawned workers only
        args = (str(assembly_path), digest, asdict(plan), str(jobdir), threads)
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=init_worker,
                                 initargs=args) as ex:
            running = {}
            while True:
                for j in ready():
                    if len(running) >= workers or (max_jobs is not None and n_run + len(running) >= max_jobs):
                        break
                    todo.remove(j)
                    running[ex.submit(run_job, j)] = j
                if not running:
                    if todo and (max_jobs is None or n_run < max_jobs):
                        raise RuntimeError("no job is ready although jobs remain (dependency error)")
                    break
                fin, _ = wait(running, return_when=FIRST_COMPLETED)
                for fut in fin:
                    j = running.pop(fut)
                    finished(j, fut.result())
    return {"jobs": len(jobs), "done": len(done), "ran_now": n_run, "remaining": len(jobs) - len(done),
            "hours_now": (time.perf_counter() - t0) / 3600}


# ------------------------------------------------------------------------------------------------
# Collecting the job files into the inputs of audit.summarise
# ------------------------------------------------------------------------------------------------
def collect(A, plan: audit.Plan, jobdir: Path) -> tuple[dict, dict, dict, dict, dict]:
    """Predictions, fit records, checks, matched-mask records and the P9 monthly predictions (deep learners), as
    audit.analyse_all takes them."""
    jobdir = Path(jobdir)
    preds, fits_info, checks, rand_info, monthly_preds = {}, {}, {}, {}, {}
    for gid, cks in A.groups.items():
        cell0 = A.cells[cks[0]]
        for kind in plan.learners:
            cfg, scores = chosen_cfg(jobdir, gid, kind, plan, cell0.origin)
            p, dscores = chosen_p(jobdir, gid, kind, plan)
            secs, epochs, monthly = 0.0, {}, {ck: {} for ck in cks}
            for st in ("search", "dropout"):
                if kind not in plan.triage:
                    n = plan.n_search if st == "search" else len(plan.dropout_grid)
                    for i in range(n):
                        o = _load(job_path(jobdir, Job(gid, kind, st, i)))
                        secs += o["seconds"]
                        epochs.setdefault(st, []).append(o["train"].get("epochs_run"))
            for role in audit.roles(kind, plan):
                outs = [_load(job_path(jobdir, Job(gid, kind, role, i))) for i in range(len(plan.seeds))]
                secs += sum(o["seconds"] for o in outs)
                epochs[role] = [o["train"].get("epochs_run") for o in outs]
                for key in outs[0]["preds"]:
                    preds[(key[0], kind) + key[1:]] = np.stack([o["preds"][key] for o in outs]).astype(np.int64)
                if role == "base":
                    for ck in cks:
                        for m in outs[0]["monthly"].get(ck, {}):
                            monthly[ck][m] = np.stack([o["monthly"][ck][m] for o in outs]).astype(np.int64)
            if kind in audit.DEEP:
                ex = _load(job_path(jobdir, Job(gid, kind, "explain", 0)))
                secs += ex["seconds"]
                for ck in cks:
                    checks[(ck, kind)] = ex["checks"][ck]
                    if monthly[ck]:
                        monthly_preds[(ck, kind)] = monthly[ck]
            for ck in cks:
                fits_info[f"{ck}|{kind}"] = {"cfg": cfg, "p_star": p, "search_val_f1": scores,
                                             "dropout_val_f1": dict(zip(map(str, plan.dropout_grid), dscores)),
                                             "seconds": secs, "triage": kind in plan.triage, "fit_group": gid,
                                             "epochs_run": epochs}
    for ck, c in A.cells.items():
        _, rand_info[ck] = audit.rand_fp_traces(c, plan.boot_seed)
        lp = audit.length_predictions(c, plan)
        for kind in plan.learners:
            preds[(ck, kind, "len", "own", "none")] = lp
    return preds, fits_info, checks, rand_info, monthly_preds


def status(jobdir: Path, groups: dict, n_train: dict, plan: audit.Plan, workers: int) -> dict:
    """Counts of finished jobs and the time estimate, from the job files alone (no assembly needed)."""
    jobs = job_list(groups, plan)
    done = [j for j in jobs if job_path(jobdir, j).exists()]
    est = Estimator(n_train, workers, plan.n_ig, plan.n_perm)
    by: dict = {}
    for j in done:
        o = _load(job_path(jobdir, j))
        est.add(j, o["seconds"])
        b = by.setdefault(j.kind, {"done": 0, "all": 0})
        b["done"] += 1
    for j in jobs:
        by.setdefault(j.kind, {"done": 0, "all": 0})["all"] += 1
    todo = [j for j in jobs if j not in set(done)]
    return {"done": len(done), "all": len(jobs), "by_learner": by, "hours_left_estimate": est.remaining_hours(todo)}
