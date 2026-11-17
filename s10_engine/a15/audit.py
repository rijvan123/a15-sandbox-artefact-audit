"""S10 engine: S7 Algorithm 4 with the S8 estimands (equation 15), cluster bootstrap (16), decision rules (17-20)
and the checks P1, P2 and P5-P9.

A cell is one installation at one origin. Every model of a cell is scored on its own test set (within) and on the
other installation's test set at the same origin (cross). Test sets are fixed per (set, origin) after near-duplicate
removal, so all models are scored on the same traces and every bootstrap replicate is paired across models.

Compute plan of amendments 3 and 5 (frozen in settings.json, 5 Nov 2026): the learners in Plan.triage (BiLSTM) are
trained unmasked and F-masked only. They have no random search, no dropout-rate selection and no artefact-only probe:
the configuration is the first draw of the registered search for the cell's origin, and the F-masked model uses
p = 1 (deterministic F-masking, equation 3). H1 is therefore "not tested" for a triaged learner (S10-13, 7 Nov 2026).
The length-only probe does not depend on the learner, so it is fitted once per cell and seed and shared (S10-14).

Deviation D1 (decided 7-8 Nov 2026, before any model was trained or scored): every model scores all shared-family
test traces, and the near-duplicate rule (equation 9) is applied at analysis time. The primary analysis uses the
registered fallback tau_J = 0.8; the registered antimode thresholds (0.535, 0.532, 0.528) and tau_J = 0.7 and 0.9
(the sensitivity values planned at S8) are sensitivity analyses of the same models (restrict(), analyse_all()).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import stats as st

from . import config as C
from . import stats as S
from .masks import apply_mask, band_edges, choose_grid, matched_random_mask, token_bands
from .models import (MASK_ID, NgramLGBM, Trace, class_shares, ig_population, integrated_gradients, predict,
                     sample_configs, train_deep)
from .probes import env_id_probe, length_matrix, length_probe, pooled_embeddings

DEEP = ("transformer", "bilstm", "cnn")
MASK_CLASSES = {"none": (), "F": ("F",), "P": ("P",), "FP": ("F", "P"), "B": ("B",)}
WITHIN = ("none", "F", "P", "FP", "rand_FP")
INSTS = ("I1", "I2")


@dataclass
class TestSet:
    name: str
    family: np.ndarray
    cluster: np.ndarray
    month: np.ndarray | None = None
    machine: np.ndarray | None = None
    lag: np.ndarray | None = None
    clusters: dict | None = None             # analysis key -> cluster ids at that analysis's tau_J (deviation D1)


@dataclass
class Cell:
    inst: str
    origin: int
    families: list
    V: int
    train: list
    val: list
    own: str
    own_idx: np.ndarray
    own_traces: list
    other: str
    other_idx: np.ndarray
    other_traces: list
    monthly: dict | None = None
    # Deviation D1: own and other traces are every shared-family test trace; each analysis keeps those whose largest
    # exact Jaccard index to the origin's training union is below its tau_J and whose binary is not in that union.
    own_maxj: np.ndarray | None = None
    own_same: np.ndarray | None = None
    other_maxj: np.ndarray | None = None
    other_same: np.ndarray | None = None
    monthly_maxj: dict | None = None
    monthly_same: dict | None = None
    tau_registered: float | None = None


@dataclass
class Plan:
    label: str = "protocol"
    learners: tuple = ("transformer", "bilstm", "cnn", "lgbm")
    confirmatory: str = "transformer"
    triage: tuple = ("bilstm",)                                   # unmasked and F-masked only (amendments 3 and 5)
    n_search: int = C.N_SEARCH
    search_fraction: float = 1.0
    seeds: tuple = C.SEEDS
    dropout_grid: tuple = C.DROPOUT_GRID
    epochs: int = 20
    patience: int = 3
    n_boot: int = C.N_BOOT
    n_ig: int = 200
    n_perm: int = 999
    boot_seed: int = 20251101
    analyses: tuple = ("0.8", "registered", "0.7", "0.9")         # deviation D1: tau_J of each analysis
    primary: str = "0.8"                                          # figures, Table 2 and the Results draft


@dataclass
class Fitted:
    kind: str
    cfg: dict
    p_star: float
    base: list
    mit: list
    art: list
    log: dict = field(default_factory=dict)


TRIAGE_P = 1.0                     # F-masked training of a triaged learner: p = 1, deterministic F-masking (eq. 3)
LGBM_THREADS: int | None = None    # set by the S10 job workers so that parallel LightGBM fits do not oversubscribe


# ------------------------------------------------------------------------------------------------
# Fitting (Algorithm 4: search, dropout rate, final seeds, probes)
# ------------------------------------------------------------------------------------------------
def class_masked(t: Trace, classes: tuple, p: float = 1.0, rng=None) -> Trace:
    ids, b, _ = apply_mask(t.ids, t.betas, t.cls, classes, MASK_ID, C.BETA_NEUTRAL, p=p, rng=rng)
    return Trace(ids, b, t.cls, t.y, t.length)


def search_configs(kind: str, plan: Plan, origin: int) -> list[dict]:
    return sample_configs(kind, plan.n_search, seed=1000 + origin)


def triage_config(kind: str, origin: int) -> dict:
    """Configuration of a triaged learner: the first draw of the registered search for this origin."""
    return sample_configs(kind, 1, seed=1000 + origin)[0]


def search_train(cell: Cell, plan: Plan) -> list:
    if plan.search_fraction < 1:                                  # cpu-reduced route: search on a sub-sample
        rng = np.random.default_rng(plan.seeds[0])
        return [t for t in cell.train if rng.random() < plan.search_fraction]
    return cell.train


def roles(kind: str, plan: Plan) -> tuple:
    return ("base", "mit") if kind in plan.triage else ("base", "mit", "art")


def _train(kind, cell, cfg, seed, plan, train=None, val=None, p_drop=0.0, val_mask=(), info: dict | None = None):
    train = cell.train if train is None else train
    val = cell.val if val is None else val
    K = len(cell.families)
    t0 = time.perf_counter()
    if kind == "lgbm":                                            # one Bernoulli draw per trace for dropout
        rng = np.random.default_rng(seed)
        tr = [class_masked(t, ("F",), p_drop, rng) for t in train] if p_drop > 0 else train
        model = NgramLGBM(K, {**cfg, "n_jobs": LGBM_THREADS} if LGBM_THREADS else cfg, seed).fit(tr, val)
        if info is not None:
            info.update({"seconds": time.perf_counter() - t0, "n_train": len(train),
                         "best_iteration": int(getattr(model.model, "best_iteration_", 0) or 0)})
        return model
    model, inf = train_deep(kind, cell.V, K, train, val, cfg, seed, p_drop=p_drop, val_mask=val_mask,
                            epochs=plan.epochs, patience=plan.patience)
    if info is not None:
        info.update({"seconds": inf["seconds"], "epochs_run": inf["epochs_run"], "best_val": inf["best_val"],
                     "n_train": len(train)})
    return model


def _pred(kind, model, traces, mask="none"):
    m = MASK_CLASSES.get(mask, ())
    return model.predict(traces, m) if kind == "lgbm" else predict(model, traces, m)


def fit_cell(kind: str, cell: Cell, plan: Plan) -> Fitted:
    """In-process Algorithm 4 for one learner and cell (rehearsal and tests; the evaluation runs it as jobs)."""
    K, t0 = len(cell.families), time.perf_counter()
    yv = np.array([t.y for t in cell.val])
    s0 = plan.seeds[0]
    if kind in plan.triage:
        cfg, p_star, scores, dscores = triage_config(kind, cell.origin), TRIAGE_P, [], []
    else:
        cfgs = search_configs(kind, plan, cell.origin)
        st_ = search_train(cell, plan)
        scores = [S.macro_f1(yv, _pred(kind, _train(kind, cell, c, s0, plan, train=st_), cell.val), K) for c in cfgs]
        cfg = cfgs[int(np.argmax(scores))]
        dscores = [S.macro_f1(yv, _pred(kind, _train(kind, cell, cfg, s0, plan, p_drop=p, val_mask=("F",)),
                                        cell.val, "F"), K) for p in plan.dropout_grid]
        p_star = plan.dropout_grid[int(np.argmax(dscores))]
    base = [_train(kind, cell, cfg, s, plan) for s in plan.seeds]
    mit = [_train(kind, cell, cfg, s, plan, p_drop=p_star, val_mask=("F",)) for s in plan.seeds]
    art = []
    if "art" in roles(kind, plan):
        trB = [class_masked(t, ("B",)) for t in cell.train]
        vaB = [class_masked(t, ("B",)) for t in cell.val]
        art = [_train(kind, cell, cfg, s, plan, train=trB, val=vaB) for s in plan.seeds]
    return Fitted(kind, cfg, p_star, base, mit, art,
                  {"search_val_f1": scores, "dropout_val_f1": dict(zip(map(str, plan.dropout_grid), dscores)),
                   "seconds": time.perf_counter() - t0, "triage": kind in plan.triage})


def length_predictions(cell: Cell, plan: Plan) -> np.ndarray:
    """f_len on the cell's own test traces, one row per seed (learner-independent, S10-14)."""
    K = len(cell.families)
    return np.stack([length_probe(cell.train, K, s).predict(length_matrix(cell.own_traces)) for s in plan.seeds])


# ------------------------------------------------------------------------------------------------
# Scoring: within masks (including the matched random mask) and cross-installation tests
# ------------------------------------------------------------------------------------------------
def rand_fp_traces(cell: Cell, seed: int) -> tuple[list, dict]:
    cnt = np.bincount(np.concatenate([t.ids for t in cell.train]), minlength=cell.V).astype(float)
    logf = np.log1p(cnt)
    type_logf = logf[cnt > 0]
    grid, shares = choose_grid([(t.ids, t.cls) for t in cell.train[:2000]], logf, type_logf, seed=seed)
    edges, gen = band_edges(type_logf, grid[0]), np.random.default_rng(seed)
    out, borrowed, masked = [], [], []
    for t in cell.own_traces:
        chosen, b, m = matched_random_mask(t.cls, token_bands(t.ids, logf, edges), ("F", "P"), gen, grid[1])
        ids, bet = t.ids.copy(), t.betas.copy()
        pos = np.fromiter(chosen, dtype=int, count=len(chosen))
        ids[pos], bet[pos] = MASK_ID, C.BETA_NEUTRAL
        out.append(Trace(ids, bet, t.cls, t.y, t.length))
        borrowed.append(b)
        masked.append(m)
    share = float(np.sum(borrowed)) / max(float(np.sum(masked)), 1.0)
    # per-trace counts (keys with "_" are dropped from results.json): P8 is recomputed on each analysis's kept traces
    return out, {"grid": list(grid), "train_borrowed_share": {str(k): v for k, v in shares.items()},
                 "test_borrowed_share": share, "P8_exact": share <= C.MASK_MAX_BORROW,
                 "_borrowed": np.asarray(borrowed, dtype=np.int64), "_masked": np.asarray(masked, dtype=np.int64)}


def model_predictions(kind: str, role: str, model, cell: Cell, rand_traces: list) -> dict:
    """One model's predictions on the cell's test traces, keyed by (role, where, mask)."""
    out = {}
    if role == "base":
        for mask in WITHIN:
            tr = rand_traces if mask == "rand_FP" else cell.own_traces
            out[("base", "own", mask)] = _pred(kind, model, tr, "none" if mask == "rand_FP" else mask)
        out[("base", "other", "none")] = _pred(kind, model, cell.other_traces)
    elif role == "mit":
        out[("mit", "own", "F")] = _pred(kind, model, cell.own_traces, "F")
        out[("mit", "other", "F")] = _pred(kind, model, cell.other_traces, "F")
    elif role == "art":
        out[("art", "own", "B")] = _pred(kind, model, cell.own_traces, "B")
    return out


def monthly_predictions(kind: str, model, cell: Cell) -> dict:
    return {m: _pred(kind, model, tr) for m, (y, tr) in sorted((cell.monthly or {}).items()) if len(y)}


def score_cell(cell: Cell, fit: Fitted, rand_traces: list) -> dict:
    """Predictions (n_seeds x n traces) keyed by (role, where, mask); the length probe is added by the caller."""
    pred = {}
    for role, models in (("base", fit.base), ("mit", fit.mit), ("art", fit.art)):
        per_seed = [model_predictions(fit.kind, role, m, cell, rand_traces) for m in models]
        for key in (per_seed[0] if per_seed else {}):
            pred[key] = np.stack([p[key] for p in per_seed])
    return pred


# ------------------------------------------------------------------------------------------------
# Estimands (equation 15), origin-averaged, from per-cell seed-mean macro-F1
# ------------------------------------------------------------------------------------------------
def psis_from_avg(A: dict, kind: str) -> dict:
    """A[(kind, inst, role, where, mask)] holds origin-averaged F1-bar in points."""
    g = lambda i, r, w, m: A[(kind, i, r, w, m)]
    out = {}
    for i in INSTS:
        if (kind, i, "art", "own", "B") in A:                     # absent for a triaged learner (no f_art)
            out[f"psi1|{i}"] = g(i, "art", "own", "B") - g(i, "len", "own", "none")
        out[f"psi2|{i}"] = g(i, "base", "own", "rand_FP") - g(i, "base", "own", "FP")
        out[f"psi3|{i}"] = g(i, "base", "own", "none") - g(i, "base", "other", "none")        # G for I -> I'
        out[f"psi4b|{i}"] = g(i, "mit", "own", "F") - g(i, "base", "own", "none")
    num = sum(g(i, "mit", "other", "F") - g(i, "base", "other", "none") for i in INSTS)
    den = sum(out[f"psi3|{i}"] for i in INSTS)
    out["rho"] = num / den if den != 0 else float("nan")
    return out


class Scorer:
    def __init__(self, sets: dict, cells: dict, preds: dict):
        self.sets, self.cells, self.preds = sets, cells, preds
        self.pos, self.y = {}, {}
        for ck, c in cells.items():
            for where, name, idx, traces in (("own", c.own, c.own_idx, c.own_traces),
                                             ("other", c.other, c.other_idx, c.other_traces)):
                pm = -np.ones(len(sets[name].family), dtype=int)
                pm[idx] = np.arange(len(idx))
                self.pos[(ck, where)] = pm
                self.y[(ck, where)] = np.array([t.y for t in traces])

    def averages(self, idx_by_set: dict) -> dict:
        per_cell = {}
        for (ck, kind, role, where, mask), P in self.preds.items():
            c = self.cells[ck]
            pos = self.pos[(ck, where)][idx_by_set[c.own if where == "own" else c.other]]
            pos = pos[pos >= 0]
            y = self.y[(ck, where)][pos]
            per_cell[(ck, kind, role, where, mask)] = float(np.mean([S.macro_f1(y, p[pos], len(c.families))
                                                                      for p in P]))
        A = {}
        for (ck, kind, role, where, mask), v in per_cell.items():
            A.setdefault((kind, self.cells[ck].inst, role, where, mask), []).append(v)
        return {k: float(np.mean(v)) for k, v in A.items()}


def bootstrap(sets: dict, scorer: Scorer, kinds: tuple, plan: Plan) -> tuple[dict, dict, dict]:
    names = list(sets)
    offs = np.cumsum([0] + [len(sets[n].family) for n in names])
    strata = np.concatenate([[f"{n}|{f}" for f in sets[n].family] for n in names])
    clusters = np.concatenate([[f"{n}|{c}" for c in sets[n].cluster] for n in names])
    boot = S.ClusterBootstrap(strata, clusters, n_boot=plan.n_boot, seed=plan.boot_seed)
    full = {n: np.arange(len(sets[n].family)) for n in names}
    A0 = scorer.averages(full)
    point = {kind: psis_from_avg(A0, kind) for kind in kinds}
    repsA = {k: np.empty(plan.n_boot) for k in A0}
    repsP = {kind: {k: np.empty(plan.n_boot) for k in point[kind]} for kind in kinds}
    for nu in range(plan.n_boot):
        g = boot.replicate(nu)
        idx = {n: g[(g >= offs[i]) & (g < offs[i + 1])] - offs[i] for i, n in enumerate(names)}
        A = scorer.averages(idx)
        for k, v in A.items():
            repsA[k][nu] = v
        for kind in kinds:
            for k, v in psis_from_avg(A, kind).items():
                repsP[kind][k][nu] = v
    deff = {n: S.ClusterBootstrap([f"{n}|{f}" for f in sets[n].family], [f"{n}|{c}" for c in sets[n].cluster],
                                  n_boot=1).design_effect() for n in names}
    return {"A": A0, "psi": point, "design_effect": deff}, {"A": repsA, "psi": repsP}, boot


# ------------------------------------------------------------------------------------------------
# Decision rules (equations 17-20)
# ------------------------------------------------------------------------------------------------
def decide(reps: dict, point: dict) -> dict:
    d, a = C.DELTA, C.ALPHA_ONE_SIDED
    r = lambda k: reps[k]
    comp = {"H1": [f"psi1|{i}" for i in INSTS], "H2": [f"psi2|{i}" for i in INSTS],
            "H3": [f"psi3|{i}" for i in INSTS], "H4": ["rho"] + [f"psi4b|{i}" for i in INSTS]}
    missing = {h: not all(k in reps for k in keys) for h, keys in comp.items()}   # H1 needs f_art (not triaged)
    nonfinite = {h: (not missing[h]) and not all(np.isfinite(point[k]) for k in keys) for h, keys in comp.items()}
    tested = {h: not (missing[h] or nonfinite[h]) for h in comp}                  # no kept test trace: not estimable
    nan = float("nan")
    p_sup = {"H1": S.iut_support([r(k) for k in comp["H1"]], d) if tested["H1"] else nan,
             "H2": S.iut_support([r(k) for k in comp["H2"]], d) if tested["H2"] else nan,
             "H3": S.iut_support([r(k) for k in comp["H3"]], d) if tested["H3"] else nan,
             "H4": S.h4_support_p(r("rho"), [r(f"psi4b|{i}") for i in INSTS]) if tested["H4"] else nan}
    p_rej = {"H1": S.bonferroni_rejection([r(k) for k in comp["H1"]], d) if tested["H1"] else nan,
             "H2": max(S.tost_equivalence(r(k), d) for k in comp["H2"]) if tested["H2"] else nan,
             "H3": S.bonferroni_rejection([r(k) for k in comp["H3"]], d) if tested["H3"] else nan,
             "H4": S.h4_rejection_p(r("rho")) if tested["H4"] else nan}
    v = S.verdicts(p_sup, p_rej, tested=tested)
    if v["verdict"]["H3"] != "supported":                         # H4 is tested only if H3 is supported
        v["verdict"]["H4"] = "not tested"
    for h in comp:
        if missing[h]:
            v["verdict"][h] = "not tested (compute triage: no artefact-only probe)"
        elif nonfinite[h]:
            v["verdict"][h] = "not estimable (a test set kept no trace)"
    rows = []
    for h, keys in comp.items():
        if not tested[h]:
            continue
        for k in keys:
            margin = C.RHO_BOUND if k == "rho" else (-d if k.startswith("psi4b") else d)
            lo, hi = S.percentile_ci(reps[k])
            rows.append({"hypothesis": h, "component": k, "estimate": point[k], "ci_lo": lo, "ci_hi": hi,
                         "margin": margin, "p_component": S.p_greater(reps[k], margin),
                         "p_support": p_sup[h], "p_rejection": p_rej[h],
                         "holm_support": v["holm_support"].get(h), "verdict": v["verdict"][h]})
    return {"rows": rows, "p_support": p_sup, "p_rejection": p_rej, **v}


# ------------------------------------------------------------------------------------------------
# Checks: P1, P2, P6, P7 (referral), P9 (monthly decline), environment-ID probe; secondary tests
# ------------------------------------------------------------------------------------------------
def explain_cell(cell: Cell, fit: Fitted, plan: Plan) -> dict:
    base0, mit0 = fit.base[0], fit.mit[0]
    ex = cell.own_traces[:plan.n_ig]
    pop = ig_population(base0, ex)
    popm = ig_population(mit0, ex, test_mask=("F",))
    shares_b = [class_shares(res["ig"], t.cls) for res, t in zip(pop["results"], ex)]
    shares_m = [class_shares(res["ig"], t.cls) for res, t in zip(popm["results"], ex)]
    p1 = max((float(np.abs(res["ig"][t.cls == 1]).max()) for res, t in zip(popm["results"], ex) if (t.cls == 1).any()),
             default=0.0)
    vphi = [class_shares(integrated_gradients(base0, t, m=pop["m"])["ig"], t.cls)["F"] for t in cell.val[:plan.n_ig]]
    theta = float(np.quantile(vphi, 1 - C.ETA_F)) if vphi else float("inf")
    phi_f = np.array([s["F"] for s in shares_b])
    referred = phi_f > theta
    err = predict(base0, ex) != np.array([t.y for t in ex])
    n = len(ex)
    lo, hi = st.binom.ppf(0.025, n, C.ETA_F) / n, st.binom.ppf(0.975, n, C.ETA_F) / n
    rng = np.random.default_rng(plan.boot_seed)
    diffs = []
    for _ in range(999):
        b = rng.integers(0, n, n)
        rf, er = referred[b], err[b]
        if rf.any() and (~rf).any():
            diffs.append(er[rf].mean() - er[~rf].mean())
    p6 = S.p_greater(np.array(diffs), 0.0) if diffs else 1.0
    mean = lambda L, c: float(np.mean([s[c] for s in L])) if L else float("nan")
    out = {"P1_max_abs_IG_masked": p1, "P1_holds": p1 == 0.0,
           "P2": {"m_final": pop["m"], "share_delta_ok": pop["share_ok"], "criterion_met": pop["criterion_met"]},
           "phi_base": {c: mean(shares_b, c) for c in "BFP"}, "phi_mit": {c: mean(shares_m, c) for c in "BFP"},
           "referral": {"theta_F": theta, "rate": float(referred.mean()), "binomial_95": [lo, hi],
                        "P7_holds": bool(lo <= referred.mean() <= hi),
                        "err_referred": float(err[referred].mean()) if referred.any() else None,
                        "err_accepted": float(err[~referred].mean()) if (~referred).any() else None,
                        "P6_p": p6, "P6_holds": p6 <= C.ALPHA_ONE_SIDED}, "n_explained": n}
    return out


def env_probe(cell: Cell, fit: Fitted, sets: dict, plan: Plan) -> dict:
    machines = sets[cell.own].machine
    if machines is None or fit.kind not in DEEP:
        return {"tested": False}
    mach = machines[cell.own_idx]
    if len(set(mach)) < 2:
        return {"tested": False, "machines": int(len(set(mach)))}
    emb = pooled_embeddings(fit.base[0], cell.own_traces)
    return env_id_probe(emb, mach, plan.boot_seed, n_perm=plan.n_perm)


def monthly_decline(cell: Cell, preds_by_month: dict) -> dict | None:
    """P9 from the base models' monthly predictions (preds_by_month[m]: n_seeds x n_m)."""
    if not cell.monthly or not preds_by_month:
        return None
    series = {m: float(np.mean([S.macro_f1(cell.monthly[m][0], p, len(cell.families)) for p in P]))
              for m, P in sorted(preds_by_month.items()) if len(cell.monthly[m][0])}
    if not series:
        return None
    vals = list(series.values())
    return {"series": series, "n_by_month": {m: int(len(cell.monthly[m][0])) for m in series}, "AUT": S.aut(vals),
            **(kendall_decline_exact(vals) if len(vals) > 2 else {})}


def kendall_decline_exact(series) -> dict:
    """P9: one-sided exact test of Kendall's tau < 0 against time, by enumerating every ordering of the series.

    Without ties this is scipy's exact test (S8: smallest attainable P 1/720 = 0.0014 for six months); with tied
    monthly values, where scipy's exact method stops, the same enumeration of tau-b stays exact (7 Nov 2026)."""
    import itertools
    s = np.asarray(series, dtype=float)
    x = np.arange(len(s))
    obs = st.kendalltau(x, s).statistic
    if not np.isfinite(obs) or len(s) > 8:
        return {**(S.kendall_decline(list(s)) if np.isfinite(obs) else {"tau": float("nan"), "p_one_sided": 1.0}),
                "method": "scipy" if np.isfinite(obs) else "constant series"}
    hits = total = 0
    for perm in itertools.permutations(range(len(s))):
        t = st.kendalltau(x, s[list(perm)]).statistic
        hits += bool(np.isfinite(t) and t <= obs + 1e-12)
        total += 1
    return {"tau": float(obs), "p_one_sided": hits / total, "method": "exact enumeration of orderings",
            "ties": bool(len(np.unique(s)) < len(s))}


def secondary_tests(cells: dict, preds: dict, kind: str) -> dict:
    """McNemar on seed-majority predictions (base vs mitigated, cross) and a Wilcoxon test over families."""
    n10 = n01 = 0
    d_base, d_mit = [], []
    for ck, c in cells.items():
        y = np.array([t.y for t in c.other_traces])
        vote = lambda P: np.array([np.bincount(col, minlength=len(c.families)).argmax() for col in P.T])
        b = vote(preds[(ck, kind, "base", "other", "none")]) == y
        m = vote(preds[(ck, kind, "mit", "other", "F")]) == y
        n10, n01 = n10 + int((b & ~m).sum()), n01 + int((~b & m).sum())
        K = len(c.families)
        fb = S.f1_per_family(S.confusion(y, preds[(ck, kind, "base", "other", "none")][0], K))
        fm = S.f1_per_family(S.confusion(y, preds[(ck, kind, "mit", "other", "F")][0], K))
        d_base += list(100 * fb)
        d_mit += list(100 * fm)
    diffs = np.array(d_mit) - np.array(d_base)
    w = st.wilcoxon(diffs) if np.count_nonzero(diffs) >= 6 else None
    return {"mcnemar_cross_base_vs_mit": {"n10": n10, "n01": n01, "p": S.mcnemar_exact(n10, n01)},
            "wilcoxon_families_cross": {"n": int(diffs.size), "p": float(w.pvalue) if w else None,
                                        "hodges_lehmann": S.hodges_lehmann(diffs) if diffs.size else None,
                                        "cliffs_delta": S.cliffs_delta(d_mit, d_base) if diffs.size else None}}


def fit_groups(cells: dict) -> dict:
    """Cells that share one training set share their fits (S10-5): group id -> cell keys, in cell order."""
    groups, seen = {}, {}
    for ck, cell in cells.items():
        key = (cell.inst, tuple(cell.families), id(cell.train))
        gid = seen.setdefault(key, ck)
        groups.setdefault(gid, []).append(ck)
    return groups


def run(sets: dict, cells: dict, plan: Plan, log=print) -> dict:
    """In-process evaluation (rehearsal and tests). run_s10 evaluate runs the same steps as restartable jobs."""
    t0 = time.perf_counter()
    preds, fits_info, checks, rand_info, cache, monthly_preds = {}, {}, {}, {}, {}, {}
    for ck, cell in cells.items():
        rtr, rand_info[ck] = rand_fp_traces(cell, plan.boot_seed)
        len_pred = length_predictions(cell, plan)
        for kind in plan.learners:
            key = (kind, cell.inst, tuple(cell.families), id(cell.train))
            f = cache[key] if key in cache else fit_cell(kind, cell, plan)   # same data, same models
            cache[key] = f
            fits_info[f"{ck}|{kind}"] = {"cfg": f.cfg, "p_star": f.p_star, **f.log}
            for k2, P in score_cell(cell, f, rtr).items():
                preds[(ck, kind) + k2] = P
            preds[(ck, kind, "len", "own", "none")] = len_pred
            log(f"[{time.perf_counter() - t0:7.1f}s] {ck} {kind}: cfg {f.cfg}, p* {f.p_star}, {f.log['seconds']:.1f}s")
            if kind in DEEP:                                   # checks on the primary analysis's test traces
                cp = restrict_cell(cell, plan.primary)[0]
                checks[(ck, kind)] = {**explain_cell(cp, f, plan), "env_id": env_probe(cp, f, sets, plan)}
                if cell.monthly:
                    monthly_preds[(ck, kind)] = {m: np.stack(v) for m, v in _stack_months(
                        [monthly_predictions(kind, b, cell) for b in f.base]).items()}
    return analyse_all(sets, cells, plan, preds, fits_info, checks, rand_info, monthly_preds, t0)


# ------------------------------------------------------------------------------------------------
# Deviation D1: one analysis per tau_J from the same predictions
# ------------------------------------------------------------------------------------------------
def analysis_tau(cell: Cell, key: str) -> float | None:
    """tau_J of an analysis for a cell's origin; None when the cell carries no near-duplicate data (synthetic)."""
    if key == "as_built" or cell.own_maxj is None:
        return None
    return float(cell.tau_registered) if key == "registered" else float(key)


def _keep(maxj: np.ndarray, same: np.ndarray, tau: float) -> np.ndarray:
    """Equation (9): test traces whose largest Jaccard index to the training union is below tau_J (S10-11: and whose
    binary has no report in that union)."""
    return np.flatnonzero((np.asarray(maxj) < tau) & ~np.asarray(same, dtype=bool))


def restrict_cell(c: Cell, key: str):
    """A cell at one analysis's tau_J: its kept test traces and their positions in the full cell (None: as built).
    Restricting a restricted cell again at the same tau_J changes nothing."""
    tau = analysis_tau(c, key)
    if tau is None:
        return c, None
    ko, kx = _keep(c.own_maxj, c.own_same, tau), _keep(c.other_maxj, c.other_same, tau)
    km = {m: _keep(c.monthly_maxj[m], c.monthly_same[m], tau) for m in (c.monthly or {})}
    monthly = ({m: (np.asarray(y)[km[m]], [tr[i] for i in km[m]]) for m, (y, tr) in c.monthly.items()}
               if c.monthly else c.monthly)
    sub = lambda a, k: np.asarray(a)[k]                                                    # noqa: E731
    return replace(c, own_idx=sub(c.own_idx, ko), own_traces=[c.own_traces[i] for i in ko],
                   other_idx=sub(c.other_idx, kx), other_traces=[c.other_traces[i] for i in kx], monthly=monthly,
                   own_maxj=sub(c.own_maxj, ko), own_same=sub(c.own_same, ko), other_maxj=sub(c.other_maxj, kx),
                   other_same=sub(c.other_same, kx),
                   monthly_maxj={m: sub(c.monthly_maxj[m], km[m]) for m in km} if c.monthly_maxj else c.monthly_maxj,
                   monthly_same={m: sub(c.monthly_same[m], km[m]) for m in km} if c.monthly_same else c.monthly_same
                   ), (ko, kx, km)


def restrict(sets: dict, cells: dict, preds: dict, monthly_preds: dict, rand_info: dict, key: str):
    """Test sets (clusters at the analysis's tau_J), cells, predictions and matched-mask records of one analysis."""
    new_cells, idx = {}, {}
    for ck, c in cells.items():
        new_cells[ck], idx[ck] = restrict_cell(c, key)
    new_preds = {}
    for k, P in preds.items():
        ix = idx[k[0]]
        new_preds[k] = P if ix is None else P[:, ix[0] if k[3] == "own" else ix[1]]
    new_monthly = {}
    for (ck, kind), mp in monthly_preds.items():
        ix = idx[ck]
        new_monthly[(ck, kind)] = mp if ix is None else {m: P[:, ix[2][m]] for m, P in mp.items()}
    new_rand = {}
    for ck, v in rand_info.items():
        r = {k: x for k, x in v.items() if not k.startswith("_")}
        ix = idx.get(ck)
        if ix is not None and "_borrowed" in v:                                 # P8 on the kept traces
            b, m = np.asarray(v["_borrowed"])[ix[0]], np.asarray(v["_masked"])[ix[0]]
            share = float(b.sum()) / max(float(m.sum()), 1.0)
            r.update(test_borrowed_share=share, P8_exact=share <= C.MASK_MAX_BORROW)
        new_rand[ck] = r
    new_sets = {n: (replace(s, cluster=s.clusters[key]) if s.clusters and key in s.clusters else s)
                for n, s in sets.items()}
    return new_sets, new_cells, new_preds, new_monthly, new_rand


def analyse_all(sets: dict, cells: dict, plan: Plan, preds: dict, fits_info: dict, checks: dict, rand_info: dict,
                monthly_preds: dict, t0: float) -> dict:
    """The primary analysis in full (figures, Table 2, Results draft) plus every other tau_J as a sensitivity
    analysis; all of them use the same models and predictions. P1, P2, P6, P7 and the environment probe come from
    the primary analysis's test traces (jobs 'explain'); P5, P8 and P9 are recomputed for every analysis."""
    real = any(c.own_maxj is not None for c in cells.values())
    keys = tuple(plan.analyses) if real else ("as_built",)
    primary = plan.primary if plan.primary in keys else keys[0]
    results = {}
    for key in keys:
        s2, c2, p2, m2, r2 = restrict(sets, cells, preds, monthly_preds, rand_info, key)
        ch = {k: dict(v) for k, v in checks.items()}
        for (ck, kind), mp in m2.items():
            md = monthly_decline(c2[ck], mp)
            if md and (ck, kind) in ch:
                ch[(ck, kind)]["P9"] = md
        res = summarise(s2, c2, plan, p2, fits_info, ch, r2, t0)
        res["analysis"] = {"key": key, "tau_J_by_origin": {str(c.origin): analysis_tau(c, key) for c in cells.values()},
                           "kept": {ck: {"own": len(c.own_traces), "other": len(c.other_traces),
                                         "own_of": len(cells[ck].own_traces), "other_of": len(cells[ck].other_traces)}
                                    for ck, c in c2.items()}}
        results[key] = res
    out = dict(results[primary])
    out["primary_analysis"] = primary
    out["sensitivity"] = {k: {"analysis": r["analysis"], "decisions": r["decisions"], "psi": r["psi"], "f1": r["f1"],
                              "design_effect": r["design_effect"], "P5_holds": r["P5_holds"],
                              "P8_max_test_borrowed_share": max((c["rand_mask"]["test_borrowed_share"]
                                                                 for c in r["cells"].values()), default=float("nan")),
                              "P9": {ck: v["P9"] for ck, v in r["checks"].items() if "P9" in v}}
                          for k, r in results.items() if k != primary}
    return out


def _stack_months(per_seed: list[dict]) -> dict:
    out = {}
    for p in per_seed:
        for m, v in p.items():
            out.setdefault(m, []).append(v)
    return out


def summarise(sets: dict, cells: dict, plan: Plan, preds: dict, fits_info: dict, checks: dict, rand_info: dict,
              t0: float) -> dict:
    """Estimands, cluster bootstrap, decisions and secondary tests from the predictions of every model."""
    scorer = Scorer(sets, cells, preds)
    point, reps, boot = bootstrap(sets, scorer, plan.learners, plan)
    decisions = {kind: decide(reps["psi"][kind], point["psi"][kind]) for kind in plan.learners}
    f1_rows = []
    for k, v in point["A"].items():
        lo, hi = S.percentile_ci(reps["A"][k])
        f1_rows.append({"learner": k[0], "inst": k[1], "role": k[2], "where": k[3], "mask": k[4], "f1": v,
                        "ci_lo": lo, "ci_hi": hi})
    return {"label": plan.label, "plan": {k: (list(v) if isinstance(v, tuple) else v) for k, v in plan.__dict__.items()},
            "cells": {ck: {"inst": c.inst, "origin": c.origin, "families": c.families, "n_train": len(c.train),
                           "n_val": len(c.val), "n_test_own": len(c.own_traces), "n_test_other": len(c.other_traces),
                           "rand_mask": rand_info[ck]} for ck, c in cells.items()},
            "fits": fits_info,
            "f1": f1_rows, "psi": point["psi"], "design_effect": point["design_effect"],
            "P5_holds": {n: v <= 1.2 for n, v in point["design_effect"].items()},
            "decisions": decisions, "confirmatory": plan.confirmatory,
            "checks": {f"{ck}|{k}": v for (ck, k), v in checks.items()},
            "secondary": {k: secondary_tests(cells, preds, k) for k in plan.learners},
            "seconds": time.perf_counter() - t0}
