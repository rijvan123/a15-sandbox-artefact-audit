"""Metrics and inference (S8 equations 11-20 and S5-S9). Every formula here was checked at S8 (C01-C22)."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy import stats as st

from .config import ALPHA_ONE_SIDED, DELTA, M_HOLM, N_BOOT, N_REQ_CF, N_REQ_DEFF, N_REQ_RECALL, RHO_BOUND


# ------------------------------------------------------------------------------------------------
# Metrics (equations 11-12)
# ------------------------------------------------------------------------------------------------
def confusion(y, yhat, K, w=None) -> np.ndarray:
    return np.bincount(np.asarray(y) * K + np.asarray(yhat), weights=w, minlength=K * K).reshape(K, K)


def f1_per_family(C: np.ndarray) -> np.ndarray:
    tp = np.diag(C)
    den = 2 * tp + (C.sum(0) - tp) + (C.sum(1) - tp)          # count form, defined when a family is never predicted
    return np.divide(2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den > 0)


def macro_f1(y, yhat, K, balanced: bool = True) -> float:
    """Macro-F1 in points over families present in y; balanced weights v_n = N / (K n_{y_n}) (decision D2)."""
    y, yhat = np.asarray(y), np.asarray(yhat)
    n_k = np.bincount(y, minlength=K)
    present = n_k > 0
    w = (len(y) / (present.sum() * n_k[y])) if balanced else None
    return 100.0 * f1_per_family(confusion(y, yhat, K, w))[present].mean()


def recall_far(y, yhat, K) -> tuple[np.ndarray, np.ndarray]:
    """Rec_k and FAR_k (equal alternative priors) for casework reporting, equation (12)."""
    C = confusion(y, yhat, K).astype(float)
    G = C / np.maximum(C.sum(1, keepdims=True), 1)
    rec = np.diag(G)
    far = (G.sum(0) - rec) / max(K - 1, 1)
    return rec, far


# ------------------------------------------------------------------------------------------------
# Cluster bootstrap (equation 16): strata (test set, origin, family), clusters within strata
# ------------------------------------------------------------------------------------------------
class ClusterBootstrap:
    def __init__(self, strata, clusters, n_boot: int = N_BOOT, seed: int = 20251101):
        groups = defaultdict(lambda: defaultdict(list))
        for i, (g, c) in enumerate(zip(strata, clusters)):
            groups[g][c].append(i)
        self.groups = [[np.array(m) for m in cl.values()] for cl in groups.values()]
        self.n_boot, self.seed = n_boot, seed

    def replicate(self, nu: int) -> np.ndarray:
        rng = np.random.default_rng([self.seed, nu])
        parts = []
        for members in self.groups:                         # as many clusters as the stratum holds
            pick = rng.integers(0, len(members), size=len(members))
            parts.extend(members[k] for k in pick)
        return np.concatenate(parts)

    def __iter__(self):
        for nu in range(self.n_boot):
            yield self.replicate(nu)

    def design_effect(self) -> float:
        sizes = np.array([len(m) for g in self.groups for m in g])
        return float((sizes ** 2).sum() / sizes.sum()) if sizes.size else 1.0   # mean cluster size seen by a trace


def seed_mean_f1(y, preds_by_seed, K, idx=None, balanced=True) -> float:
    """F1-bar(f; D*): family-balanced macro-F1 averaged over the n_s seeds, on replicate indices idx."""
    sel = slice(None) if idx is None else idx
    return float(np.mean([macro_f1(np.asarray(y)[sel], np.asarray(p)[sel], K, balanced) for p in preds_by_seed]))


def bootstrap_replicates(stat_fn, boot: ClusterBootstrap) -> np.ndarray:
    return np.array([stat_fn(idx) for idx in boot])


def percentile_ci(reps: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.percentile(reps, 100 * a)), float(np.percentile(reps, 100 * (1 - a)))


# ------------------------------------------------------------------------------------------------
# Decision rules (equations 17-20)
# ------------------------------------------------------------------------------------------------
def p_greater(reps: np.ndarray, psi0: float) -> float:          # claim psi > psi0
    return (1 + int(np.sum(reps <= psi0))) / (len(reps) + 1)


def p_less(reps: np.ndarray, psi0: float) -> float:             # claim psi < psi0
    return (1 + int(np.sum(reps >= psi0))) / (len(reps) + 1)


def iut_support(component_reps: list[np.ndarray], margin: float) -> float:
    return max(p_greater(r, margin) for r in component_reps)


def bonferroni_rejection(component_reps: list[np.ndarray], margin: float) -> float:
    return min(1.0, 2 * min(p_less(r, margin) for r in component_reps))


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    adj, running = [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def tost_equivalence(reps: np.ndarray, delta: float = DELTA) -> float:
    """Two one-sided tests: equivalence within +/- delta (H2's robustness verdict)."""
    return max(p_greater(reps, -delta), p_less(reps, delta))


def verdicts(p_sup: dict, p_rej: dict, tested: dict | None = None, alpha: float = ALPHA_ONE_SIDED) -> dict:
    """Equation (20): Holm over the four support P values; a second Holm family for rejections of the rest."""
    tested = tested or {h: True for h in p_sup}
    hs = [h for h in p_sup if tested.get(h, True)]
    sup_adj = dict(zip(hs, holm([p_sup[h] for h in hs])))
    rest = [h for h in hs if sup_adj[h] > alpha]
    rej_adj = dict(zip(rest, holm([p_rej[h] for h in rest]))) if rest else {}
    out = {}
    for h in p_sup:
        if not tested.get(h, True):
            out[h] = "not tested"
        elif sup_adj[h] <= alpha:
            out[h] = "supported"
        elif rej_adj.get(h, 1.0) <= alpha:
            out[h] = "rejected"
        else:
            out[h] = "inconclusive"
    return {"verdict": out, "holm_support": sup_adj, "holm_rejection": rej_adj}


# ------------------------------------------------------------------------------------------------
# Secondary tests and effect sizes (equations S7-S9)
# ------------------------------------------------------------------------------------------------
def mcnemar_exact(n10: int, n01: int) -> float:
    n = n10 + n01
    return 1.0 if n == 0 else min(1.0, 2 * st.binom.cdf(min(n10, n01), n, 0.5))


def hodges_lehmann(diffs) -> float:
    d = np.asarray(diffs, dtype=float)
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2))


def cliffs_delta(x, y) -> float:
    u = st.mannwhitneyu(x, y, alternative="two-sided").statistic          # ties count one half
    return float(2 * u / (len(x) * len(y)) - 1)


def aut(series) -> float:
    s = np.asarray(series, dtype=float)
    return float(np.mean((s[:-1] + s[1:]) / 2)) if len(s) > 1 else float(s[0])


def kendall_decline(series) -> dict:
    t = st.kendalltau(np.arange(len(series)), series, method="exact", alternative="less")
    return {"tau": float(t.statistic), "p_one_sided": float(t.pvalue)}


# ------------------------------------------------------------------------------------------------
# Planning formulas (equations S5-S6): K5 and prediction P4
# ------------------------------------------------------------------------------------------------
Z975 = st.norm.ppf(0.975)


def half_width(n_te: float, rec: float = N_REQ_RECALL, deff: float = N_REQ_DEFF, c_f: float = N_REQ_CF,
               n_sets: int = 2) -> float:
    """95% half-width, in points, of a gap between n_sets independent test sets of n_te traces each."""
    return float(100 * Z975 * math.sqrt(n_sets * c_f ** 2 * deff * rec * (1 - rec) / n_te))   # Python float


def n_required(rec: float = N_REQ_RECALL, deff: float = N_REQ_DEFF, c_f: float = N_REQ_CF,
               delta: float = DELTA) -> int:
    return int(math.ceil(2 * c_f ** 2 * deff * Z975 ** 2 * rec * (1 - rec) / (delta / 100) ** 2))


def delta_method_rho(var_n, var_g, cov_ng, rho, g_pool) -> float:
    return (var_n - 2 * rho * cov_ng + rho ** 2 * var_g) / g_pool ** 2


def h4_support_p(rho_reps, within_change_reps_by_inst: list[np.ndarray]) -> float:
    """H4 support: rho > 0.10 and non-inferiority (> -Delta) of the within-installation change everywhere."""
    return max([p_greater(rho_reps, RHO_BOUND)] + [p_greater(r, -DELTA) for r in within_change_reps_by_inst])


def h4_rejection_p(rho_reps) -> float:
    return p_less(rho_reps, RHO_BOUND)


__all__ = [n for n in dir() if not n.startswith("_")] + ["M_HOLM"]
