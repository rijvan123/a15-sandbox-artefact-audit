#!/usr/bin/env python3
"""
s8_checks.py - verification suite for the S8 mathematical model (A15 path).

Every derived result in S8 (Methods equations 1-28, Supplementary Equations S1-S9) is
checked symbolically (SymPy) or numerically (NumPy/SciPy) with a fixed seed.
Synthetic data only: no malware, no network, CPU only, about one minute.

Usage:  python s8_checks.py | tee s8_checks_log.txt
Label:  every number this script prints is measured-on-sandbox on synthetic data.
"""
import hashlib
import hmac
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations

import numpy as np
import scipy
import sklearn
import sympy as sp
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer

SEED = 20261101
rng = np.random.default_rng(SEED)
RESULTS = []


def report(cid, eqs, desc, ok, detail=""):
    """Print one check result; detail lines are indented under it."""
    RESULTS.append((cid, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {cid} ({eqs}) {desc}")
    for line in str(detail).splitlines():
        if line.strip():
            print(f"         {line}")


print(f"S8 verification suite | Python {platform.python_version()} | NumPy {np.__version__} | "
      f"SciPy {scipy.__version__} | SymPy {sp.__version__} | scikit-learn {sklearn.__version__} | seed {SEED}\n")

# ---------------------------------------------------------------------------------------------
# C00  Sequence construction (eq. 2): repeat collapse with run-length bucket, head-plus-tail cut
# ---------------------------------------------------------------------------------------------
def bucket(n_run):
    return min(int(math.floor(math.log2(n_run))), 7)


def collapse(tokens):
    out, i = [], 0
    while i < len(tokens):
        j = i
        while j < len(tokens) and tokens[j] == tokens[i]:
            j += 1
        out.append((tokens[i], bucket(j - i)))
        i = j
    return out


def head_tail(seq, L):
    return list(seq) if len(seq) <= L else list(seq[: L // 2]) + list(seq[len(seq) - L // 2:])


c00 = collapse(list("aaabbcaaaaaaaaa"))
ok00 = (c00 == [("a", 1), ("b", 1), ("c", 0), ("a", 3)]
        and head_tail(list(range(10)), 4) == [0, 1, 8, 9]
        and bucket(1) == 0 and bucket(2) == 1 and bucket(255) == 7 and bucket(10**6) == 7)
report("C00", "eq. 2", "repeat collapse with run-length bucket min(floor(log2 n), 7); head-plus-tail truncation", ok00,
       f"collapse('aaabbcaaaaaaaaa') = {c00}")

# ---------------------------------------------------------------------------------------------
# C01  Per-family F1 (eq. 11)
# ---------------------------------------------------------------------------------------------
TP, FP, FN = sp.symbols("TP FP FN", positive=True)
prec_s, rec_s = TP / (TP + FP), TP / (TP + FN)
report("C01", "eq. 11", "F1 as harmonic mean of precision and recall equals 2TP/(2TP+FP+FN)",
       sp.simplify(2 * prec_s * rec_s / (prec_s + rec_s) - 2 * TP / (2 * TP + FP + FN)) == 0)

# ---------------------------------------------------------------------------------------------
# C02  Bayes form of family precision; recall is prior-free; family-balanced F1 (eq. 12)
# ---------------------------------------------------------------------------------------------
K3 = 3
pis = sp.symbols("pi1:4", positive=True)
Ns = sp.Symbol("N", positive=True)
Gm = sp.Matrix(K3, K3, lambda a, b: sp.Symbol(f"Gamma_{a + 1}{b + 1}", positive=True))  # Pr(yhat=b | y=a)
TP1 = Ns * pis[0] * Gm[0, 0]
FP1 = sum(Ns * pis[j] * Gm[j, 0] for j in range(1, K3))
Sx = sum(pis[j] * Gm[j, 0] for j in range(1, K3))
prec_bayes = pis[0] * Gm[0, 0] / (pis[0] * Gm[0, 0] + Sx)
ok_a = sp.simplify(TP1 / (TP1 + FP1) - prec_bayes) == 0
ok_b = sp.simplify(TP1 / (Ns * pis[0]) - Gm[0, 0]) == 0
ok_c = sp.simplify(sp.diff(prec_bayes, pis[0]) - Gm[0, 0] * Sx / (pis[0] * Gm[0, 0] + Sx) ** 2) == 0
wbal = Ns / K3
F1w = 2 * wbal * Gm[0, 0] / (2 * wbal * Gm[0, 0] + sum(wbal * Gm[j, 0] for j in range(1, K3)) + wbal * (1 - Gm[0, 0]))
ok_d = sp.simplify(F1w - 2 * Gm[0, 0] / (1 + sum(Gm[j, 0] for j in range(K3)))) == 0
report("C02", "eq. 12", "Bayes form of precision; recall prior-free; family-balanced F1 closed form",
       ok_a and ok_b and ok_c and ok_d,
       f"Prec_k = pi_k Rec_k / (pi_k Rec_k + sum_k'!=k pi_k' Gamma_k'k): {ok_a}\n"
       f"Rec_k independent of priors: {ok_b}; dPrec_k/dpi_k = Rec_k S / (pi_k Rec_k + S)^2 > 0: {ok_c}\n"
       f"balanced F1_k = 2 Gamma_kk / (1 + sum_k' Gamma_k'k): {ok_d}")

# ---------------------------------------------------------------------------------------------
# C03  Label-shift confound, numerical illustration (eq. 12)
# ---------------------------------------------------------------------------------------------
Gam = np.array([[0.80, 0.10, 0.06, 0.04],
                [0.08, 0.78, 0.10, 0.04],
                [0.05, 0.12, 0.75, 0.08],
                [0.03, 0.05, 0.12, 0.80]])


def f1_vec(conf):
    tp = np.diag(conf)
    return 2 * tp / (2 * tp + (conf.sum(0) - tp) + (conf.sum(1) - tp))


def macro_f1_expected(priors, balanced=False, N=10_000):
    counts = N * np.asarray(priors)
    conf = counts[:, None] * Gam
    if balanced:
        conf = conf * (N / len(priors)) / counts[:, None]
    return 100 * f1_vec(conf).mean()


uni, skew = np.full(4, 0.25), np.array([0.55, 0.25, 0.15, 0.05])
raw_u, raw_s = macro_f1_expected(uni), macro_f1_expected(skew)
bal_u, bal_s = macro_f1_expected(uni, True), macro_f1_expected(skew, True)
closed = 100 * np.mean(2 * np.diag(Gam) / (1 + Gam.sum(0)))
report("C03", "eq. 12", "label shift alone moves raw macro-F1; family-balanced macro-F1 does not move",
       abs(raw_u - raw_s) > 1 and abs(bal_u - bal_s) < 1e-9 and abs(bal_u - closed) < 1e-9,
       f"identical confusion rates; raw macro-F1 {raw_u:.2f} (uniform priors) vs {raw_s:.2f} (skewed priors): "
       f"{raw_s - raw_u:+.2f} points from priors alone\n"
       f"family-balanced macro-F1 {bal_u:.2f} vs {bal_s:.2f}; closed form {closed:.2f}")

# ---------------------------------------------------------------------------------------------
# C04  Likelihood ratio: posterior odds = LR x prior odds; FAR prior-free under equal alternatives
# ---------------------------------------------------------------------------------------------
pk, Rk = sp.symbols("pi_k Rec_k", positive=True)
alt = sp.symbols("pa1:4", positive=True)
gal = sp.symbols("g1:4", positive=True)
post = pk * Rk / (pk * Rk + sum(a * g for a, g in zip(alt, gal)))
FAR = sum(a * g for a, g in zip(alt, gal)) / sum(alt)
ok1 = sp.simplify(post / (1 - post) - (Rk / FAR) * (pk / sum(alt))) == 0
ok2 = sp.simplify(FAR.subs({a: (1 - pk) / 3 for a in alt}) - sum(gal) / 3) == 0
report("C04", "eq. 12", "posterior odds = (Rec_k / FAR_k) x prior odds; FAR_k prior-free when alternatives are equally likely",
       ok1 and ok2)

# ---------------------------------------------------------------------------------------------
# C05  Softmax cross-entropy derivatives (eqs. 6-7, S3)
# ---------------------------------------------------------------------------------------------
zs = sp.symbols("z1:4")
den = sum(sp.exp(v) for v in zs)
ps = [sp.exp(v) / den for v in zs]
loss = -sp.log(ps[0])
okg = all(sp.simplify(sp.diff(loss, zs[k]) - (ps[k] - (1 if k == 0 else 0))) == 0 for k in range(3))
okh = all(sp.simplify(sp.diff(loss, zs[k], 2) - ps[k] * (1 - ps[k])) == 0 for k in range(3))
report("C05", "eqs. 6-7, S3", "softmax cross-entropy: gradient p_k - 1[k=y]; diagonal Hessian p_k(1-p_k)", okg and okh,
       "LightGBM multiplies this Hessian by K/(K-1): factor_ = num_class/(num_class-1) in "
       "src/objective/multiclass_objective.hpp (read from source, master branch, 1 Nov 2026)")

# ---------------------------------------------------------------------------------------------
# C06  Second-order boosting: leaf weight, leaf score, split gain (eqs. S2-S3)
# ---------------------------------------------------------------------------------------------
wv, Gs, Hs, lam2 = sp.symbols("w G H lambda2", real=True)
GL, GR, HL, HR = sp.symbols("G_L G_R H_L H_R", real=True)
obj = Gs * wv + sp.Rational(1, 2) * (Hs + lam2) * wv ** 2
wstar = sp.solve(sp.diff(obj, wv), wv)[0]
score = lambda g, h: -g ** 2 / (2 * (h + lam2))  # noqa: E731
gain = score(GL + GR, HL + HR) - score(GL, HL) - score(GR, HR)
ok6 = (sp.simplify(wstar + Gs / (Hs + lam2)) == 0
       and sp.simplify(obj.subs(wv, wstar) + Gs ** 2 / (2 * (Hs + lam2))) == 0
       and sp.simplify(gain - sp.Rational(1, 2) * (GL ** 2 / (HL + lam2) + GR ** 2 / (HR + lam2)
                                                   - (GL + GR) ** 2 / (HL + HR + lam2))) == 0)
report("C06", "eqs. S2-S3", "leaf weight w* = -G/(H+lambda2); leaf score -G^2/(2(H+lambda2)); split gain", ok6)

# ---------------------------------------------------------------------------------------------
# C07  TF-IDF n-gram weights reproduce scikit-learn (eq. S1)
# ---------------------------------------------------------------------------------------------
docs = ["NtCreateFile|user_doc|ok NtWriteFile|-|ok NtClose|-|ok",
        "RegOpenKeyExW|hklm_run|ok RegSetValueExW|hklm_run|ok NtClose|-|ok NtClose|-|ok",
        "NtDelayExecution|sleep_long|ok GetTickCount|-|ok NtDelayExecution|sleep_long|ok",
        "NtCreateFile|user_doc|ok NtCreateFile|user_doc|ok NtWriteFile|-|ok"]
vec = TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=False, ngram_range=(1, 3), smooth_idf=True, norm="l2")
Xs = vec.fit_transform(docs).toarray()


def ngram_counts(doc):
    toks, out = doc.split(" "), Counter()
    for n in (1, 2, 3):
        for i in range(len(toks) - n + 1):
            out[" ".join(toks[i:i + n])] += 1
    return out


cnts = [ngram_counts(d) for d in docs]
dfc = Counter(g for c in cnts for g in c)
Xm = np.zeros_like(Xs)
for i, c in enumerate(cnts):
    for g, tf in c.items():
        Xm[i, vec.vocabulary_[g]] = tf * (math.log((1 + len(docs)) / (1 + dfc[g])) + 1)
    Xm[i] /= np.linalg.norm(Xm[i])
err7 = np.abs(Xs - Xm).max()
report("C07", "eq. S1", "TF-IDF with idf = ln((1+N)/(1+df)) + 1 and L2 norm equals scikit-learn output", err7 < 1e-12,
       f"max absolute difference {err7:.1e} over {Xs.shape[1]} n-gram features")

# ---------------------------------------------------------------------------------------------
# C08  Masked scaled dot-product attention (eq. 5)
# ---------------------------------------------------------------------------------------------
Lq, dm, dh, real = 10, 16, 8, 7
U0 = rng.normal(size=(Lq, dm))
Wq, Wk, Wv = (rng.normal(size=(dm, dh)) for _ in range(3))
Pi = np.zeros((Lq, Lq))
Pi[:, real:] = -np.inf


def attention(U):
    S = (U @ Wq) @ (U @ Wk).T / np.sqrt(dh) + Pi
    S = S - S.max(axis=1, keepdims=True)
    A = np.exp(S)
    A /= A.sum(axis=1, keepdims=True)
    return A, A @ (U @ Wv)


A1, O1 = attention(U0)
U1 = U0.copy()
U1[real:] = rng.normal(size=(Lq - real, dm)) * 10
A2, O2 = attention(U1)
report("C08", "eq. 5", "padding mask: rows sum to 1, zero weight on padded keys, padded content cannot change real positions",
       np.allclose(A1.sum(1), 1) and np.all(A1[:, real:] == 0) and np.allclose(O1[:real], O2[:real]))

# ---------------------------------------------------------------------------------------------
# C09  Artefact dropout (eq. 3)
# ---------------------------------------------------------------------------------------------
p_drop, nF, trials = 0.5, 6, 200_000
xi = rng.random((trials, nF)) < p_drop
p_none = (xi.sum(1) == 0).mean()
se_none = math.sqrt((1 - p_drop) ** nF * (1 - (1 - p_drop) ** nF) / trials)
report("C09", "eq. 3", "artefact dropout: Pr(no F token masked) = (1-p)^n_F and E[number masked] = p n_F",
       abs(p_none - (1 - p_drop) ** nF) < 4 * se_none and abs(xi.sum(1).mean() - p_drop * nF) < 0.02,
       f"simulated {p_none:.5f} vs (1-p)^n_F = {(1 - p_drop) ** nF:.5f}; mean masked {xi.sum(1).mean():.4f} vs {p_drop * nF}")

# ---------------------------------------------------------------------------------------------
# C10  Matched random mask (eq. 4): same count per (frequency band, position decile) cell
# ---------------------------------------------------------------------------------------------
def matched_random_mask(cls, band, S, gen, n_pos=10):
    L_ = len(cls)
    dec = np.minimum((np.arange(L_) * n_pos) // L_, n_pos - 1)
    in_S = np.isin(cls, list(S))
    need = Counter((band[j], dec[j]) for j in range(L_) if in_S[j])
    pool = defaultdict(list)
    for j in range(L_):
        if not in_S[j]:
            pool[(band[j], dec[j])].append(j)
    chosen, unmatched = set(), []
    for cell, cnt in need.items():
        take = min(cnt, len(pool[cell]))
        if take:
            chosen.update(gen.choice(pool[cell], take, replace=False).tolist())
        unmatched += [cell] * (cnt - take)
    borrowed = 0
    for (b_, d_) in unmatched:  # borrow: same band first, then nearest decile, then nearest band
        free = [j for j in range(L_) if not in_S[j] and j not in chosen]
        if not free:
            break
        j = min(free, key=lambda q: (abs(int(band[q]) - int(b_)), abs(int(dec[q]) - int(d_)), gen.random()))
        chosen.add(j)
        borrowed += 1
    return chosen, borrowed, set(np.flatnonzero(in_S).tolist()), dec


viol, exact_cells, borrow_rate = 0, 0, {}
for grid in ((10, 10), (5, 5), (4, 4), (3, 3)):
    trace_gen = np.random.default_rng(SEED + 10)      # identical synthetic traces for every grid
    mask_gen = np.random.default_rng(SEED + 11)
    tot_masked = tot_borrowed = 0
    for _ in range(2000):
        L_ = int(trace_gen.integers(64, 513))
        early = np.arange(L_) < L_ // 5               # footprint concentrated in the first fifth
        cls = np.where(trace_gen.random(L_) < np.where(early, 0.45, 0.10), "F",
                       np.where(trace_gen.random(L_) < 0.08, "P", "B"))
        band = (trace_gen.integers(0, 10, L_) * grid[0]) // 10
        R, borrowed, MS, dec = matched_random_mask(cls, band, {"F", "P"}, mask_gen, n_pos=grid[1])
        viol += (len(R & MS) > 0) + (len(R) != min(len(MS), L_ - len(MS)))
        if borrowed == 0:
            exact_cells += (Counter((band[j], dec[j]) for j in R) != Counter((band[j], dec[j]) for j in MS))
        tot_masked += len(MS)
        tot_borrowed += borrowed
    borrow_rate[grid] = 100 * tot_borrowed / tot_masked
report("C10", "eq. 4", "matched random mask: disjoint from the F, P positions, same size, same cell counts unless borrowed",
       viol == 0 and exact_cells == 0,
       f"2,000 synthetic traces per grid (footprint concentrated in the first fifth); constraint violations {viol}\n"
       "borrowed share of masked positions, by grid (frequency bands x position bins): "
       + "; ".join(f"{g[0]} x {g[1]}: {v:.1f}%" for g, v in borrow_rate.items()))

# ---------------------------------------------------------------------------------------------
# C11  Integrated Gradients: completeness (eq. 22), masked-position identity, class shares (eqs. 21-23)
# ---------------------------------------------------------------------------------------------
e1, e2, b1s, b2s, lam = sp.symbols("e1 e2 b1 b2 lambda", real=True)
fx = e1 ** 2 * e2 + 3 * e2 - e1 * e2 ** 2 + sp.Rational(1, 2) * e1 ** 3
path = {e1: b1s + lam * (e1 - b1s), e2: b2s + lam * (e2 - b2s)}
IG1 = (e1 - b1s) * sp.integrate(sp.diff(fx, e1).subs(path, simultaneous=True), (lam, 0, 1))
IG2 = (e2 - b2s) * sp.integrate(sp.diff(fx, e2).subs(path, simultaneous=True), (lam, 0, 1))
ok_sym = sp.simplify(IG1 + IG2 - (fx - fx.subs({e1: b1s, e2: b2s}, simultaneous=True))) == 0

Lx, dd, hid, Kc, Vsz, MASK = 16, 8, 24, 5, 40, 0
emb = rng.normal(0, 1, (Vsz, dd))
posE = rng.normal(0, 0.5, (Lx, dd))
W1 = rng.normal(0, 0.25, (hid, Lx * dd))
bb1 = rng.normal(0, 0.1, hid)
W2 = rng.normal(0, 0.6, (Kc, hid))
bb2 = rng.normal(0, 0.1, Kc)


def logits(E):
    return W2 @ np.tanh(W1 @ (E + posE).ravel() + bb1) + bb2


def grad_logit(E, k):
    h = np.tanh(W1 @ (E + posE).ravel() + bb1)
    return (W1.T @ (W2[k] * (1 - h ** 2))).reshape(Lx, dd)


def integrated_gradients(tok, k, method, m=50):
    E, Eb = emb[tok], emb[np.full(Lx, MASK)]
    if method == "gausslegendre":
        nodes, wts = np.polynomial.legendre.leggauss(m)
        lams, ws = 0.5 * (1 + nodes), 0.5 * wts
    elif method == "trapezoid":
        lams = np.linspace(0, 1, m + 1)
        ws = np.full(m + 1, 1 / m)
        ws[[0, -1]] = 1 / (2 * m)
    else:
        lams, ws = np.arange(1, m + 1) / m, np.full(m, 1 / m)
    avg = sum(w_ * grad_logit(Eb + l_ * (E - Eb), k) for l_, w_ in zip(lams, ws))
    return ((E - Eb) * avg).sum(axis=1)


gaps = {"gausslegendre": [], "trapezoid": [], "riemann_right": []}
masked_ig_max, phiF_masked_max, share_sum_err = 0.0, 0.0, 0.0
for _ in range(200):
    tok = rng.integers(1, Vsz, Lx)
    cls = rng.choice(np.array(["B", "F", "P"]), Lx, p=[0.6, 0.25, 0.15])
    k = int(np.argmax(logits(emb[tok])))
    diff = logits(emb[tok])[k] - logits(emb[np.full(Lx, MASK)])[k]
    for meth in gaps:
        ig = integrated_gradients(tok, k, meth)
        gaps[meth].append(abs(ig.sum() - diff) / abs(diff))
    tok_m = tok.copy()
    tok_m[cls == "F"] = MASK                                  # deployment input: F masked
    km = int(np.argmax(logits(emb[tok_m])))
    ig_m = integrated_gradients(tok_m, km, "gausslegendre")
    masked_ig_max = max(masked_ig_max, np.abs(ig_m[cls == "F"]).max(initial=0.0))
    share = {C: np.abs(ig_m[cls == C]).sum() / np.abs(ig_m).sum() for C in "BFP"}
    phiF_masked_max = max(phiF_masked_max, share["F"])
    share_sum_err = max(share_sum_err, abs(sum(share.values()) - 1))
gl = np.array(gaps["gausslegendre"])
report("C11", "eqs. 21-23", "IG completeness (symbolic and numeric); IG_j = 0 at masked positions; phi_F = 0 on F-masked input",
       ok_sym and gl.max() < 1e-6 and masked_ig_max == 0.0 and phiF_masked_max == 0.0 and share_sum_err < 1e-12,
       f"symbolic completeness for a cubic f: {ok_sym}\n"
       f"completeness gap delta(x), 200 traces, m = 50: Gauss-Legendre max {gl.max():.1e}; "
       f"trapezoid median {np.median(gaps['trapezoid']):.1e}; right Riemann median {np.median(gaps['riemann_right']):.1e}\n"
       f"largest |IG_j| at a masked position: {masked_ig_max}; largest phi_F on F-masked input: {phiF_masked_max}\n"
       f"consequence: the S7 rule 'refer if phi_F > theta_F', applied to the deployed F-masked input, can never fire")

# ---------------------------------------------------------------------------------------------
# C12  MinHash estimator (eq. 8): unbiased, variance J(1-J)/n_h
# ---------------------------------------------------------------------------------------------
Uu, nh, reps = 400, 128, 1500
A_set, B_set = np.arange(0, 200), np.arange(50, 250)
Jtrue = 150 / 250
est = np.empty(reps)
for r_ in range(reps):
    ranks = rng.random((nh, Uu))
    est[r_] = (A_set[ranks[:, A_set].argmin(1)] == B_set[ranks[:, B_set].argmin(1)]).mean()
var_ratio = est.var(ddof=1) / (Jtrue * (1 - Jtrue) / nh)
report("C12", "eq. 8", "MinHash: E[J_hat] = J and Var[J_hat] = J(1-J)/n_h",
       abs(est.mean() - Jtrue) < 4 * math.sqrt(Jtrue * (1 - Jtrue) / nh / reps) and 0.85 < var_ratio < 1.15,
       f"J = {Jtrue}; mean estimate {est.mean():.4f}; observed/theoretical variance {var_ratio:.3f} "
       f"({reps} repetitions, n_h = {nh})")

# ---------------------------------------------------------------------------------------------
# C13  LSH banding (eq. S4): candidate probability and the configuration chosen for each tau_J
# ---------------------------------------------------------------------------------------------
Jsym, nb_s, nr_s = sp.symbols("J n_band n_row", positive=True)
Pc = 1 - (1 - Jsym ** nr_s) ** nb_s
configs = [(32, 4), (25, 5), (21, 6), (16, 8)]
ok13 = all(abs(float(Pc.subs({nb_s: b, nr_s: r, Jsym: (1 - 2 ** (-1 / b)) ** (1 / r)})) - 0.5) < 1e-12
           for b, r in configs)
pc = lambda J, b, r: 1 - (1 - J ** r) ** b  # noqa: E731
lines = ["n_band x n_row  J50    P(0.3)  P(0.5)  P(0.7)   P(0.8)    P(0.9)"]
for b, r in configs:
    lines.append(f"{b:>4} x {r:<4}     {(1 - 2 ** (-1 / b)) ** (1 / r):.3f}  " +
                 "  ".join(f"{pc(J, b, r):.4f}" if J < 0.75 else f"{pc(J, b, r):.6f}" for J in (0.3, 0.5, 0.7, 0.8, 0.9)))
for tau in (0.7, 0.8, 0.9):
    ok_cfg = [(b, r) for b, r in configs if pc(tau, b, r) >= 0.999]
    best = min(ok_cfg, key=lambda c: pc(0.3, *c))
    lines.append(f"tau_J = {tau}: choose {best[0]} x {best[1]} (n_h = {best[0] * best[1]}): "
                 f"recall at tau {pc(tau, *best):.5f}, candidate rate at J = 0.3 {pc(0.3, *best):.3f}")
report("C13", "eq. S4", "LSH candidate probability 1-(1-J^n_row)^n_band and its 50% point (1-2^(-1/n_band))^(1/n_row)",
       ok13, "\n".join(lines))

# ---------------------------------------------------------------------------------------------
# C14  Weighted Jaccard = (1 - TV)/(1 + TV) (eq. 10)
# ---------------------------------------------------------------------------------------------
maxerr = 0.0
for _ in range(10_000):
    m_ = int(rng.integers(2, 60))
    Pd = rng.dirichlet(np.ones(m_) * rng.uniform(0.2, 3))
    Qd = rng.dirichlet(np.ones(m_) * rng.uniform(0.2, 3))
    TV = 0.5 * np.abs(Pd - Qd).sum()
    maxerr = max(maxerr, abs(np.minimum(Pd, Qd).sum() / np.maximum(Pd, Qd).sum() - (1 - TV) / (1 + TV)))
Tq = sp.Symbol("T", nonnegative=True)
ok14 = maxerr < 1e-12 and sp.simplify(sp.diff((1 - Tq) / (1 + Tq), Tq) + 2 / (1 + Tq) ** 2) == 0
report("C14", "eq. 10", "weighted Jaccard of two distributions equals (1-TV)/(1+TV), decreasing in TV", ok14,
       f"largest deviation over 10,000 random distribution pairs {maxerr:.1e}")

# ---------------------------------------------------------------------------------------------
# C15  Delta-method variance of the recovered share (eq. S5)
# ---------------------------------------------------------------------------------------------
Nn, Gg = sp.symbols("N_num G", positive=True)
okd = sp.simplify(sp.diff(Nn / Gg, Nn) - 1 / Gg) == 0 and sp.simplify(sp.diff(Nn / Gg, Gg) + Nn / Gg ** 2) == 0
muN, muG, sN, sG, cor = 0.06, 0.15, 0.010, 0.012, 0.5
draws = rng.multivariate_normal([muN, muG], [[sN ** 2, cor * sN * sG], [cor * sN * sG, sG ** 2]], 400_000)
var_sim = np.var(draws[:, 0] / draws[:, 1])
r0 = muN / muG
var_delta = (sN ** 2 - 2 * r0 * cor * sN * sG + r0 ** 2 * sG ** 2) / muG ** 2
report("C15", "eq. S5", "Var(rho) ~ [Var(N) - 2 rho Cov(N,G) + rho^2 Var(G)] / G^2",
       okd and abs(var_sim / var_delta - 1) < 0.10,
       f"gap 15 points, numerator 6 points (rho = {r0:.2f}): simulated SD {math.sqrt(var_sim):.4f}, "
       f"delta-method SD {math.sqrt(var_delta):.4f}")

# ---------------------------------------------------------------------------------------------
# C16  Planning formula for the K5 check (eq. S6), calibrated by simulation
# ---------------------------------------------------------------------------------------------
zq, Rr, Dl, deff, Nreq, cF = sp.symbols("z R Delta deff N c_F", positive=True)
Nsol = sp.solve(sp.Eq(zq * sp.sqrt(2 * deff * cF ** 2 * Rr * (1 - Rr) / Nreq), Dl), Nreq)[0]
ok16a = sp.simplify(Nsol - 2 * deff * cF ** 2 * zq ** 2 * Rr * (1 - Rr) / Dl ** 2) == 0
z975 = float(stats.norm.ppf(0.975))
table16 = [f"N_req per test set (Delta = 2 points, c_F = 1): " +
           "; ".join(f"Rec {R_}: {math.ceil(2 * de * z975 ** 2 * R_ * (1 - R_) / 0.02 ** 2):,} (deff {de})"
                     for R_ in (0.7, 0.8, 0.9) for de in (1.0, 1.5))]


def simulate_macro(Kf, nk, Rec, scenario, sims, gen):
    mf1, mrec = np.empty(sims), np.empty(sims)
    sib = np.roll(np.arange(Kf), 1)
    for s_ in range(sims):
        conf = np.zeros((Kf, Kf))
        tp = gen.binomial(nk, Rec, Kf)
        for k in range(Kf):
            conf[k, k] = tp[k]
            others = np.array([j for j in range(Kf) if j != k])
            if scenario == "uniform":
                probs = np.full(Kf - 1, 1 / (Kf - 1))
            else:                                   # 80% of errors go to one sibling family
                probs = np.full(Kf - 1, 0.2 / (Kf - 2))
                probs[others == sib[k]] = 0.8
            conf[k, others] = gen.multinomial(nk - tp[k], probs)
        mf1[s_] = f1_vec(conf).mean()
        mrec[s_] = (np.diag(conf) / conf.sum(1)).mean()
    return mf1.std(ddof=1), mrec.std(ddof=1)


Kf, nk, Rec = 20, 150, 0.8
sd_formula = math.sqrt(Kf * Rec * (1 - Rec) / nk) / Kf
ratios = {}
for scen in ("uniform", "sibling"):
    sd_f1, sd_rec = simulate_macro(Kf, nk, Rec, scen, 3000, rng)
    ratios[scen] = sd_f1 / sd_formula
    table16.append(f"{scen:8s} errors, K = {Kf}, n_k = {nk}, Rec = {Rec}: SD(macro-F1) {100 * sd_f1:.3f}, "
                   f"SD(macro-recall) {100 * sd_rec:.3f}, formula {100 * sd_formula:.3f} points; c_F = {ratios[scen]:.2f}")
hw = z975 * math.sqrt(2) * max(ratios.values()) * sd_formula * 100
table16.append(f"implied 95% half-width of a within-minus-cross gap at N = {Kf * nk:,} traces per test set: {hw:.2f} points")
report("C16", "eq. S6", "N_req = 2 deff c_F^2 z^2 Rec(1-Rec) / Delta^2; c_F from simulation",
       ok16a and all(0.5 < v < 1.5 for v in ratios.values()), "\n".join(table16))

# ---------------------------------------------------------------------------------------------
# C17  Unit of inference (eq. 16): trace bootstrap understates SE when near-duplicate clusters exist
# ---------------------------------------------------------------------------------------------
def make_testset(nc, gen):
    sizes = gen.geometric(0.5, nc)                        # mean cluster size 2
    pc_ = gen.beta(1.867, 0.467, nc)                      # mean accuracy 0.8, ICC 0.30
    return sizes, gen.binomial(sizes, pc_)


nc = 400
true_se = np.std([c.sum() / s.sum() for s, c in (make_testset(nc, rng) for _ in range(3000))], ddof=1)
tr_se, cl_se = [], []
for _ in range(150):
    s, c = make_testset(nc, rng)
    acc = c.sum() / s.sum()
    tr_se.append(math.sqrt(acc * (1 - acc) / s.sum()))   # limit of the trace-level bootstrap
    idx = rng.integers(0, nc, (500, nc))
    cl_se.append(np.std(c[idx].sum(1) / s[idx].sum(1), ddof=1))
deff17 = (np.mean(cl_se) / np.mean(tr_se)) ** 2
report("C17", "eq. 16", "cluster bootstrap recovers the true SE; trace bootstrap understates it",
       abs(np.mean(cl_se) / true_se - 1) < 0.10 and np.mean(tr_se) < 0.9 * true_se,
       f"true SE {100 * true_se:.2f}; cluster bootstrap {100 * np.mean(cl_se):.2f}; trace bootstrap "
       f"{100 * np.mean(tr_se):.2f} points; design effect {deff17:.2f} (mean cluster size 2, ICC 0.30)")

# ---------------------------------------------------------------------------------------------
# C18  Holm = closed testing with Bonferroni local tests; intersection-union size (eqs. 17-19)
# ---------------------------------------------------------------------------------------------
def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for i, idx in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - i) * p[idx]))
        adj[idx] = run
    return adj


def closed_bonferroni(p, a):
    m = len(p)
    return np.array([all(min(p[j] for j in sub) <= a / len(sub)
                         for r in range(1, m + 1) for sub in combinations(range(m), r) if i in sub)
                     for i in range(m)])


ex = holm([0.01, 0.04, 0.03, 0.005])
agree = all(np.array_equal(holm(p) <= 0.025, closed_bonferroni(p, 0.025)) for p in rng.uniform(0, 0.05, (5000, 4)))
zb = rng.normal(size=(400_000, 2))
iut_boundary = np.mean((zb > z975).all(1))
iut_sup = np.mean((zb[:, 0] > z975) & (zb[:, 1] + 10 > z975))
nboot = 9999
report("C18", "eqs. 17-19", "Holm adjustment; equivalence with closed testing; IUT size <= alpha",
       np.allclose(ex, [0.03, 0.06, 0.06, 0.02]) and agree and iut_sup <= 0.025 + 3 * math.sqrt(0.025 * 0.975 / 400_000),
       f"Holm([0.01, 0.04, 0.03, 0.005]) = {ex.tolist()}; agreement with closed testing on 5,000 random sets: {agree}\n"
       f"IUT rejection rate at the boundary {iut_boundary:.5f}; least favourable {iut_sup:.5f} (alpha = 0.025)\n"
       f"smallest bootstrap P with N_boot = {nboot:,}: {1 / (nboot + 1):.4f} < smallest Holm threshold "
       f"{0.025 / 4:.5f}")

# ---------------------------------------------------------------------------------------------
# C19  Bootstrap P value <-> order-statistic bound; TOST <-> 95% interval inside +/-Delta (eq. 20)
# ---------------------------------------------------------------------------------------------
kk = int(round(0.025 * (nboot + 1)))
mismatch = 0
for _ in range(2000):
    psi = rng.normal(rng.normal(0, 0.015), rng.uniform(0.003, 0.012), nboot)
    srt = np.sort(psi)
    lo, hi = srt[kk - 1], srt[nboot - kk]
    for c in (-0.02, 0.02):
        mismatch += ((1 + (psi <= c).sum()) / (nboot + 1) <= 0.025) != (lo > c)
        mismatch += ((1 + (psi >= c).sum()) / (nboot + 1) <= 0.025) != (hi < c)
report("C19", "eq. 20", "p_>(c) <= 0.025 iff the 250th smallest replicate > c (N_boot = 9,999); TOST iff 95% interval inside +/-Delta",
       mismatch == 0, f"disagreements in 8,000 comparisons: {mismatch}")

# ---------------------------------------------------------------------------------------------
# C20  Exact McNemar test (eq. S7)
# ---------------------------------------------------------------------------------------------
def mcnemar_exact(n10, n01):
    n = n10 + n01
    return 1.0 if n == 0 else min(1.0, 2 * stats.binom.cdf(min(n10, n01), n, 0.5))


pairs = [(0, 5), (3, 9), (10, 10), (25, 40), (120, 160), (7, 0)]
report("C20", "eq. S7", "exact McNemar P = min(1, 2 Binom(min(n10, n01); n10 + n01, 1/2)) equals SciPy binomtest",
       all(abs(mcnemar_exact(a, b) - stats.binomtest(a, a + b, 0.5).pvalue) < 1e-12 for a, b in pairs))

# ---------------------------------------------------------------------------------------------
# C21  Hodges-Lehmann estimator and Cliff's delta (eq. S8)
# ---------------------------------------------------------------------------------------------
def hodges_lehmann(d):
    d = np.asarray(d, float)
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2))


xa = rng.normal(0.5, 1, 40).round(1)
ya = rng.normal(0.0, 1, 35).round(1)
cliff = ((xa[:, None] > ya[None, :]).sum() - (xa[:, None] < ya[None, :]).sum()) / (len(xa) * len(ya))
Ustat = stats.mannwhitneyu(xa, ya, alternative="two-sided").statistic
report("C21", "eq. S8", "Hodges-Lehmann = median of Walsh averages; Cliff's delta = 2U/(n_x n_y) - 1 (ties included)",
       abs(hodges_lehmann([1, 2, 4]) - 2.25) < 1e-12 and abs(cliff - (2 * Ustat / (len(xa) * len(ya)) - 1)) < 1e-12,
       f"HL([1, 2, 4]) = {hodges_lehmann([1, 2, 4])}; Cliff's delta {cliff:.4f} on tied data")

# ---------------------------------------------------------------------------------------------
# C22  Area Under Time (eq. S9) and the smallest exact Kendall P for six monthly slots
# ---------------------------------------------------------------------------------------------
aut = lambda P: float(np.mean((np.asarray(P)[:-1] + np.asarray(P)[1:]) / 2))  # noqa: E731
Pm = rng.uniform(0, 1, 6)
kt = stats.kendalltau(np.arange(6), -np.arange(6), alternative="less", method="exact")
report("C22", "eq. S9", "AUT = normalised trapezoid area; constant series returns the constant; Kendall min P for n = 6",
       abs(aut([0.8, 0.75, 0.7]) - 0.75) < 1e-12 and abs(aut([0.6] * 6) - 0.6) < 1e-12
       and abs(aut(Pm) - np.trapezoid(Pm) / 5) < 1e-12 and abs(kt.pvalue - 1 / 720) < 1e-12,
       f"exact one-sided Kendall P for a perfectly decreasing 6-month series: {kt.pvalue:.5f} (= 1/720)")

# ---------------------------------------------------------------------------------------------
# C23  Ledger: hash chain (eq. 25), RFC 6962 Merkle roots and inclusion proofs (eq. 26),
#      tamper evidence (eq. 27). The TSA is simulated with an HMAC key (development only).
# ---------------------------------------------------------------------------------------------
def Hh(b):
    return hashlib.sha256(b).digest()


def enc(entry):  # canonical, injective: sorted keys, length-prefixed UTF-8 fields
    out = bytearray()
    for key in sorted(entry):
        kb, vb = key.encode(), str(entry[key]).encode()
        out += len(kb).to_bytes(4, "big") + kb + len(vb).to_bytes(8, "big") + vb
    return bytes(out)


def leaf_h(d):
    return Hh(b"\x00" + d)


def node_h(left, right):
    return Hh(b"\x01" + left + right)


def mth(ds):
    if len(ds) == 1:
        return leaf_h(ds[0])
    nL = 1 << ((len(ds) - 1).bit_length() - 1)          # largest power of two below n
    return node_h(mth(ds[:nL]), mth(ds[nL:]))


def audit_path(m, ds):
    if len(ds) == 1:
        return []
    nL = 1 << ((len(ds) - 1).bit_length() - 1)
    return audit_path(m, ds[:nL]) + [mth(ds[nL:])] if m < nL else audit_path(m - nL, ds[nL:]) + [mth(ds[:nL])]


def verify_inclusion(m, n, d, proof, root):          # RFC 9162, section 2.1.3.2
    if m >= n:
        return False
    fn, sn, r = m, n - 1, leaf_h(d)
    for p in proof:
        if sn == 0:
            return False
        if (fn & 1) or fn == sn:
            r = node_h(p, r)
            while not (fn & 1) and fn != 0:
                fn >>= 1
                sn >>= 1
        else:
            r = node_h(r, p)
        fn >>= 1
        sn >>= 1
    return sn == 0 and r == root


incl_ok, len_ok = True, True
for n in range(1, 65):
    ds = [Hh(f"h{i}".encode()) for i in range(n)]
    root = mth(ds)
    for m in range(n):
        pr = audit_path(m, ds)
        incl_ok &= verify_inclusion(m, n, ds[m], pr, root)
        len_ok &= len(pr) <= math.ceil(math.log2(n)) if n > 1 else len(pr) == 0
        incl_ok &= not verify_inclusion(m, n, Hh(b"other"), pr, root)


def mth_noprefix(ds):
    if len(ds) == 1:
        return Hh(ds[0])
    nL = 1 << ((len(ds) - 1).bit_length() - 1)
    return Hh(mth_noprefix(ds[:nL]) + mth_noprefix(ds[nL:]))


d4 = [Hh(bytes([i])) for i in range(4)]
forged = [Hh(d4[0]) + Hh(d4[1]), Hh(d4[2]) + Hh(d4[3])]
attack_plain, attack_rfc = mth_noprefix(forged) == mth_noprefix(d4), mth(forged) == mth(d4)

TSA_KEY = b"development-only simulated TSA key"


def tsa_sign(root, gen_time):
    imprint = Hh(root)
    return {"imprint": imprint.hex(), "genTime": gen_time,
            "sig": hmac.new(TSA_KEY, imprint + gen_time.to_bytes(8, "big"), hashlib.sha256).hexdigest()}


def tsa_ok(tok, root):
    exp = hmac.new(TSA_KEY, bytes.fromhex(tok["imprint"]) + tok["genTime"].to_bytes(8, "big"), hashlib.sha256).hexdigest()
    return tok["imprint"] == Hh(root).hex() and hmac.compare_digest(tok["sig"], exp)


def pseudonym(actor):
    return hmac.new(b"custodian-held key", actor.encode(), hashlib.sha256).hexdigest()[:16]


def build_ledger(events, batch_size, t0=1_790_000_000):
    hashes = [Hh(enc({"study": "A15 S8 demonstration ledger"}))]
    entries = []
    clock, start = t0, 1

    def append(entry):
        entry = dict(entry, i=len(entries) + 1, prev=hashes[-1].hex())
        entries.append(entry)
        hashes.append(Hh(enc(entry) + hashes[-1]))

    for ev in events:
        clock += 1
        append(dict(ev, actor=pseudonym(ev.get("actor", "analyst")), tau=clock))
        if len(entries) - start + 1 >= batch_size:
            root = mth(hashes[start:len(entries) + 1])
            clock += 1
            append({"action": "anchor", "batch": (start, len(entries)), "root": root.hex(),
                    "tst": tsa_sign(root, clock), "tau": clock, "actor": pseudonym("custodian")})
            start = len(entries)                       # the anchor entry opens the next batch
    return entries, hashes


def verify(entries, hashes):
    h = [hashes[0]]
    first_bad = None
    for i, e in enumerate(entries, start=1):
        h.append(Hh(enc(e) + h[-1]))
        if first_bad is None and h[i] != hashes[i]:
            first_bad = i
    bad_batches = [e["batch"] for e in entries if e["action"] == "anchor"
                   and (mth(h[e["batch"][0]:e["batch"][1] + 1]).hex() != e["root"]
                        or not tsa_ok(e["tst"], mth(h[e["batch"][0]:e["batch"][1] + 1])))]
    return first_bad is None and not bad_batches, first_bad, bad_batches


def rewrite_tail(entries, hashes, k, new_payload):
    """Attacker edits entry k and recomputes every later hash and root; cannot forge TSA tokens."""
    E = [dict(e) for e in entries]
    h = list(hashes)
    E[k - 1]["payload"] = new_payload
    for i in range(k, len(E) + 1):
        E[i - 1]["prev"] = h[i - 1].hex()
        if E[i - 1]["action"] == "anchor":
            s_, t_ = E[i - 1]["batch"]
            E[i - 1]["root"] = mth(h[s_:t_ + 1]).hex()
        h[i] = Hh(enc(E[i - 1]) + h[i - 1])
    return E, h


events = [{"action": "acquire", "payload": f"trace-{i:05d}"} for i in range(2000)]
entries, hashes = build_ledger(events, batch_size=128)
clean_ok = verify(entries, hashes)[0]
last_anchor = max(e["i"] for e in entries if e["action"] == "anchor")
anchored = [e["i"] for e in entries if e["action"] == "acquire" and e["i"] < last_anchor
            and any(a["action"] == "anchor" and a["batch"][0] <= e["i"] <= a["batch"][1] for a in entries)]
det_plain = det_first = det_rewrite = det_batch = 0
trials23 = 500
for k in rng.choice(anchored, trials23, replace=False):
    k = int(k)
    E = [dict(e) for e in entries]
    E[k - 1]["payload"] += "x"                          # edit without recomputing hashes
    ok_, first_bad, _ = verify(E, hashes)
    det_plain += not ok_
    det_first += first_bad == k
    E2, h2 = rewrite_tail(entries, hashes, k, entries[k - 1]["payload"] + "x")
    ok2_, first2, bad2 = verify(E2, h2)
    det_rewrite += not ok2_
    det_batch += any(s_ <= k <= t_ for s_, t_ in bad2) and first2 is None
E3, h3 = rewrite_tail(entries, hashes, len(entries), "edited in the open batch")
open_tail_detected = not verify(E3, h3)[0]

t0 = time.perf_counter()
big, big_h = build_ledger([{"action": "acquire", "payload": f"t{i}"} for i in range(50_000)], batch_size=1024)
t_build = (time.perf_counter() - t0) / len(big) * 1e6
t0 = time.perf_counter()
big_ok = verify(big, big_h)[0]
t_verify = (time.perf_counter() - t0) / len(big) * 1e6
nq = 2 ** 80
log2_bound = math.log2(Fraction(nq * (nq - 1), 2 ** 257))
report("C23", "eqs. 25-27", "hash chain, RFC 6962 roots and inclusion proofs, tamper detection, open-batch exposure",
       clean_ok and incl_ok and len_ok and attack_plain and not attack_rfc and det_plain == trials23
       and det_first == trials23 and det_rewrite == trials23 and det_batch == trials23
       and not open_tail_detected and big_ok,
       f"clean ledger of {len(entries):,} entries ({sum(e['action'] == 'anchor' for e in entries)} anchors) verifies: {clean_ok}\n"
       f"inclusion proofs for every leaf of trees with 1-64 leaves verify, length <= ceil(log2 n): {incl_ok and len_ok}\n"
       f"second-preimage forgery of a Merkle root: succeeds without RFC 6962 prefixes {attack_plain}; with prefixes {attack_rfc}\n"
       f"{trials23} edits of anchored entries without rehashing: detected {det_plain}, first failing link = edited entry {det_first}\n"
       f"{trials23} edits with every later hash and root recomputed: detected {det_rewrite}, by the token of the edited batch {det_batch}\n"
       f"edit in the open (unanchored) batch with recomputation detected before the next anchor: {open_tail_detected} (expected False)\n"
       f"cost: {t_build:.1f} us per entry to seal and {t_verify:.1f} us to verify (50,000 entries, batch 1,024, pure Python)\n"
       f"generic collision bound for 2^80 hash evaluations: 2^{log2_bound:.1f}")

# ---------------------------------------------------------------------------------------------
# C24  Pre-registration ordering predicate (eq. 28)
# ---------------------------------------------------------------------------------------------
def freeze_valid(entries, hashes):
    ok_chain = verify(entries, hashes)[0]
    i_frz = next(e["i"] for e in entries if e["action"] == "freeze")
    i_1 = min((e["i"] for e in entries if e["action"] == "score"), default=None)
    anc = next((e for e in entries if e["action"] == "anchor" and e["batch"][0] <= i_frz <= e["batch"][1]), None)
    return bool(ok_chain and anc is not None and i_1 is not None and anc["i"] < i_1)


pre = [{"action": "acquire", "payload": f"trace-{i}"} for i in range(300)]
frz = [{"action": "freeze", "payload": "SHA-256 of lexicon, H1-H4, margins, manifests"}]
post_acq = [{"action": "acquire", "payload": f"trace-{i}"} for i in range(300, 420)]
scores = [{"action": "score", "payload": f"test result {i}"} for i in range(50)]
good_e, good_h = build_ledger(pre + frz + post_acq + scores, 128)
bad_e, bad_h = build_ledger(pre + frz + scores[:1] + post_acq + scores[1:], 128)   # scoring before the anchor
i_anc = next(e for e in good_e if e["action"] == "anchor" and e["batch"][0] <= 301 <= e["batch"][1])["i"]
Ealt = [dict(e) for e in good_e]
Ealt[i_anc - 1]["tst"] = dict(Ealt[i_anc - 1]["tst"], genTime=Ealt[i_anc - 1]["tst"]["genTime"] + 1)
halt = [good_h[0]]
for e in Ealt:
    halt.append(Hh(enc(e) + halt[-1]))
dependent = all(halt[i] != good_h[i] for i in range(i_anc, len(good_e) + 1))
report("C24", "eq. 28", "freeze precedes scoring iff the freeze batch's anchor entry precedes the first scoring entry",
       freeze_valid(good_e, good_h) and not freeze_valid(bad_e, bad_h) and dependent,
       f"study ledger V_frz = {int(freeze_valid(good_e, good_h))}; negative control (score before anchor) "
       f"V_frz = {int(freeze_valid(bad_e, bad_h))}\n"
       f"every hash after the anchor entry changes if its token changes: {dependent}")

# ---------------------------------------------------------------------------------------------
n_pass = sum(ok for _, ok in RESULTS)
print(f"\nSUMMARY: {n_pass} of {len(RESULTS)} checks passed")
sys.exit(0 if n_pass == len(RESULTS) else 1)
