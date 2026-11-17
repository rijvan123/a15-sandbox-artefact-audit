"""Installation overlap by token class (S8 equation 10) and prediction P3, computed before any model is trained."""
from __future__ import annotations

from collections import Counter

import numpy as np
import scipy.sparse as sp


def count_matrix(seqs: list[list[str]], index: dict) -> sp.csr_matrix:
    rows, cols = [], []
    for r, s in enumerate(seqs):
        for t in s:
            rows.append(r)
            cols.append(index[t])
    data = np.ones(len(rows), dtype=np.float64)
    return sp.csr_matrix((data, (rows, cols)), shape=(len(seqs), len(index)))


class CountBuilder:
    """Trace-by-token count matrices assembled one trace at a time (the S9 overlap stage streams its records).

    Rows follow the order in which traces are added for each installation; columns follow the sorted vocabulary of
    all installations, as in count_matrix, so class_overlap_counts on these matrices gives class_overlap's result."""

    def __init__(self, installations=("I1", "I2")):
        self.local: dict[str, int] = {}                    # token type -> id in order of first appearance
        self.parts = {inst: ([0], [], []) for inst in installations}

    def add(self, inst: str, seq) -> None:
        cnt = Counter(seq)
        indptr, cols, vals = self.parts.setdefault(inst, ([0], [], []))
        cols.append(np.fromiter((self.local.setdefault(t, len(self.local)) for t in cnt), dtype=np.int64,
                                count=len(cnt)))
        vals.append(np.fromiter(cnt.values(), dtype=np.float64, count=len(cnt)))
        indptr.append(indptr[-1] + len(cnt))

    def matrices(self) -> tuple[dict, list[str]]:
        vocab = sorted(self.local)
        remap = np.empty(len(vocab), dtype=np.int64)
        for k, t in enumerate(vocab):
            remap[self.local[t]] = k
        out = {}
        for inst in list(self.parts):
            indptr, cols, vals = self.parts.pop(inst)          # released as each matrix is built
            c = remap[np.concatenate(cols)] if cols else np.zeros(0, dtype=np.int64)
            v = np.concatenate(vals) if vals else np.zeros(0)
            m = sp.csr_matrix((v, c, np.array(indptr, dtype=np.int64)), shape=(len(indptr) - 1, len(vocab)))
            m.sort_indices()
            out[inst] = m
        return out, vocab


def jw(a: np.ndarray, b: np.ndarray) -> float:
    """Weighted Jaccard of two relative-frequency vectors; equals (1 - TV)/(1 + TV) (check C14)."""
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    qa, qb = a / a.sum(), b / b.sum()
    return float(np.minimum(qa, qb).sum() / np.maximum(qa, qb).sum())


def class_overlap(seqs1, seqs2, classify, n_boot: int = 1999, seed: int = 0, chunk: int = 50) -> dict:
    """J_w(Q_C^(I1), Q_C^(I2)) for C in {B, F, P}, with a trace bootstrap of J_w(B) - J_w(F) (prediction P3)."""
    vocab = sorted({t for s in seqs1 for t in s} | {t for s in seqs2 for t in s})
    index = {t: i for i, t in enumerate(vocab)}
    X1, X2 = count_matrix(seqs1, index), count_matrix(seqs2, index)
    return class_overlap_counts(X1, X2, vocab, classify, n_boot, seed, chunk)


def class_overlap_counts(X1: sp.csr_matrix, X2: sp.csr_matrix, vocab: list[str], classify, n_boot: int = 1999,
                         seed: int = 0, chunk: int = 50) -> dict:
    """class_overlap from trace-by-token count matrices whose columns follow the sorted vocabulary.

    Split out on 3 November 2026 so that the S9 overlap stage can build the counts one trace at a time. Counts are
    whole numbers and every sum below is exact in float64, so the result does not depend on how the matrices were
    assembled. The column slices are taken once per class instead of once per bootstrap chunk (same values)."""
    cls = np.array([classify(t) for t in vocab])
    cols = {C: np.flatnonzero(cls == C) for C in "BFP"}
    S1 = {C: X1[:, c] for C, c in cols.items() if c.size}
    S2 = {C: X2[:, c] for C, c in cols.items() if c.size}
    point = {C: jw(np.asarray(S1[C].sum(0)).ravel(), np.asarray(S2[C].sum(0)).ravel()) if c.size else
             float("nan") for C, c in cols.items()}
    rng = np.random.default_rng(seed)
    n1, n2 = X1.shape[0], X2.shape[0]
    reps = {C: [] for C in "BFP"}
    done = 0
    while done < n_boot:
        r = min(chunk, n_boot - done)
        W1 = rng.multinomial(n1, np.full(n1, 1 / n1), size=r).astype(np.float64)
        W2 = rng.multinomial(n2, np.full(n2, 1 / n2), size=r).astype(np.float64)
        w1, w2 = sp.csr_matrix(W1), sp.csr_matrix(W2)
        for C, c in cols.items():
            if not c.size:
                continue
            A, B = (w1 @ S1[C]).toarray(), (w2 @ S2[C]).toarray()
            reps[C].extend(jw(A[k], B[k]) for k in range(r))
        done += r
    d = np.array(reps["B"]) - np.array(reps["F"]) if reps["F"] else np.array([])
    lo, hi = (np.nanpercentile(d, [2.5, 97.5]) if d.size else (np.nan, np.nan))
    return {"J_w": point, "diff_B_minus_F": point["B"] - point["F"], "ci95": [float(lo), float(hi)],
            "P3_holds": bool(lo > 0), "n_boot": n_boot, "n_traces": [n1, n2],
            "n_token_types": {C: int(c.size) for C, c in cols.items()},
            "implication": "" if lo > 0 else "P3 fails: S8 predicts an upper bound of rho below 0.10 (H4 rejected)"}
