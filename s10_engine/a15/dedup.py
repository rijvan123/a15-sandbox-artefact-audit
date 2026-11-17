"""Near-duplicates and the unit of inference (S8 equations 8-9 and S4; S7 pipeline step 7).

MinHash with LSH banding only proposes candidate pairs; every keep-or-drop decision uses the exact Jaccard index
of API-name 3-gram sets. tau_J is fixed on training traces by the antimode rule (fallback 0.8).
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np

from .config import LSH_CONFIG, TAU_J_FALLBACK, TAU_J_KDE_RANGE

_P = (1 << 31) - 1                                         # Mersenne prime: (a x + b) stays below 2^63


def api_3grams(api_names: list[str]) -> np.ndarray:
    """Set of API-name 3-grams of a harmonised trace, as unique 64-bit integers."""
    if len(api_names) < 3:
        api_names = api_names + ["<END>"] * (3 - len(api_names))
    grams = {"\x1f".join(api_names[i:i + 3]) for i in range(len(api_names) - 2)}
    return np.unique(np.array([int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big") & _P
                               for g in grams], dtype=np.uint64))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / (a.size + b.size - inter) if (a.size + b.size) else 1.0


def jaccard_sorted(a: np.ndarray, b: np.ndarray) -> float:
    """jaccard() for sorted arrays of unique values (api_3grams output), by binary search instead of a merge sort.

    Same value as jaccard(a, b): the intersection size is the same whole number (S9 amendment, 3 November 2026)."""
    if a.size > b.size:
        a, b = b, a
    if a.size == 0:
        return 1.0 if b.size == 0 else 0.0
    idx = np.searchsorted(b, a)
    inter = int(np.count_nonzero(b[np.minimum(idx, b.size - 1)] == a))
    return inter / (a.size + b.size - inter)


class MinHasher:
    def __init__(self, n_h: int, seed: int = 20251101):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, _P, size=n_h, dtype=np.uint64)
        self.b = rng.integers(0, _P, size=n_h, dtype=np.uint64)
        self.n_h = n_h

    def signature(self, s: np.ndarray, chunk: int = 4096) -> np.ndarray:
        sig = np.full(self.n_h, np.iinfo(np.uint64).max, dtype=np.uint64)
        for i in range(0, s.size, chunk):                  # (a x + b) mod p, exact in uint64 because a, x < 2^31
            x = s[i:i + chunk]
            h = (self.a[:, None] * x[None, :] + self.b[:, None]) % np.uint64(_P)
            sig = np.minimum(sig, h.min(axis=1))
        return sig


def lsh_buckets(sigs: np.ndarray, bands: int, rows: int) -> list[dict]:
    tables = []
    for b in range(bands):
        t = defaultdict(list)
        block = sigs[:, b * rows:(b + 1) * rows]
        for idx, row in enumerate(block):
            t[row.tobytes()].append(idx)
        tables.append(t)
    return tables


def config_for(tau: float) -> tuple[int, int]:
    return LSH_CONFIG[min(LSH_CONFIG, key=lambda k: abs(k - tau))]


def candidates_within(sigs: np.ndarray, bands: int, rows: int) -> set[tuple[int, int]]:
    pairs = set()
    for t in lsh_buckets(sigs, bands, rows):
        for members in t.values():
            if 1 < len(members) <= 5_000:                  # giant buckets are exact duplicates of a template
                m = sorted(members)
                pairs.update((m[i], m[j]) for i in range(len(m)) for j in range(i + 1, len(m)))
    return pairs


def nn_similarities(sets: list[np.ndarray], sigs: np.ndarray) -> np.ndarray:
    """Nearest-neighbour exact Jaccard within the training traces (candidates from the 32 x 4 banding)."""
    nn = np.zeros(len(sets))
    for i, j in candidates_within(sigs[:, :128], 32, 4):
        jj = jaccard(sets[i], sets[j])
        nn[i], nn[j] = max(nn[i], jj), max(nn[j], jj)
    return nn


def antimode_tau(nn: np.ndarray) -> tuple[float, dict]:
    """tau_J = antimode of the kernel density of nearest-neighbour similarities on [0.5, 0.95]; else 0.8."""
    from scipy.stats import gaussian_kde
    lo, hi = TAU_J_KDE_RANGE
    x = nn[(nn > 0)]
    info = {"n": int(x.size)}
    if x.size < 50 or np.ptp(x) == 0:
        return TAU_J_FALLBACK, {**info, "rule": "fallback (too few or constant similarities)"}
    grid = np.linspace(0.0, 1.0, 401)
    dens = gaussian_kde(x)(grid)
    peaks = [i for i in range(1, len(grid) - 1) if dens[i] > dens[i - 1] and dens[i] >= dens[i + 1]]
    inside = [i for i in range(1, len(grid) - 1) if lo <= grid[i] <= hi and dens[i] < dens[i - 1]
              and dens[i] <= dens[i + 1]]
    for i in inside:                                       # a minimum between a lower and a higher peak
        if any(p < i for p in peaks) and any(p > i for p in peaks):
            return float(round(grid[i], 3)), {**info, "rule": "antimode", "density_at_tau": float(dens[i])}
    return TAU_J_FALLBACK, {**info, "rule": "fallback (not bimodal on [0.5, 0.95])"}


def dedup_test(train_sets, test_sets, train_sigs, test_sigs, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Eq. (9), first set: keep test traces with no training near-duplicate; returns (keep mask, max J)."""
    bands, rows = config_for(tau)
    sub_tr, sub_te = train_sigs[:, :bands * rows], test_sigs[:, :bands * rows]
    tables = lsh_buckets(sub_tr, bands, rows)
    maxj = np.zeros(len(test_sets))
    for i, s in enumerate(sub_te):
        cand = set()
        for b, t in enumerate(tables):
            cand.update(t.get(s[b * rows:(b + 1) * rows].tobytes(), ()))
        for j in cand:
            maxj[i] = max(maxj[i], jaccard(test_sets[i], train_sets[j]))
    return maxj < tau, maxj


MAX_BUCKET = 5_000                                         # as in candidates_within: larger buckets are skipped


def candidate_neighbours(sigs: np.ndarray, bands: int, rows: int, max_bucket: int = MAX_BUCKET, stats: dict | None = None):
    """Yield (i, J_i) for every trace i, J_i the sorted traces j > i that share an LSH bucket of 2..max_bucket
    members with i in at least one band.

    The pairs (i, j) are exactly the set returned by candidates_within(sigs, bands, rows), produced one trace at a
    time in increasing (i, j) order, so memory is O(n x bands) instead of one Python tuple per candidate pair.
    Buckets are keyed by the full row of `rows` signature values of a band, as in lsh_buckets."""
    n = sigs.shape[0]
    per_band = []
    sizes = []
    for b in range(bands):
        block = np.ascontiguousarray(sigs[:, b * rows:(b + 1) * rows])
        _, inv, cnt = np.unique(block, axis=0, return_inverse=True, return_counts=True)
        inv = np.asarray(inv).reshape(-1)
        order = np.argsort(inv, kind="stable")             # bucket members in increasing index order
        start = np.concatenate([[0], np.cumsum(cnt)])
        per_band.append((inv, order, start, cnt))
        sizes.append(cnt)
    if stats is not None:
        allc = np.concatenate(sizes) if sizes else np.zeros(0, dtype=np.int64)
        stats.update({"traces": int(n), "bands": bands, "rows": rows, "max_bucket": int(allc.max()) if allc.size else 0,
                      "buckets_2_or_more": int((allc > 1).sum()), "buckets_over_cap": int((allc > max_bucket).sum()),
                      "traces_in_buckets_over_cap": int(allc[allc > max_bucket].sum()),
                      "pairs_upper_bound": int((allc[(allc > 1) & (allc <= max_bucket)].astype(np.int64)
                                                * (allc[(allc > 1) & (allc <= max_bucket)] - 1) // 2).sum())})
    for i in range(n):
        js = []
        for inv, order, start, cnt in per_band:
            k = inv[i]
            if 1 < cnt[k] <= max_bucket:
                m = order[start[k]:start[k + 1]]
                m = m[np.searchsorted(m, i, side="right"):]
                if m.size:
                    js.append(m)
        if js:
            yield i, np.unique(np.concatenate(js)) if len(js) > 1 else js[0]


def test_clusters(test_sets, test_sigs, labels, strata, tau: float, stats: dict | None = None) -> tuple[np.ndarray, int]:
    """Eq. (9), relation ~: connected components of same-label near-duplicates within each stratum.

    Returns cluster ids and the number of cross-label near-duplicate pairs (label-quality flag, never merged).

    Amendment of 3 November 2026 (S9, before the freeze). Candidate pairs come from candidate_neighbours, one trace
    at a time, instead of one Python set holding every pair: at WinMET scale that set could exceed the memory of the
    analysis machine. Each candidate pair is visited once, in increasing (i, j) order; a pair already inside one
    cluster is skipped, because merging it changes nothing. The partition and the cross-label count are those of the
    earlier implementation (test_clusters_amendment_equivalence in tests/test_harness.py). Cluster ids are now
    numbered by each cluster's first member, so they no longer depend on the iteration order of a Python set. The sets
    must be sorted arrays of unique values, as api_3grams returns them (jaccard_sorted)."""
    n = len(test_sets)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    bands, rows = config_for(tau)
    cross = visited = evaluated = merged = 0
    for i, js in candidate_neighbours(test_sigs[:, :bands * rows], bands, rows, stats=stats):
        si, li = strata[i], labels[i]
        for j in js.tolist():
            visited += 1
            if strata[j] != si:
                continue
            ri, rj = find(i), find(j)
            if ri == rj:                                   # same cluster already (clusters never mix labels)
                continue
            evaluated += 1
            if jaccard_sorted(test_sets[i], test_sets[j]) < tau:
                continue
            if labels[j] != li:
                cross += 1
                continue
            parent[ri] = rj
            merged += 1
    first: dict[int, int] = {}
    ids = np.array([first.setdefault(find(i), len(first)) for i in range(n)], dtype=np.int64)
    if stats is not None:
        stats.update({"pairs_visited": visited, "jaccard_evaluated": evaluated, "merges": merged,
                      "cross_label_pairs": cross, "clusters": len(first), "tau_J": tau})
    return ids, cross
