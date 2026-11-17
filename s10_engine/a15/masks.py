"""Masking operators (S8 equation 3) and the matched random mask for the H2 control (equation 4, decision D5)."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .config import MASK_GRID_LADDER, MASK_MAX_BORROW

CLS_ID = {"B": 0, "F": 1, "P": 2}


def apply_mask(ids, betas, cls, S, mask_id: int, beta_neutral: int, p: float = 1.0, rng=None):
    """[M^xi_S(x)]_j = <M> (bucket neutral) if c(t_j) in S and xi_j = 1; xi = 1 for test masks, Bernoulli(p) for
    artefact dropout. The whole element is replaced, so the run length of a masked call cannot leak."""
    ids, betas = np.array(ids, copy=True), np.array(betas, copy=True)
    sel = np.isin(np.asarray(cls), [CLS_ID[s] for s in S])
    if p < 1.0:
        sel &= rng.random(len(ids)) < p
    ids[sel], betas[sel] = mask_id, beta_neutral
    return ids, betas, sel


def band_edges(type_log_freq: np.ndarray, n_bands: int) -> np.ndarray:
    """Quantile edges of the training log-frequencies of token types (equation 4's frequency bands)."""
    return np.quantile(type_log_freq, np.linspace(0, 1, n_bands + 1)[1:-1]) if n_bands > 1 else np.array([])


def token_bands(ids, log_freq_by_id: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, log_freq_by_id[np.asarray(ids)], side="right")


def matched_random_mask(cls, band, S, gen, n_pos=10):
    """Port of the S8-verified function (check C10): same count per (band, position bin) cell, never touching S;
    shortfalls borrow from the same band at the nearest position bin, then from the nearest band."""
    cls = np.asarray(cls)
    L_ = len(cls)
    dec = np.minimum((np.arange(L_) * n_pos) // L_, n_pos - 1)
    in_S = np.isin(cls, [CLS_ID[s] for s in S])
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
    for (b_, d_) in unmatched:
        free = [j for j in range(L_) if not in_S[j] and j not in chosen]
        if not free:
            break
        j = min(free, key=lambda q: (abs(int(band[q]) - int(b_)), abs(int(dec[q]) - int(d_)), gen.random()))
        chosen.add(j)
        borrowed += 1
    return chosen, borrowed, int(in_S.sum())


def choose_grid(train_traces, log_freq_by_id, type_log_freq, S=("F", "P"), seed=0):
    """Decision D5: the finest grid of the ladder whose borrowed share on training traces is at most 10%."""
    shares = {}
    for nb, npos in MASK_GRID_LADDER:
        edges, gen = band_edges(type_log_freq, nb), np.random.default_rng(seed)
        masked = borrowed = 0
        for ids, cls in train_traces:
            if len(ids) == 0:
                continue
            _, b, m = matched_random_mask(cls, token_bands(ids, log_freq_by_id, edges), S, gen, npos)
            masked, borrowed = masked + m, borrowed + b
        shares[(nb, npos)] = borrowed / masked if masked else 0.0
        if shares[(nb, npos)] <= MASK_MAX_BORROW:
            return (nb, npos), shares
    return MASK_GRID_LADDER[-1], shares                       # reported as approximately matched (P8)
