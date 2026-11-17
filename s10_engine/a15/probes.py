"""Probes (S7 analysis table): length-only (H1 baseline), artefact-only (H1) and environment-ID (secondary)."""
from __future__ import annotations

import numpy as np
import torch

from .models import PAD_ID, Trace, collate

LENGTH_KEYS = ("n_calls", "n_apis", "n_procs", "duration_s", "early_exit")


def length_matrix(traces: list[Trace]) -> np.ndarray:
    return np.array([[t.length.get(k, 0) for k in LENGTH_KEYS] for t in traces], dtype=float)


def length_probe(train: list[Trace], K: int, seed: int, cfg: dict | None = None):
    import lightgbm as lgb
    cfg = cfg or {"num_leaves": 15, "n_estimators": 200, "learning_rate": 0.05, "min_child_samples": 10}
    m = lgb.LGBMClassifier(objective="multiclass", num_class=K, random_state=seed, verbose=-1, **cfg)
    m.fit(length_matrix(train), np.array([t.y for t in train]))
    return m


def pooled_embeddings(model, traces: list[Trace], batch: int = 64) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(traces), batch):
            ids, betas, _ = collate(traces[b:b + batch])
            out.append(model.pool(model.content(ids, betas), ids == PAD_ID))
    return torch.cat(out).numpy()


def env_id_probe(emb: np.ndarray, machine: np.ndarray, seed: int, n_perm: int = 999) -> dict:
    """Balanced accuracy of a logistic regression on frozen embeddings, against chance 1/M (permutation test)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler          # z-scores fitted inside each training fold
    labels, codes = np.unique(machine, return_inverse=True)
    if len(labels) < 2:
        return {"machines": len(labels), "tested": False}
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    obs = balanced_accuracy_score(codes, cross_val_predict(clf, emb, codes, cv=cv))
    rng, null = np.random.default_rng(seed), []
    for _ in range(n_perm):
        perm = rng.permutation(codes)
        null.append(balanced_accuracy_score(perm, cross_val_predict(clf, emb, perm, cv=cv)))
    p = (1 + sum(v >= obs for v in null)) / (n_perm + 1)
    return {"machines": len(labels), "tested": True, "balanced_accuracy": float(obs), "chance": 1 / len(labels),
            "p_permutation": float(p)}
