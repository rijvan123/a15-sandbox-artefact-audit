"""Splits, family sets and label harmonisation (S7 split table; S8 freeze manifest; kill criteria K3 and K4)."""
from __future__ import annotations

import difflib
import hashlib
import json
import re

import numpy as np
import pandas as pd

from .config import DRYRUN_PSEUDO_ORIGIN, K3_MIN_FAMILIES, K_MIN_TEST, K_MIN_TRAIN, ORIGINS

EXCLUDED_LABELS = {"", "unknown", "na", "none", "generic", "malware", "trojan"}


def norm_family(name) -> str:
    """Case and punctuation harmonisation ('Cobalt Strike' == 'CobaltStrike'); catch-all labels are excluded."""
    s = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return "" if s in EXCLUDED_LABELS else s


def apply_aliases(names, alias_table: dict) -> list[str]:
    return [alias_table.get(n, n) for n in names]


def alias_candidates(names1, names2, cutoff: float = 0.8) -> list[tuple[str, str, float]]:
    """Name pairs for the authors' alias review (never merged automatically): similar or prefix-related names."""
    out = []
    for a in sorted(set(names1) - set(names2)):
        for b in sorted(set(names2) - set(names1)):
            r = difflib.SequenceMatcher(None, a, b).ratio()
            if r >= cutoff or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a))):
                out.append((a, b, round(r, 3)))
    return out


def window_masks(dates: pd.Series, val_month: str, t0: str, t1: str) -> dict:
    v0 = pd.Timestamp(val_month + "-01")
    v1 = v0 + pd.offsets.MonthBegin(1)
    end = pd.Timestamp(t1) + pd.Timedelta(days=1)
    return {"train": (dates < v0).to_numpy(), "val": ((dates >= v0) & (dates < v1)).to_numpy(),
            "test": ((dates >= pd.Timestamp(t0)) & (dates < end)).to_numpy()}


def origin_masks(dates: pd.Series) -> dict:
    return {oid: window_masks(dates, vm, t0, t1) for oid, vm, t0, t1 in ORIGINS}


def dryrun_masks(dates: pd.Series) -> dict:
    """Pseudo-origin inside every origin's training period: no test window is opened before the freeze."""
    vm, t0, t1 = DRYRUN_PSEUDO_ORIGIN
    return window_masks(dates, vm, t0, t1)


def family_counts(families: np.ndarray, masks: dict) -> pd.DataFrame:
    fam = pd.Series(families)
    df = pd.DataFrame({"n_train_period": fam[masks["train"] | masks["val"]].value_counts(),
                       "n_test": fam[masks["test"]].value_counts()}).fillna(0).astype(int)
    return df.drop(index="", errors="ignore")


def shared_families(c1: pd.DataFrame, c2: pd.DataFrame) -> list[str]:
    ok1 = c1[(c1.n_train_period >= K_MIN_TRAIN) & (c1.n_test >= K_MIN_TEST)].index
    ok2 = c2[(c2.n_train_period >= K_MIN_TRAIN) & (c2.n_test >= K_MIN_TEST)].index
    return sorted(set(ok1) & set(ok2))


def k3_status(shared: list[str]) -> dict:
    return {"n_shared": len(shared), "K3_triggered": len(shared) < K3_MIN_FAMILIES}


def temporal_subsample(keys: pd.DataFrame, frac: float, seed: int) -> np.ndarray:
    """Documented month x family sub-sample (S7 failure table): ceil(frac x n) per non-empty cell."""
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(keys), dtype=bool)
    for _, idx in keys.groupby(["month", "family"]).indices.items():
        k = int(np.ceil(frac * len(idx)))
        keep[rng.choice(idx, size=k, replace=False)] = True
    return keep


def grouped_split(cluster_ids: np.ndarray, families: np.ndarray, seed: int, test_frac=0.2, val_frac=0.1) -> dict:
    """WinMET fallback without dates: whole near-duplicate clusters go to one part, family by family."""
    rng = np.random.default_rng(seed)
    part = np.empty(len(cluster_ids), dtype=object)
    for fam in np.unique(families):
        cl = np.unique(cluster_ids[families == fam])
        rng.shuffle(cl)
        n_te, n_va = int(round(test_frac * len(cl))), int(round(val_frac * len(cl)))
        for k, c in enumerate(cl):
            part[(cluster_ids == c) & (families == fam)] = "test" if k < n_te else ("val" if k < n_te + n_va else "train")
    return {p: part == p for p in ("train", "val", "test")}


def winmet_split_rule(time_key_share: float, per_origin_test_families: dict) -> str:
    """Rolling origins only if >= 95% of traces carry a time key and every 2025 window holds enough families."""
    if time_key_share >= 0.95 and all(n >= K3_MIN_FAMILIES for n in per_origin_test_families.values()):
        return "rolling"
    return "grouped"


def manifest(name: str, ids_by_part: dict) -> dict:
    body = {p: sorted(map(str, ids)) for p, ids in ids_by_part.items()}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return {"name": name, "counts": {p: len(v) for p, v in body.items()}, "sha256": hashlib.sha256(blob).hexdigest(),
            "ids": body}
