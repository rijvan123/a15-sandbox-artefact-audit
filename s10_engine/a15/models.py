"""L4 learners (S7 analysis table; S8 equations 5-7 and 21-23).

Three deep learners share one content embedding (token + run-length bucket); positions are added inside each
model, so Integrated Gradients integrates over content only and the <M> baseline differs from x only where x is
unmasked (S8 Section 6). The n-gram LightGBM is audited by masking only.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .config import BETA_MAX, BETA_NEUTRAL, L_CAP
from .masks import apply_mask
from .stats import macro_f1

N_BETA = BETA_MAX + 2                                          # buckets 0..7 and the neutral bucket under <M>
PAD_ID, MASK_ID = 0, 1
# Threads: every logical CPU unless A15_THREADS is set (amendment 4, 5 Nov 2026). On machines with many logical CPUs,
# hyper-threads or efficiency cores, small sequence models can run several times slower on all threads than on a
# few; bench_threads.py measures this on the analysis machine. The thread count changes speed, not the analysis.
torch.set_num_threads(max(1, int(os.environ.get("A15_THREADS") or 0) or os.cpu_count() or 1))


@dataclass
class Trace:
    ids: np.ndarray       # token ids (int64)
    betas: np.ndarray     # run-length buckets
    cls: np.ndarray       # lexicon class per element: 0 B, 1 F, 2 P
    y: int                # family index within the shared family set
    length: dict | None = None


class SeqModel(nn.Module):
    def __init__(self, V: int, K: int, d: int = 128, L: int = L_CAP):
        super().__init__()
        self.tok = nn.Embedding(V, d, padding_idx=PAD_ID)
        self.beta = nn.Embedding(N_BETA, d)
        self.pos = nn.Embedding(L, d)
        self.d, self.K = d, K

    def content(self, ids, betas):                             # emb(t_j) + emb_beta(beta_j), equation (5) inputs
        return self.tok(ids) + self.beta(betas)

    def add_pos(self, E):
        return E + self.pos(torch.arange(E.shape[1], device=E.device))[None]

    def forward(self, ids, betas):
        return self.from_content(self.content(ids, betas), ids == PAD_ID)

    def from_content(self, E, pad):
        return self.head(self.pool(E, pad))


class TransformerClassifier(SeqModel):
    def __init__(self, V, K, d=128, heads=4, layers=2, ff=256, dropout=0.1, **_):
        super().__init__(V, K, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, K)

    def pool(self, E, pad):                                    # padding mask Pi of equation (5); mean pooling (6)
        U = self.enc(self.add_pos(E), src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).to(U.dtype)
        return (U * keep).sum(1) / keep.sum(1).clamp(min=1)


class BiLSTMAttention(SeqModel):
    def __init__(self, V, K, d=128, layers=1, hidden=64, dropout=0.1, **_):
        super().__init__(V, K, d)
        self.rnn = nn.LSTM(d, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.att = nn.Linear(2 * hidden, 1)
        self.head = nn.Linear(2 * hidden, K)

    def pool(self, E, pad):                                    # packed sequences: padding never reaches a state
        lengths = (~pad).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.add_pos(E), lengths, batch_first=True, enforce_sorted=False)
        H, _ = self.rnn(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(H, batch_first=True, total_length=E.shape[1])
        a = torch.softmax(self.att(H).squeeze(-1).masked_fill(pad, float("-inf")), dim=1)
        return (a.unsqueeze(-1) * H).sum(1)


class CNN1D(SeqModel):
    def __init__(self, V, K, d=128, channels=128, kernels=(3, 5, 7), dropout=0.1, **_):
        super().__init__(V, K, d)
        self.convs = nn.ModuleList(nn.Conv1d(d, channels, k, padding=k // 2) for k in kernels)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(channels * len(kernels), K)

    def pool(self, E, pad):                                    # max pooling over real positions only
        X = self.add_pos(E).masked_fill(pad.unsqueeze(-1), 0.0).transpose(1, 2)
        outs = [torch.relu(c(X)).masked_fill(pad.unsqueeze(1), float("-inf")).amax(dim=2) for c in self.convs]
        return self.drop(torch.cat(outs, dim=1))


BUILDERS = {"transformer": TransformerClassifier, "bilstm": BiLSTMAttention, "cnn": CNN1D}
SEARCH_SPACE = {
    "transformer": {"layers": [2, 3, 4], "ff": [256, 512], "dropout": [0.1, 0.2], "lr": [3e-4, 1e-3],
                    "wd": [0.01, 0.05], "gamma": [0.0, 0.5, 1.0], "batch": [32]},
    "bilstm": {"layers": [1, 2], "hidden": [64], "dropout": [0.1, 0.2], "lr": [1e-3, 3e-3], "wd": [0.01, 0.05],
               "gamma": [0.0, 0.5, 1.0], "batch": [32]},
    "cnn": {"channels": [64, 128], "dropout": [0.1, 0.3], "lr": [1e-3, 3e-3], "wd": [0.01, 0.05],
            "gamma": [0.0, 0.5, 1.0], "batch": [32]},
    "lgbm": {"num_leaves": [31, 63], "n_estimators": [200, 400], "learning_rate": [0.05, 0.1],
             "min_child_samples": [10, 20], "gamma": [0.0, 0.5, 1.0]},
}


def sample_configs(kind: str, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    space = SEARCH_SPACE[kind]
    return [{k: v[rng.integers(len(v))] for k, v in space.items()} for _ in range(n)]


# ------------------------------------------------------------------------------------------------
# Batching, artefact dropout and training (equation 7)
# ------------------------------------------------------------------------------------------------
def collate(batch: list[Trace], p_drop: float = 0.0, rng=None, test_mask: tuple = ()):
    T = max(len(t.ids) for t in batch)
    ids = np.zeros((len(batch), T), dtype=np.int64)
    betas = np.zeros((len(batch), T), dtype=np.int64)
    for r, t in enumerate(batch):
        i, b = t.ids, t.betas
        if test_mask:                                          # deterministic test-time mask M_S
            i, b, _ = apply_mask(i, b, t.cls, test_mask, MASK_ID, BETA_NEUTRAL)
        if p_drop > 0:                                         # artefact dropout on F, redrawn at every pass
            i, b, _ = apply_mask(i, b, t.cls, ("F",), MASK_ID, BETA_NEUTRAL, p=p_drop, rng=rng)
        ids[r, :len(i)], betas[r, :len(b)] = i, b
    y = torch.tensor([t.y for t in batch], dtype=torch.long)
    return torch.from_numpy(ids), torch.from_numpy(betas), y


def class_weights(train: list[Trace], K: int, gamma: float) -> torch.Tensor:
    n_k = np.bincount([t.y for t in train], minlength=K).astype(float)
    w = (n_k[n_k > 0].mean() / np.maximum(n_k, 1)) ** gamma
    return torch.tensor(w, dtype=torch.float32)


def predict(model, traces: list[Trace], test_mask: tuple = (), batch: int = 64, logits: bool = False):
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(traces), batch):
            ids, betas, _ = collate(traces[b:b + batch], test_mask=test_mask)
            out.append(model(ids, betas))
    z = torch.cat(out).numpy() if out else np.zeros((0, model.K))
    return z if logits else z.argmax(1)


def train_deep(kind: str, V: int, K: int, train: list[Trace], val: list[Trace], cfg: dict, seed: int,
               p_drop: float = 0.0, val_mask: tuple = (), epochs: int = 20, patience: int = 3,
               time_limit_s: float | None = None) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = BUILDERS[kind](V, K, **{k: v for k, v in cfg.items() if k not in ("lr", "wd", "gamma", "batch")})
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    lossf = nn.CrossEntropyLoss(weight=class_weights(train, K, cfg["gamma"]))   # mean over sum of weights, eq. (7)
    best, best_state, bad, hist, t0 = -1.0, None, 0, [], time.perf_counter()
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(train))
        for b in range(0, len(order), cfg["batch"]):
            ids, betas, y = collate([train[i] for i in order[b:b + cfg["batch"]]], p_drop, rng)
            opt.zero_grad()
            loss = lossf(model(ids, betas), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        score = macro_f1([t.y for t in val], predict(model, val, val_mask), K) if val else float("nan")
        hist.append({"epoch": ep + 1, "val_bal_macro_f1": score, "loss": float(loss)})
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        # Amendment 4 (5 Nov 2026): early stopping needs a validation set. Without one every score is NaN, so the
        # earlier test `bad >= patience` stopped training after `patience` epochs whatever `epochs` asked for (the
        # 4 Nov dry run trained 3 of 5 epochs). Without a validation set the model now trains for all `epochs`.
        if (val and bad >= patience) or (time_limit_s and time.perf_counter() - t0 > time_limit_s):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"history": hist, "best_val": best, "seconds": time.perf_counter() - t0, "epochs_run": len(hist)}


# ------------------------------------------------------------------------------------------------
# Integrated Gradients (equations 21-23)
# ------------------------------------------------------------------------------------------------
def integrated_gradients(model: SeqModel, trace: Trace, m: int = 50, test_mask: tuple = (),
                         target: int | None = None) -> dict:
    """Gauss-Legendre IG on the content embedding, logit target, <M> baseline at every real position."""
    model.eval()
    ids, betas, _ = collate([trace], test_mask=test_mask)
    pad = ids == PAD_ID
    with torch.no_grad():
        E = model.content(ids, betas)
        E0 = model.content(torch.full_like(ids, MASK_ID), torch.full_like(betas, BETA_NEUTRAL))
        E0 = torch.where(pad.unsqueeze(-1), E, E0)
        z1 = model.from_content(E, pad)[0]
        yhat = int(z1.argmax()) if target is None else target
        z0 = model.from_content(E0, pad)[0, yhat]
    x, w = np.polynomial.legendre.leggauss(m)
    lam = torch.tensor((x + 1) / 2, dtype=E.dtype).view(m, 1, 1)
    q = torch.tensor(w / 2, dtype=E.dtype)
    El = (E0 + lam * (E - E0)).requires_grad_(True)
    z = model.from_content(El, pad.expand(m, -1))[:, yhat]
    (g,) = torch.autograd.grad((z * q).sum(), El)              # sum_l q_l dz/dE at lambda_l
    ig = ((E - E0)[0] * g.sum(0)).sum(-1).detach().numpy()
    dz = float(z1[yhat] - z0)
    delta = abs(ig.sum() - dz) / abs(dz) if dz != 0 else 0.0
    return {"ig": ig, "target": yhat, "delta": delta, "dz": dz}


def ig_population(model: SeqModel, traces: list, test_mask: tuple = (), steps=(50, 100, 200),
                  delta_max: float = 0.05, pass_share: float = 0.95) -> dict:
    """S8 criterion: delta <= 0.05 for >= 95% of explained traces; otherwise m rises to 100, then 200 (P2)."""
    for m in steps:
        res = [integrated_gradients(model, t, m=m, test_mask=test_mask) for t in traces]
        share = float(np.mean([r["delta"] <= delta_max for r in res])) if res else 1.0
        if share >= pass_share:
            break
    return {"m": m, "share_ok": share, "criterion_met": share >= pass_share, "results": res}


def class_shares(ig: np.ndarray, cls: np.ndarray) -> dict:
    a = np.abs(ig)
    tot = a.sum()
    return {c: float(a[cls == k].sum() / tot) if tot > 0 else 0.0 for k, c in enumerate("BFP")}


def time_per_trace(kind: str, V: int, K: int, L: int, cfg: dict, n: int = 32, reps: int = 3) -> float:
    """Seconds per trace for one forward-backward pass at full length L (K6 projection input)."""
    model = BUILDERS[kind](V, K, **{k: v for k, v in cfg.items() if k not in ("lr", "wd", "gamma", "batch")})
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.randint(4, V, (n, L))
    betas = torch.randint(0, BETA_MAX + 1, (n, L))
    y = torch.randint(0, K, (n,))
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        opt.zero_grad()
        nn.functional.cross_entropy(model(ids, betas), y).backward()
        opt.step()
        best = min(best, time.perf_counter() - t0)
    return best / n


def time_per_trace_configs(kind: str, V: int, K: int, L: int, configs: list[dict], n: int = 32,
                           reps: int = 3) -> dict:
    """K6 input over the configurations the random search will try (amendment 4, 5 Nov 2026).

    The earlier K6 input timed one configuration (the first draw of seed 0). The search trains every configuration,
    so the projection uses the mean over them; each is the best of `reps` passes over `n` traces at full length L."""
    times = [time_per_trace(kind, V, K, L, cfg, n=n, reps=reps) for cfg in configs]
    return {"mean": float(np.mean(times)), "min": float(np.min(times)), "max": float(np.max(times)),
            "per_config": [float(t) for t in times], "n": n, "reps": reps}


# ------------------------------------------------------------------------------------------------
# TF-IDF n-gram LightGBM (equation S1; S7 non-deep baseline)
# ------------------------------------------------------------------------------------------------
class NgramLGBM:
    def __init__(self, K: int, cfg: dict, seed: int):
        self.K, self.cfg, self.seed = K, cfg, seed

    @staticmethod
    def docs(traces: list[Trace], test_mask: tuple = ()):
        out = []
        for t in traces:
            ids = apply_mask(t.ids, t.betas, t.cls, test_mask, MASK_ID, BETA_NEUTRAL)[0] if test_mask else t.ids
            out.append(" ".join(f"t{i}" for i in ids))
        return out

    def fit(self, train: list[Trace], val: list[Trace]):
        import lightgbm as lgb
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 3), lowercase=False, smooth_idf=True,
                                   sublinear_tf=False, norm="l2", min_df=2)
        Xtr = self.vec.fit_transform(self.docs(train))
        ytr = np.array([t.y for t in train])
        n_k = np.bincount(ytr, minlength=self.K).astype(float)
        sw = ((n_k[n_k > 0].mean() / np.maximum(n_k, 1)) ** self.cfg["gamma"])[ytr]
        params = {k: v for k, v in self.cfg.items() if k != "gamma"}
        self.model = lgb.LGBMClassifier(objective="multiclass", num_class=self.K, random_state=self.seed, verbose=-1,
                                        **params)
        fit_kw = {}
        if val:
            fit_kw = {"eval_set": [(self.vec.transform(self.docs(val)), np.array([t.y for t in val]))],
                      "callbacks": [lgb.early_stopping(20, verbose=False)]}
        self.model.fit(Xtr, ytr, sample_weight=sw, **fit_kw)
        return self

    def predict(self, traces: list[Trace], test_mask: tuple = ()):
        proba = self.model.predict_proba(self.vec.transform(self.docs(traces, test_mask)))
        full = np.zeros((len(traces), self.K))
        full[:, self.model.classes_] = proba                     # families absent from training get probability 0
        return full.argmax(1)
