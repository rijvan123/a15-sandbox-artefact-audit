"""Algorithm 2: freeze the lexicon and analysis plan, anchor it, and draft the OSF registration (S7, S8)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import config as C

_D = f"{C.DELTA:g}"                                        # margin in points (3 after amendment 5; 2 at S6-S8)

HYPOTHESES = {
    "H1": {"claim": "artefact tokens alone identify families", "estimand": "psi1(I) = F1(f_art; M_B(D_te)) - "
           "F1(f_len; D_te)", "components": "each installation", "margin": C.DELTA,
           "support": f"IUT: every component's lower bound above {_D} points", "reject": f"one component below {_D}"},
    "H2": {"claim": "test-time masking of F and P lowers macro-F1 more than matched random masks",
           "estimand": "psi2(I) = F1(f_base; M_rand(D_te)) - F1(f_base; M_FP(D_te))", "components": "each installation",
           "margin": C.DELTA, "support": f"IUT above {_D} points", "reject": f"TOST equivalence within +/-{_D} in both"},
    "H3": {"claim": "cross-installation macro-F1 on shared families is lower than within-installation",
           "estimand": "psi3(I->I') = G_{I->I'}, equation 13", "components": "both directions", "margin": C.DELTA,
           "support": f"IUT above {_D} points in both directions", "reject": f"upper bound below {_D} in either"},
    "H4": {"claim": f"F-masked training recovers part of the gap without losing more than {_D} points within",
           "estimand": "psi4a = rho (equation 14, pooled); psi4b(I) = F1(f_mit; M_F(D_te)) - F1(f_base; D_te)",
           "components": "rho; each installation for psi4b", "margin": {"rho": C.RHO_BOUND, "psi4b": -C.DELTA},
           "support": f"rho > 0.10 and psi4b > -{_D} everywhere", "reject": "rho < 0.10",
           "gate": "tested only if H3 is supported"},
}

SETTINGS = {
    "primary_metric": "family-balanced macro-F1 (decision D2); raw macro-F1 beside it",
    "confirmatory_learner": "Transformer encoder; BiLSTM, 1D-CNN and n-gram LightGBM secondary",
    "error_control": {"alpha_one_sided": C.ALPHA_ONE_SIDED, "holm_m": C.M_HOLM, "families": ["support", "rejection"]},
    "bootstrap": {"unit": "near-duplicate cluster within (test set, origin, family)", "n_boot": C.N_BOOT,
                  "bounds": "percentile"},
    "seeds": list(C.SEEDS), "origins": [list(o) for o in C.ORIGINS], "family_thresholds":
        {"train_period": C.K_MIN_TRAIN, "test": C.K_MIN_TEST}, "recovered_share": "pooled over both directions",
    "sequences": {"truncation": "head-plus-tail", "L_rule": f"smallest power of two >= P95, capped at {C.L_CAP} (512 before amendment 3)",
                  "beta": "min(floor(log2 n_run), 7)", "vocab_min_traces": C.VOCAB_MIN_TRACES,
                  "oos_class": "F (B in sensitivity)"},
    "dropout_grid": list(C.DROPOUT_GRID), "gamma_grid": list(C.GAMMA_GRID), "search_configs": C.N_SEARCH,
    "dedup": {"tau_rule": "antimode on [0.5, 0.95], fallback 0.8", "lsh": {str(k): v for k, v in C.LSH_CONFIG.items()}},
    "matched_mask_ladder": [list(g) for g in C.MASK_GRID_LADDER], "max_borrow": C.MASK_MAX_BORROW,
    "integrated_gradients": {"target": "logit", "baseline": "<M>", "quadrature": "Gauss-Legendre",
                             "m": list(C.IG_STEPS), "criterion": [C.IG_DELTA_MAX, C.IG_PASS_SHARE]},
    "referral": {"statistic": "phi_F of the base model on the unmasked trace (D1)", "eta_F": C.ETA_F,
                 "removed_if": "P6 fails"},
    "gate_fields": list(C.ENV_FIELDS),
    "k5_action": "widen windows or narrow families, else raise the margin to 3 points before scoring",
    "k5_applied": {"amendment": "5 (5 November 2026, before the freeze)", "margin_points": C.DELTA,
                   "registered_half_width_limit_points": C.K5_MAX_HALF_WIDTH,
                   "why": "the corrected dry run projected a 95% half-width of 5.8 points at the smallest origin; the "
                          "test windows cannot widen (the three windows cover March-August 2025 and MELD-DS-448 "
                          "ends in August 2025); the authors kept every family",
                   "consequence": "the projected half-width still exceeds 3 points, so equivalence and "
                                  "non-inferiority verdicts may be inconclusive; this is stated before scoring"},
    "compute_plan": {"amendment": "3 (4 November 2026) and 5 (5 November 2026), before the freeze",
                     "hardware": "authors' PC, CPU only; PyTorch thread count set by A15_THREADS and recorded per run",
                     "deep_learners": "Transformer (confirmatory) and 1D-CNN: full factorial, search_configs and seeds "
                                      "as above; BiLSTM: unmasked and F-masked only; n-gram LightGBM: full factorial",
                     "sequence_cap": C.L_CAP,
                     "reason": "K6: the registered plan projected above 24 CPU-hours and no GPU is available; the "
                               "corrected dry run (amendment 4) projected about 59 h for this plan"},
}


def _write(path: Path, obj) -> str:
    blob = json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False)
    path.write_text(blob)
    return hashlib.sha256(blob.encode()).hexdigest()


def build_bundle(out: Path, lexicon_doc: dict, alias_table: dict, family_sets: dict, split_manifests: dict,
                 measured: dict) -> dict:
    """Write every frozen file and return name -> SHA-256 (the digests go into the freeze entry)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dig = {"lexicon.json": _write(out / "lexicon.json", lexicon_doc),
           "hypotheses.json": _write(out / "hypotheses.json", HYPOTHESES),
           "settings.json": _write(out / "settings.json", {**SETTINGS, "measured_at_S9": measured}),
           "alias_table.json": _write(out / "alias_table.json", alias_table),
           "family_sets.json": _write(out / "family_sets.json", family_sets)}
    for name, man in split_manifests.items():
        dig[f"split_{name}.json"] = _write(out / f"split_{name}.json", man)
    return dig


def freeze(ledger, digests: dict) -> dict:
    """Freeze entry with every digest, then anchor its batch: scoring is valid only after this anchor (eq. 28)."""
    ledger.append("freeze", payload=digests, note="lexicon, H1-H4, margins, thresholds, alias table, splits")
    return ledger.anchor()


def osf_registration(digests: dict, lexicon_sha: str) -> str:
    lines = ["# Pre-registration: sandbox-artefact audit of deep-learning malware family attribution (A15)", "",
             "Hypotheses H1-H4, margins and decision rules are given in hypotheses.json and settings.json; the "
             "token-level lexicon is embargoed and shared with reviewers.", "",
             f"SHA-256 of the lexicon file: {lexicon_sha}", "", "| File | SHA-256 |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in sorted(digests.items())]
    lines += ["", "Kill criteria K3-K6 and the K5 action are fixed in settings.json. Any change after this "
              "registration is reported as a deviation."]
    return "\n".join(lines) + "\n"
