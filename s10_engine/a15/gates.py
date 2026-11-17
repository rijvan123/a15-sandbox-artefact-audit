"""Kill criteria K3-K6, predictions P3-P4, the paired-test rule and the residual-data scan (S6, S7, S8)."""
from __future__ import annotations

import re

from .config import (DELTA, K4_MIN_AGREEMENT, K5_MAX_HALF_WIDTH, K6_MAX_CPU_HOURS, PAIRED_MIN_FAMILIES,
                     PAIRED_MIN_TRACES)
from .stats import half_width, n_required


def k4_label_agreement(pairs: list[tuple[str, str]]) -> dict:
    """Agreement of an installation's label with an independent labeller on shared SHA-256s (K4: >= 80%)."""
    n = len(pairs)
    agree = sum(a == b and a != "" for a, b in pairs)
    rate = agree / n if n else float("nan")
    return {"n_shared": n, "agreement": rate, "K4_triggered": (n > 0 and rate < K4_MIN_AGREEMENT),
            "evaluable": n > 0}


def k5_p4(measured_hw: float, n_dry: float, rec: float, deff: float, n_full: float) -> dict:
    """P4: dry-run half-width within +/-25% of equation S6 at the dry-run size; K5: projected half-width <= 2.

    Amendment 5 (5 Nov 2026): the record also gives the margin in force (DELTA) and whether the projected half-width
    exceeds it; K5 itself keeps its registered limit of 2 points."""
    formula = half_width(n_dry, rec, deff, 1.0)
    ratio = measured_hw / formula if formula > 0 else float("nan")
    c_f = ratio if ratio == ratio else 1.0
    projected = half_width(n_full, rec, deff, c_f)
    return {"measured_hw_dry": measured_hw, "formula_hw_dry": formula, "ratio": ratio,
            "P4_holds": bool(0.75 <= ratio <= 1.25), "c_F": c_f, "projected_hw_full": projected,
            "N_req_planning": n_required(rec, deff, c_f, DELTA),
            "K5_triggered": bool(projected > K5_MAX_HALF_WIDTH),
            "margin_in_force": DELTA, "projected_exceeds_margin": bool(projected > DELTA),
            "K5_action": "widen windows or narrow families; else raise the margin to 3 points before scoring"
            if projected > K5_MAX_HALF_WIDTH else ""}


def k6_projection(sec_per_trace: dict, n_train: dict, epochs: float = 10.0, eval_factor: float = 0.35,
                  runs_per_cell: int = 39, origins: dict | None = None) -> dict:
    """Projected CPU hours: per learner, installation and origin, 20 search + 4 dropout + 10 final + 5 probe runs."""
    origins = origins or {inst: 3 for inst in n_train}
    hours = {}
    for kind, s in sec_per_trace.items():
        tot = sum(s * n * epochs * runs_per_cell * (1 + eval_factor) * origins[inst] for inst, n in n_train.items())
        hours[kind] = tot / 3600
    total = sum(hours.values())
    return {"hours_by_learner": hours, "total_hours": total, "K6_triggered": bool(total > K6_MAX_CPU_HOURS),
            "triage": "full factorial for Transformer and LightGBM only; BiLSTM and 1D-CNN unmasked and F-masked"
            if total > K6_MAX_CPU_HOURS else "", "assumptions": {"epochs": epochs, "eval_factor": eval_factor,
                                                                   "runs_per_cell": runs_per_cell}}


def paired_test_rule(shared_by_family: dict) -> dict:
    fams = [f for f, n in shared_by_family.items() if n > 0]
    n = sum(shared_by_family.values())
    return {"shared_traces": n, "families": len(fams),
            "qualifies": n >= PAIRED_MIN_TRACES and len(fams) >= PAIRED_MIN_FAMILIES}


PRIVACY_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "ipv6": re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){4,7}[0-9a-fA-F]{1,4}\b"),
    "sid": re.compile(r"\bS-1-5-21-\d+-\d+-\d+"),
    "mac": re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"),
    "user_path": re.compile(r"(?i)\\users\\(?!\*\\|public\\)[^\\|;]+\\"),
}


def residual_data_scan(strings: list[str], n_traces: int) -> dict:
    """Release gate of the privacy plane: any hit blocks release (S7)."""
    hits = {k: sum(bool(p.search(s)) for s in strings) for k, p in PRIVACY_PATTERNS.items()}
    per_1000 = {k: 1000 * v / max(n_traces, 1) for k, v in hits.items()}
    return {"hits": hits, "per_1000_traces": per_1000, "release_blocked": any(hits.values()),
            "strings_scanned": len(strings)}
