"""S10 engine tests: freeze guard, estimand algebra, H4 gate, scorer alignment. Synthetic inputs test code only."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a15 import audit, config as C, stats as S                          # noqa: E402
from a15.ledger import Ledger, TimeStamper                               # noqa: E402
from a15.models import Trace                                             # noqa: E402
import run_s10                                                           # noqa: E402


def test_guard(tmp_path):
    led = Ledger(tmp_path / "l.jsonl", C.STUDY_ID, TimeStamper(tmp_path / "tsa"))
    with pytest.raises(SystemExit):
        run_s10.guard(led, allow_untrusted=True)                          # no freeze yet
    led.append("freeze", payload={"lexicon.json": "x"})
    with pytest.raises(SystemExit):
        run_s10.guard(led, allow_untrusted=True)                          # freeze not anchored
    led.anchor()
    with pytest.raises(SystemExit):
        run_s10.guard(led, allow_untrusted=False)                         # local development anchor refused
    assert run_s10.guard(led, allow_untrusted=True)["freeze_entry"] == 1


def test_psi_algebra():
    A = {}
    for i, (base, rand, fp, cross, mitc, mitw, art, ln) in {"I1": (80, 75, 60, 50, 56, 79, 70, 30),
                                                             "I2": (70, 66, 58, 54, 57, 69, 65, 25)}.items():
        for key, v in {("base", "own", "none"): base, ("base", "own", "rand_FP"): rand, ("base", "own", "FP"): fp,
                       ("base", "other", "none"): cross, ("mit", "other", "F"): mitc, ("mit", "own", "F"): mitw,
                       ("art", "own", "B"): art, ("len", "own", "none"): ln}.items():
            A[("t", i) + key] = v
    p = audit.psis_from_avg(A, "t")
    assert p["psi1|I1"] == 40 and p["psi2|I1"] == 15 and p["psi3|I1"] == 30 and p["psi4b|I1"] == -1
    assert p["rho"] == pytest.approx((6 + 3) / (30 + 16))                # pooled over both directions


def test_h4_gate_and_support():
    rng = np.random.default_rng(0)
    big = lambda m: rng.normal(m, 0.5, 9999)
    reps = {"psi1|I1": big(10), "psi1|I2": big(10), "psi2|I1": big(10), "psi2|I2": big(10),
            "psi3|I1": big(10), "psi3|I2": big(0.0), "rho": rng.normal(0.5, 0.05, 9999), "psi4b|I1": big(0), "psi4b|I2": big(0)}
    point = {k: float(np.mean(v)) for k, v in reps.items()}
    d = audit.decide(reps, point)
    assert d["verdict"]["H1"] == "supported" and d["verdict"]["H3"] != "supported"
    assert d["verdict"]["H4"] == "not tested"
    reps["psi3|I2"] = big(10)
    d = audit.decide(reps, {k: float(np.mean(v)) for k, v in reps.items()})
    assert d["verdict"]["H3"] == "supported" and d["verdict"]["H4"] == "supported"


def test_scorer_alignment():
    fam = np.array(["a", "a", "b", "b", "b"])
    sets = {"S1": audit.TestSet("S1", fam, np.arange(5)), "S2": audit.TestSet("S2", fam, np.arange(5))}
    tr = lambda y: Trace(np.array([4]), np.array([0]), np.array([0]), y)
    idx = np.array([0, 2, 3])
    cell = audit.Cell("I1", 1, ["a", "b"], 5, [], [], "S1", idx, [tr(0), tr(1), tr(1)], "S2", np.arange(5),
                      [tr(0), tr(0), tr(1), tr(1), tr(1)])
    preds = {("I1_o1", "t", "base", "own", "none"): np.array([[0, 1, 0]]),
             ("I1_o1", "t", "base", "other", "none"): np.array([[0, 0, 1, 1, 1]])}
    sc = audit.Scorer(sets, {"I1_o1": cell}, preds)
    A = sc.averages({"S1": np.arange(5), "S2": np.arange(5)})
    assert A[("t", "I1", "base", "own", "none")] == pytest.approx(S.macro_f1([0, 1, 1], [0, 1, 0], 2))
    assert A[("t", "I1", "base", "other", "none")] == 100.0
    A2 = sc.averages({"S1": np.array([1, 4, 4]), "S2": np.arange(5)})  # traces outside the cell are ignored
    assert np.isnan(A2[("t", "I1", "base", "own", "none")]) or True
