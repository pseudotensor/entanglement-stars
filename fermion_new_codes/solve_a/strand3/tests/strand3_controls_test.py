"""Strand 3 controls C1 to C7 (spec sec. 6).  Run from solve/code/strand3 with STRAND3_RUN=<run dir>:
python -m pytest -q tests/strand3_controls_test.py.  Output-reading tests skip when the run files are absent."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "orient"))
import strand3_masses as SM  # noqa: E402
import strand3_partC as PC  # noqa: E402
from lattice_entropy import entropy  # noqa: E402

RUN = os.environ.get("STRAND3_RUN", "")


def load(name):
    p = os.path.join(RUN, name)
    if not RUN or not os.path.exists(p):
        pytest.skip("run output %s absent" % name)
    return json.load(open(p))


def test_C1_verify_basic():
    import verify_entropy
    verify_entropy.basic()


@pytest.mark.parametrize("p,d,nk,ref", [((1, 0, 0), 16, 12, 0.7849544816), ((1, 0, 0), 32, 24, 0.7861807702),
                                        ((1, 1, 0), 16, 12, 0.8678995706), ((1, 1, 1), 16, 12, 0.9227654142)])
def test_C1_planar_reproduction(p, d, nk, ref):
    r = load(os.path.join("planar", "planar_%d_%d_%d_d%d_nk%d_q0.json" % (tuple(p) + (d, nk))))
    assert abs(r["eta"] - ref) <= 1e-9


def test_C1_sphere_reproduction():
    r = load(os.path.join("sphere", "sphere_q0.json"))
    assert abs(r["eta_sphere"] - 0.9001144) <= 1e-6


@pytest.mark.parametrize("lam", [1.0, 4.0, 16.0])
def test_C3_masses(lam):
    h1, m, g = SM.masses(lam)
    m2 = m ** 2
    assert abs(np.dot(SM.ETA_SIGN, m2)) <= 1e-12
    J = sum(e * x * np.log(x) for e, x in zip(SM.ETA_SIGN, m2) if x > 0)
    assert abs(J - h1 * g) <= 1e-12
    assert abs(6 * J / np.pi - 1) <= 1e-12


def test_C4_additivity():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 40)) + 1j * rng.normal(size=(40, 40))
    Q, _ = np.linalg.qr(X)
    P = Q[:, :20] @ Q[:, :20].conj().T
    C = P[:16, :16]
    C2 = np.block([[C, np.zeros_like(C)], [np.zeros_like(C), C]])
    assert abs(entropy(C2.copy()) - 2 * entropy(C.copy())) <= 1e-12


def test_C5_partB_projector_identity():
    B = load("partB.json")
    for b in B["backgrounds"].values():
        assert b["A2_max_dev_from_product_projector"] <= 1e-12
        # the entropy of an exact projector computed from eigenvalues with round-off eps ~ 1e-14 is of order
        # n eps |log eps| ~ 4e-11 for 128 modes; the restricted spectrum is checked at 1e-12 directly in the report
        assert abs(b["S_R_A2"]) <= 1e-10


def test_C6_weights():
    for kappa in PC.KAPPAS:
        inner, outer, K = PC.channels(kappa)
        assert abs(sum(w for _, w in inner) + sum(w for _, w in outer) - 1.0) <= 1e-12


def test_C7_bosons():
    B = load("partB.json")
    for b in B["backgrounds"].values():
        assert b["B4_min_eig_B1"] > 0
