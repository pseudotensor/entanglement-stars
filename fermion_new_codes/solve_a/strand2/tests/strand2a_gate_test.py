"""Strand 2a gate G1 to G6 at the spec's tolerances on small cases (spec sec. 4).  On the box, from solve/code/strand2:
python -m pytest -q tests/strand2a_gate_test.py"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2a_lib as S2  # noqa: E402
import strand2a_gate as G  # noqa: E402

CASES = [("L2", "nonuniform", 0.05, 0.5, 1.0), ("R4", "nonuniform", 0.2, 1.0, 1.0), ("R4", "random7", 0.05, 0.0, np.inf)]


@pytest.mark.parametrize("latname,bg,gM,gB,Lam", CASES)
def test_G1_G2b_G3_fock_crosscheck(latname, bg, gM, gB, Lam):
    lat = S2.lattice(latname)
    psi, N = S2.backgrounds(lat)[bg]
    o = G.light_case(lat, bg, psi, N, 0.1, gM, gB, Lam, True, None)
    r = o["rec"]
    assert r["G1"]["S_hermitian"] <= 1e-12
    assert r["G1"]["S_on_nexc_le1"] == 0.0
    assert r["G1c_vIJ_alg_minus_fock"] <= 1e-12
    for blk, item in r["G1c_blocks"].items():
        for key, val in item.items():
            assert val <= 1e-11, (blk, key, val)
    assert r["G1d_fock"] <= 1e-12
    assert abs(r["G2b"]["lowest_bilinear_plus_TrB10"]) <= 1e-12
    assert r["G2b"]["H_exc_Omega"] <= 1e-12
    assert r["G3"]["P1_(1,0)"] <= 1e-12 and r["G3"]["P1_(0,1)"] <= 1e-12
    assert r["G3"]["P2_(2,0)"] > 0
    assert abs(r["G6"]["ratio"]["1,0"] - r["G6"]["expected"]) <= 1e-10


@pytest.mark.parametrize("latname", ["L2", "R4", "C8"])
def test_G2_G4_vacuum(latname):
    lat = S2.lattice(latname)
    for bg, (psi, N) in S2.backgrounds(lat).items():
        r = G.vac_case(lat, bg, psi, N, 0.1, 1.0, None)
        assert r["G2a_max"] <= 1e-12
        assert abs(r["sum_rule"]) <= 1e-12
        for v in r["G4_affinity"].values():
            assert v <= 1e-12
        if bg == "flat":
            assert abs(r["W0"]) <= 1e-12
            assert max(abs(v) for v in r["dW0_dN"]) <= 1e-12
        assert r["K_NN_W0_max"] <= 1e-12


def test_C8_light_one_case():
    lat = S2.lattice("C8")
    psi, N = S2.backgrounds(lat)["nonuniform"]
    o = G.light_case(lat, "nonuniform", psi, N, 0.1, 0.05, 0.5, 1.0, False, None)
    r = o["rec"]
    assert r["G1"]["S_hermitian"] <= 1e-12
    assert r["G1"]["S_on_nexc_le1"] == 0.0
    assert r["G3"]["P1_(1,0)"] <= 1e-12 and r["G3"]["P1_(0,1)"] <= 1e-12
    assert r["G3"]["H_exc_on_Omega"] == 0.0
    assert min(r["min_eig_Hexc"].values()) >= -1e-12
