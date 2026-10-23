"""Strand 0 controls C1 to C12 (spec sec. 5) at the stated tolerances, on small cases.  On the box:
python -m pytest -q tests/strand0_controls_test.py (from solve/code/strand0)."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand0_lib as L  # noqa: E402
import strand0_partA_data_v2 as D  # noqa: E402
import strand0_ringL_onebody_v2 as RL  # noqa: E402
import native_core as nc  # noqa: E402
import native_sources as ns  # noqa: E402

LAT = nc.lattice("ring4", (0.0, 0.0, 0.0))
FOCK = ns.Fock(8)
PSI_B2 = 1 + 0.2 * np.cos(np.pi * np.arange(4) / 2)


@pytest.fixture(scope="module")
def comp_b2():
    law = L.Law(0.1, PSI_B2)
    return L.Completion(law, FOCK, 0.1, sectors=range(2, 8), dress=(4, 5, 6))


def test_C1_W_map_and_gamma():
    name, W, res = D.choose_W(0.1)
    assert res[name] <= 1e-12
    gams, gc = D.gamma_controls(np.kron(np.eye(4), W), FOCK, LAT, 0.1)
    for k, v in gc.items():
        assert v <= 1e-12, k


def test_C2_C3_polar_and_spectrum():
    law = L.Law(0.1, PSI_B2)
    assert np.linalg.norm(law.A - law.U @ law.E0) <= 1e-12
    assert np.linalg.norm(law.U.conj().T @ law.U - np.eye(4)) <= 1e-12
    assert np.linalg.norm(law.E1 - law.U @ law.E0 @ law.U.conj().T) <= 1e-12
    ev = np.sort(np.linalg.eigvalsh(L.herm(law.h_star())))
    eps = np.linalg.eigvalsh(law.B0)
    assert np.max(np.abs(ev - np.sort(np.concatenate([-eps, eps])))) <= 1e-12
    lu = L.Law(0.2, 1.1 * np.ones(4))
    p = 2 * np.pi * np.arange(4) / 4
    assert np.max(np.abs(np.sort(lu.eps) - np.sort(np.sqrt(0.04 + 1.1 ** -4 * np.sin(p) ** 2)))) <= 1e-12
    evn = np.sort(np.linalg.eigvalsh(L.native_h(LAT, 0.2, 1.1 * np.ones(4))))
    assert np.max(np.abs(np.sort(np.linalg.eigvalsh(L.herm(lu.h_star()))) - evn)) <= 1e-12


def test_C4_free_clock_derivative(comp_b2):
    law = comp_b2.law
    for N in (4, 5, 6):
        sec = FOCK.sector(N)
        for x in range(4):
            cb = sec.bilinear(law.dh_dN(x)) + np.trace(law.X(x)).real * np.eye(sec.d)
            assert np.linalg.norm(cb - comp_b2.rhoP(x, N, dressed=False)) <= 1e-12
    w, V = np.linalg.eigh(L.herm(FOCK.sector(4).bilinear(law.h_star())))
    assert abs(w[0] + np.trace(law.B0).real) <= 1e-12
    assert abs(abs(np.vdot(V[:, 0], comp_b2.Omega)) - 1) <= 1e-12


def test_C5_completion(comp_b2):
    c = comp_b2
    assert abs(c.Omega.conj() @ c.Vno[4] @ c.Omega) <= 1e-12
    for N in (4, 5, 6):
        S = c.S[N]
        assert np.linalg.norm(S @ c.nex[N] - c.nex[N] @ S) <= 1e-12
        assert np.linalg.norm(S - S.conj().T) <= 1e-12
        assert np.linalg.norm(c.monomial_sum(N) - c.Vstar[N]) <= 1e-12
        low = [P for k, P in c.proj[N].items() if k <= 1]
        if low:
            P = np.hstack(low)
            assert np.linalg.norm(S @ P) <= 1e-12
            assert np.linalg.norm(P.conj().T @ (c.H_exc(N) - c.H_free(N)) @ P) <= 1e-12
        assert np.linalg.eigvalsh(L.herm(c.H_exc(N)))[0] >= -1e-12


def test_C5e_second_order():
    law = L.Law(0.1, PSI_B2)
    r = []
    for U in (0.01, 0.1):
        c = L.Completion(law, FOCK, U, sectors=range(2, 8), dress=(5,))
        r.append(np.linalg.norm(c.H_exc(5) - c.H_free(5) - c.Vstar[5]))
    assert abs(r[0] / r[1] - 0.01) <= 0.001


def test_C6_N_linearity(comp_b2):
    law = comp_b2.law
    for x in range(4):
        rho = comp_b2.rhoP(x, 5)
        Np, Nm = np.ones(4), np.ones(4)
        Np[x] += 1e-3
        Nm[x] -= 1e-3
        Bp = L.herm(law.E0h @ np.diag(Np * law.psi) @ law.E0h)
        Bm = L.herm(law.E0h @ np.diag(Nm * law.psi) @ law.E0h)
        d = (comp_b2.H_exc(5, B0=Bp) - comp_b2.H_exc(5, B0=Bm)) / 2e-3
        assert np.linalg.norm(d - rho) <= 1e-9


def test_C7_reference():
    lu = L.Law(0.1, 1.3 * np.ones(4))
    assert max(abs(lu.psi[x] * (lu.E1[x, x] - lu.E0[x, x])) for x in range(4)) <= 1e-12
    lb = L.Law(0.1, PSI_B2)
    s = sum((lb.psi[x] * (lb.E1[x, x] - lb.E0[x, x])).real for x in range(4))
    assert abs(s - np.trace(np.diag(lb.psi) @ (lb.E1 - lb.E0)).real) <= 1e-12


def test_C8_native_identification(comp_b2):
    G, nsea = ns.sea_contractions(LAT, 0.1, 8)
    name, W, res = D.choose_W(0.1)
    for N in (4, 5):
        sec = FOCK.sector(N)
        co0 = ns.Coefficients(LAT, FOCK, N, 0.1, 0.0, G, "wick")
        for s in (np.ones(4), np.array([1.1, 0.9, 1.05, 0.95])):
            for x in range(4):
                Mx = L.native_midpoint(LAT, 0.1, s, x)
                bil = sec.bilinear(Mx) - np.sum(Mx * G) * np.eye(sec.d)
                assert np.linalg.norm(co0.scalar_source(x, s) - bil) <= 1e-12
        law1 = L.Law(0.1, np.ones(4))
        comp1 = L.Completion(law1, FOCK, 0.0, sectors=range(2, 8), dress=(N,))
        Gm = L.gamma_of(np.kron(np.eye(4), W), sec)
        for x in range(4):
            rm, _ = L.normal_ordered_bilinear(sec, law1.midpoint(x), comp1.Omega, comp1.N0, FOCK)
            assert np.linalg.norm(Gm @ co0.scalar_source(x, np.ones(4)) @ Gm.conj().T - rm) <= 1e-12


def test_C10_conserving_part():
    G, nsea = ns.sea_contractions(LAT, 0.1, 8)
    name, W, res = D.choose_W(0.1)
    law1 = L.Law(0.1, np.ones(4))
    comp = L.Completion(law1, FOCK, 0.1, sectors=range(2, 8), dress=(5,))
    sec = FOCK.sector(5)
    Gm = L.gamma_of(np.kron(np.eye(4), W), sec)
    co = ns.Coefficients(LAT, FOCK, 5, 0.1, 0.1, G, "wick")
    Hm = sum(Gm @ co.scalar_source(x, np.ones(4)) @ Gm.conj().T for x in range(4))
    Dn = Hm - comp.H_free(5)
    Pc = sum(P @ (P.conj().T @ Dn @ P) @ P.conj().T for P in comp.proj[5].values())
    assert np.linalg.norm(Pc - comp.Vstar[5]) <= 1e-12


def test_C11_contact1():
    law = L.Law(0.1, PSI_B2)
    c = L.Completion(law, FOCK, 1e-4, sectors=range(2, 8), dress=(5,))
    for x in range(4):
        num = (c.rhoP(x, 5) - c.rhoP(x, 5, dressed=False)) / 1e-4
        form = c.contact1(x, 5) / 1e-4
        assert np.linalg.norm(num - form) / np.linalg.norm(form) <= 1e-3


@pytest.mark.parametrize("L_", [8, 16])
def test_C12_ringL(L_):
    r = RL.one(L_, 0.2)
    assert max(r["C12"].values()) <= 1e-12
    assert r["C12_U_eigen"] <= 1e-12
