#!/usr/bin/env python
"""Strand 6 (c3) (spec solve/specs/SPEC_strand6_validation.md (c3); builder analyst): lane strand B's finite collar dilation identity on the
open 3^3 cube with the gate's realization and seed (2a's packets, regulator and seed bilinears; not lane's eight cell filters).

Geometry: sites x in {0,1,2}^3 (x_0 fastest), centre o = (1,1,1), collar B = the other 26 sites; the two-component sine stencil with the
zero-extended shifts (the native cube's rule), D[x, y] += sgn sigma_j/(2i) for y = x + sgn e_j inside.  Lane's parameters (B1): a = 1,
m_j^2 = (1/4, 3/4, 1, 3/2), orientations (+, -, -, +), the light packet j = 1 (m = 1/2) carrying the seed, Lambda = 1/2, l = 1/Lambda = 2,
g_M = 1/80, g_B = 1/200; the candidate psi = 1 + (s - 1) e_o, N = 1 + (n - 1) e_o at (s, n) = (1.05, 0.95) and (1.1, 0.9).
F(N, psi, Psi) = W(N, psi) + <Psi, H_exc(N, psi) Psi> (B3), W = sum_j eta_j Tr L (E_j1 - E_j0) (2a's W_0 with these masses).
H_exc on the charge-2 pair block (54 particle modes, 1431 pairs), the gate's construction H = e^S [Br (x) 1 + 1 (x) Br] e^S (Dressed.H on the
(2,0) block; e^{-2 S} on the one-particle block is 1), Br = B_0(L) = E_0^(1/2) L E_0^(1/2), S solving the Sylvester equation with the
unit-lapse one-body B_0(Psi) (the gate's denominators eps_i1 + eps_i2 + eps_j1 + eps_j2) and source v_pp, the (2,2) particle part of the
gate's seed; written in lane's natural sea chart (B7): the particle modes are the columns of W_+ = 2^(-1/2) (1; U) (U the light packet's
polar unitary), so fixed pair amplitudes transport with the background, and the pair basis is smooth in it.  The gate's eigenbasis enters
only to solve the Sylvester equation and cancels in the operator.  --check compares this construction's spectrum with the gate's own
Dressed (2,0) block on the 2 x 2 x 2 cube.
Probes (B13): F[p] by central differences at fixed Psi (h = 1e-4, 5e-5, Richardson): T_B = F[(1_B, 0)], T_F = F[(f, 0)], C_F = F[(0, f)],
R_2 = F[(e_o, 0)], f the indicator of the six face-centre neighbours of o; w[p] and <h[p]> separately along the dilation tangent (B6)
p_D = (psi/2 - (s - 1) f/2, -(n - 1) f/2).  The generator (B4): xi_i = x_i - 1, grad_i = (T_i - T_i^dag)/2 with the zero-extended forward
shift, V = sum_i diag(xi_i) grad_i - diag(V 1), A_sp = (V - V^dag)/2 (x) 1_2; g = (A_sp + U^dag A_sp U - U^dag dU)/2 (B8) with dU the
central difference of U along p_D; J_2 = dGamma_2(g) on the pair block; kappa = -2 Re <J_2 Psi, (H - eps) Psi> and the commutator form.
Pieces: Q_a = T_B/2, D_vac = w[p_D], D_mat = <h[p_D]> - kappa, D_bdy = (s - 1) T_F/2 + (n - 1) C_F/2, R_stat = -s R_2/2 + kappa; the closure
Q_a - (D_vac + D_mat + D_bdy + R_stat).  Two pair vectors: the lowest eigenvector of H at the candidate, and a fixed random normalized
vector (seed 7), for which kappa is not zero.  Writes <out>/collar_c3.json (or <out>/check_c8.json)."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import sys
import time

import numpy as np
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "..", "strand2", "frozen"), os.path.join(HERE, "..", "strand2"), os.path.join(HERE, "frozen"), HERE):
    sys.path.insert(0, _p)
import strand2a_lib as S2  # noqa: E402

M2 = (0.25, 0.75, 1.0, 1.5)
ETAS = (1, -1, -1, 1)
GM, GB, LAM = 1.0 / 80, 1.0 / 200, 0.5


def lattice_c27():
    positions = [(x0, x1, x2) for x2 in range(3) for x1 in range(3) for x0 in range(3)]
    pidx = {p: i for i, p in enumerate(positions)}
    n = len(positions)
    D = np.zeros((2 * n, 2 * n), dtype=complex)
    for p in positions:
        x = pidx[p]
        for j in range(3):
            for sgn in (1, -1):
                q = list(p)
                q[j] += sgn
                if tuple(q) in pidx:
                    y = pidx[tuple(q)]
                    D[2 * x:2 * x + 2, 2 * y:2 * y + 2] += sgn * S2.SIG[j] / 2j
    pos = np.array(positions, float)
    dist = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    return {"name": "C27", "nsite": n, "ncomp": 2, "DW": S2.herm(D), "positions": pos, "dist": dist, "dirs": [0, 1, 2],
            "coord": pos.astype(int)}


class Pairs:
    def __init__(self, n):
        self.n = n
        self.pairs = np.array([(a, b) for a in range(n) for b in range(a + 1, n)])
        self.a, self.b = self.pairs[:, 0], self.pairs[:, 1]

    def one_body(self, A):
        a, b = self.a, self.b
        dbb = (b[:, None] == b[None, :])
        daa = (a[:, None] == a[None, :])
        dba = (b[:, None] == a[None, :])
        dab = (a[:, None] == b[None, :])
        return (A[np.ix_(a, a)] * dbb + A[np.ix_(b, b)] * daa - A[np.ix_(a, b)] * dba - A[np.ix_(b, a)] * dab)

    def two_body(self, T):
        a, b = self.a, self.b
        A1, B1, A2, B2 = a[:, None], b[:, None], a[None, :], b[None, :]
        return T[A1, B1, A2, B2] - T[A1, B1, B2, A2] - T[B1, A1, A2, B2] + T[B1, A1, B2, A2]

    def unitary(self, R):
        a, b = self.a, self.b
        return R[np.ix_(a, a)] * R[np.ix_(b, b)] - R[np.ix_(b, a)] * R[np.ix_(a, b)]


def W_vac(lat, psi, N, m2=M2, etas=ETAS):
    return float(sum(S2.Packet(lat, np.sqrt(x), e, psi, N).vacuum_energy() for x, e in zip(m2, etas)))


def H_natural(lat, psi, N, PR, m_light=0.5, gM=GM, gB=GB, Lam=LAM, dressed=True):
    light = S2.Packet(lat, m_light, +1, psi, N)
    nW = light.nW
    Qp = light.P @ np.vstack([np.eye(nW), light.U]) / np.sqrt(2.0)          # natural particle modes in the mode layout
    Br = light.B(0)
    D1 = light.B(0, light.L1)
    terms = S2.seed_bilinears(lat, psi, gM, gB, Lam)
    # group the terms by their first matrix (identity of the object) to form T'_pp = sum w m1 (x) m2
    groups = {}
    for w, M1, M2_ in terms:
        key = id(M1)
        g = groups.setdefault(key, [M1, []])
        g[1].append((w, M2_))
    Tp = np.zeros((nW, nW, nW, nW), dtype=complex)
    cache = {}

    def comp(M):
        k = id(M)
        if k not in cache:
            cache[k] = Qp.conj().T @ M @ Qp
        return cache[k]
    for M1, lst in groups.values():
        m1 = comp(M1)
        m2 = sum(w * comp(M2_) for w, M2_ in lst)
        Tp += np.einsum("ik,jl->ijkl", m1, m2, optimize=True)
    v_nat = PR.two_body(Tp)
    H1 = PR.one_body(Br)
    if not dressed:
        return H1, light, v_nat
    eps, R = np.linalg.eigh(S2.herm(D1))
    Up = PR.unitary(R)
    v_eig = Up.conj().T @ v_nat @ Up
    E = eps[PR.a] + eps[PR.b]
    S_eig = v_eig / (E[:, None] + E[None, :])
    S_nat = Up @ S_eig @ Up.conj().T
    S_nat = S2.herm(S_nat)
    eS = sla.expm(S_nat)
    return eS @ H1 @ eS, light, v_nat


def generator(lat):
    pos = lat["coord"]
    n = len(pos)
    idx = {tuple(p): i for i, p in enumerate(pos)}
    V0 = np.zeros((n, n))
    for i in range(3):
        Tsh = np.zeros((n, n))
        for x in range(n):
            q = pos[x].copy()
            q[i] += 1
            y = idx.get(tuple(q))
            if y is not None:
                Tsh[x, y] = 1.0
        grad = (Tsh - Tsh.T) / 2.0
        V0 += np.diag(pos[:, i] - 1.0) @ grad
    V = V0 - np.diag(V0 @ np.ones(n))
    return V


def check_c8(out):
    lat = S2.lattice("C8")
    rng = np.random.default_rng(3)
    psi = 1 + 0.05 * rng.uniform(-1, 1, 8)
    N = 1 + 0.05 * rng.uniform(-1, 1, 8)
    m, gM, gB, Lam = 0.1, 0.2, 1.0, 1.0
    light = S2.Light(lat, m, psi, N)
    terms = S2.seed_bilinears(lat, psi, gM, gB, Lam)
    T = S2.seed_tensor(terms, 2 * light.nW)
    Tp = S2.transform(T, light.Q)
    WA = S2.two_two(Tp, light.nW)
    v = S2.vIJ_from_WA(WA, light.nW)
    dr = S2.Dressed(light, v, blocks=((0, 0), (1, 0), (2, 0)))
    Br_r = light.rbasis(light.pk.B(0))
    Hg = dr.H((2, 0), Br_r)
    PR = Pairs(light.nW)
    Hn, _, _ = H_natural(lat, psi, N, PR, m_light=m, gM=gM, gB=gB, Lam=Lam)
    eg = np.linalg.eigvalsh(S2.herm(Hg))
    en = np.linalg.eigvalsh(S2.herm(Hn))
    Hg0 = dr.H((2, 0), Br_r, dressed=False)
    Hn0, _, _ = H_natural(lat, psi, N, PR, m_light=m, gM=gM, gB=gB, Lam=Lam, dressed=False)
    res = {"max_eig_diff_dressed": float(np.max(np.abs(eg - en))), "max_eig_diff_undressed": float(np.max(np.abs(np.linalg.eigvalsh(S2.herm(Hg0)) - np.linalg.eigvalsh(S2.herm(Hn0))))),
           "lowest": [float(eg[0]), float(en[0])], "dim": int(len(eg))}
    print(json.dumps(res), flush=True)
    with open(os.path.join(out, "check_c8.json"), "w") as fh:
        json.dump(res, fh, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.check:
        check_c8(a.out)
        return
    t0 = time.time()
    lat = lattice_c27()
    n = lat["nsite"]
    o = 13
    assert tuple(lat["coord"][o]) == (1, 1, 1)
    e_o = np.zeros(n)
    e_o[o] = 1.0
    one = np.ones(n)
    one_B = one - e_o
    f = np.array([1.0 if np.sum(np.abs(lat["coord"][x] - 1)) == 1 else 0.0 for x in range(n)])
    V = generator(lat)
    res = {"meta": {"V_one_max": float(np.max(np.abs(V @ one))), "V_eo_plus_f_over_2_max": float(np.max(np.abs(V @ e_o + f / 2))),
                    "f_count": int(f.sum())}, "points": []}
    PR = Pairs(2 * n)
    A_sp = np.kron(0.5 * (V - V.T), np.eye(2))
    hs = (1e-4, 5e-5)
    for (s, nn) in ((1.05, 0.95), (1.1, 0.9)):
        psi0 = one + (s - 1) * e_o
        N0 = one + (nn - 1) * e_o
        H0, light0, _ = H_natural(lat, psi0, N0, PR)
        H0 = S2.herm(H0)
        W0v = W_vac(lat, psi0, N0)
        ev, evec = np.linalg.eigh(H0)
        rng = np.random.default_rng(7)
        z = rng.normal(size=H0.shape[0]) + 1j * rng.normal(size=H0.shape[0])
        vecs = {"ground state": evec[:, 0], "random (seed 7)": z / np.linalg.norm(z)}
        probes = {"T_B": (one_B, 0 * one), "T_F": (f, 0 * one), "C_F": (0 * one, f), "R_2": (e_o, 0 * one),
                  "p_D": (psi0 / 2 - (s - 1) * f / 2, -(nn - 1) * f / 2)}
        Hcache = {}
        for name, (pp, pN) in probes.items():
            for h in hs:
                for sg in (+1, -1):
                    Hs, lt, _ = H_natural(lat, psi0 + sg * h * pp, N0 + sg * h * pN, PR)
                    Hcache[(name, h, sg)] = (S2.herm(Hs), W_vac(lat, psi0 + sg * h * pp, N0 + sg * h * pN), lt.U)
            print("s %g n %g probe %s built (%.0f s)" % (s, nn, name, time.time() - t0), flush=True)
        # dU along p_D
        dU = []
        for h in hs:
            dU.append((Hcache[("p_D", h, 1)][2] - Hcache[("p_D", h, -1)][2]) / (2 * h))
        dU = (4 * dU[1] - dU[0]) / 3
        U = light0.U
        g = 0.5 * (A_sp + U.conj().T @ A_sp @ U - U.conj().T @ dU)
        J2 = PR.one_body(g)
        pt = {"s": s, "n": nn, "eps_ground": float(ev[0]), "gap": float(ev[1] - ev[0]), "W": W0v,
              "g_antihermitian_defect": float(np.max(np.abs(g + g.conj().T))), "vectors": {}}
        for vname, Psi in vecs.items():
            eps = float(np.real(Psi.conj() @ H0 @ Psi))
            r = H0 @ Psi - eps * Psi
            kappa = float(-2 * np.real((J2 @ Psi).conj() @ r))
            kappa_comm = float(np.real(Psi.conj() @ (J2 @ H0 - H0 @ J2) @ Psi))
            Fd, wd, hd = {}, {}, {}
            for name in probes:
                vals = []
                for h in hs:
                    Fp = Hcache[(name, h, 1)]
                    Fm = Hcache[(name, h, -1)]
                    w_ = (Fp[1] - Fm[1]) / (2 * h)
                    h_ = float(np.real(Psi.conj() @ (Fp[0] - Fm[0]) @ Psi)) / (2 * h)
                    vals.append((w_, h_))
                wR = (4 * vals[1][0] - vals[0][0]) / 3
                hR = (4 * vals[1][1] - vals[0][1]) / 3
                wd[name], hd[name], Fd[name] = wR, hR, wR + hR
                Fd[name + "_step_agreement"] = abs((vals[0][0] + vals[0][1]) - (vals[1][0] + vals[1][1]))
            Qa = Fd["T_B"] / 2
            Dvac = wd["p_D"]
            Dmat = hd["p_D"] - kappa
            Dbdy = (s - 1) * Fd["T_F"] / 2 + (nn - 1) * Fd["C_F"] / 2
            Rstat = -s * Fd["R_2"] / 2 + kappa
            pt["vectors"][vname] = {"eps": eps, "eigen_residual_norm": float(np.linalg.norm(r)), "kappa": kappa, "kappa_commutator": kappa_comm,
                                    "probes": Fd, "w": wd, "h": hd, "Q_a": Qa, "D_vac": Dvac, "D_mat": Dmat, "D_bdy": Dbdy, "R_stat": Rstat,
                                    "closure": Qa - (Dvac + Dmat + Dbdy + Rstat),
                                    "linearity_check": Fd["p_D"] - (Fd["T_B"] / 2 + s * Fd["R_2"] / 2 - (s - 1) * Fd["T_F"] / 2 - (nn - 1) * Fd["C_F"] / 2)}
            print("s %g n %g %s: Q_a %.10f D_vac %.10f D_mat %.10f D_bdy %.3e R_stat %.3e kappa %.3e closure %.2e (%.0f s)" % (
                s, nn, vname, Qa, Dvac, Dmat, Dbdy, Rstat, kappa, pt["vectors"][vname]["closure"], time.time() - t0), flush=True)
        res["points"].append(pt)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "collar_c3.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
