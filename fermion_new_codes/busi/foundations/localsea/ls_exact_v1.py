#!/usr/bin/env python
"""ls_exact_v1.py (busi, inbox #447): the second route for ls_zone_v1.py, by exact diagonalization on finite lattices.

DATA script, box CPU only (torch complex128, at most 8 threads). Lattice: periodic Lx x Ly in x, y; the z momentum p_z on a
midpoint grid of nz values (fields independent of z), so each p_z is one dense block of dimension 4 Lx Ly. Operators as in
engine_ops_v1.lattice3d_parts (sine hopping -i/2 alpha to x + e, two-step current -i/4 to x + 2e), the z parts on site:
alpha_z sin p_z and Q_z = sin(2 p_z)/2. h[N, v] = {N, h0}/2 + sum_i {v_i, Q_i}/2. The sea is the negative spectral projector of
h[u] (gapped: m = 0.2, small amplitudes). Densities: rho_x = Re sum_occ psi_x^+ (h0 psi)_x (the allocation {Pi_x, h0}/2),
J_x,i likewise with Q_i, averaged over p_z (the 3D lattice's site densities).

Modes (each compared with the ls_zone_v1 formula evaluated on the SAME discrete momentum grid, so agreement is exact up to the
amplitude corrections):
  two   : E'' of a real flow or clock wave, second difference in the amplitude, against two_point_finite.
  three : the cross term of two flow waves (amplitudes +-A1, +-A2) at K = q1 + q2, against three_point_clean.
  rate1 : the first-order adiabatic density rho1 = Tr(C1 e_x), C1 = i Pdot_ab/(eps_a - eps_b) between bands, Pdot by a central
          difference of P[u(t)] in t, for a flow ramp (linear in the amplitude), against rate_linear.
  rate2 : the same at second order in the amplitude on 1D modulations: (n, udot) and (u, ndot) pairs, fitted on the harmonics
          0 and 2q to the forms n' udot, n udot' and ndot u', u ndot' (reviewer A's (d_t N) div v and grad N . d_t v; the second of
          each pair is excluded by homogeneity and by the absence of undifferentiated flow, and is read as a check); and the
          current's response J1 at second order on (n, ndot), (u, udot) pairs: forms ndot n', ndot' n and udot u', u udot'.
Outputs: <out>/ls_exact_v1_<tag>.json."""
import os
import sys


def _threads():
    t = "8"
    if "--threads" in sys.argv:
        t = sys.argv[sys.argv.index("--threads") + 1]
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = t
    return int(t)


NT = _threads()
import argparse
import json
import math
import time
import numpy as np
import torch

torch.set_num_threads(NT)
PI = math.pi
CD = torch.complex128
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ls_zone_v1 as Z  # noqa: E402

SX = torch.tensor([[0, 1], [1, 0]], dtype=CD)
SY = torch.tensor([[0, -1j], [1j, 0]], dtype=CD)
SZ = torch.tensor([[1, 0], [0, -1]], dtype=CD)
Z2 = torch.zeros(2, 2, dtype=CD)
ALPHA = [torch.cat([torch.cat([Z2, s], 1), torch.cat([s, Z2], 1)], 0) for s in (SX, SY, SZ)]
BETA = torch.diag(torch.tensor([1, 1, -1, -1], dtype=CD))
I4 = torch.eye(4, dtype=CD)


def shift(L, step):
    """permutation matrix S with (S)_{x, x+step} = 1 on a ring of L."""
    S = torch.zeros(L, L, dtype=CD)
    for x in range(L):
        S[x, (x + step) % L] = 1
    return S


class Lat:
    def __init__(self, Lx, Ly, m, ny=1):
        self.Lx, self.Ly, self.m = Lx, Ly, m
        self.ny = ny if Ly == 1 else 1   # Ly = 1: p_y on a midpoint grid of ny values (1D modulation along x)
        Ix = torch.eye(Lx, dtype=CD); Iy = torch.eye(Ly, dtype=CD)
        Sx1, Sx2 = shift(Lx, 1), shift(Lx, 2)
        Sy1, Sy2 = shift(Ly, 1), shift(Ly, 2)
        # site index s = ix * Ly + iy; operator on (site, comp) = kron(site op, 4x4)
        Dx = torch.kron(Sx1, Iy) * (-0.5j); Dx = Dx + Dx.conj().T
        Dy = torch.kron(Ix, Sy1) * (-0.5j); Dy = Dy + Dy.conj().T
        Qx = torch.kron(Sx2, Iy) * (-0.25j); Qx = Qx + Qx.conj().T
        Qy = torch.kron(Ix, Sy2) * (-0.25j); Qy = Qy + Qy.conj().T
        self.ns = Lx * Ly
        Is = torch.eye(self.ns, dtype=CD)
        self.h_xy = torch.kron(Dx, ALPHA[0]) + torch.kron(Dy, ALPHA[1]) + m * torch.kron(Is, BETA)
        self.Qx = torch.kron(Qx, I4); self.Qy = torch.kron(Qy, I4)
        self.Is4 = torch.eye(4 * self.ns, dtype=CD)
        self.az = torch.kron(Is, ALPHA[2])
        self.ay = torch.kron(Is, ALPHA[1])
        X, Y = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing="ij")
        self.X = X.reshape(-1).astype(float); self.Y = Y.reshape(-1).astype(float)

    def parts(self, pyz):
        py, pz = pyz
        h0 = self.h_xy + math.sin(pz) * self.az
        Qz = (math.sin(2 * pz) / 2) * self.Is4
        Qy = self.Qy
        if self.Ly == 1:
            h0 = h0 + math.sin(py) * self.ay
            Qy = (math.sin(2 * py) / 2) * self.Is4
        return h0, (self.Qx, Qy, Qz)

    def blocks(self, nz):
        pys = pz_grid(self.ny) if self.Ly == 1 else [0.0]
        return [(py, pz) for py in pys for pz in pz_grid(nz)]

    @staticmethod
    def anti(F, A):
        """{F, A}/2 for a site field F (length ns) expanded to components."""
        Fd = torch.as_tensor(np.repeat(F, 4), dtype=CD)
        return 0.5 * (Fd[:, None] * A + A * Fd[None, :])

    def hm(self, pz, N, v):
        h0, Qs = self.parts(pz)
        H = self.anti(N, h0)
        for i in range(3):
            if np.any(v[i] != 0):
                H = H + self.anti(v[i], Qs[i])
        return H, h0, Qs


def site_dens(A, U):
    """per-site Re sum_cols U^+ (A U) (summed over the 4 components)."""
    d = torch.real((U.conj() * (A @ U)).sum(1))
    return d.reshape(-1, 4).sum(1)


def sea(H):
    e, U = torch.linalg.eigh(H)
    occ = e < 0
    return e, U, occ


def pz_grid(nz):
    return [(-PI + 2 * PI * (k + 0.5) / nz) for k in range(nz)]


def zone_discrete(Lx, Ly, nz, ny=1):
    """the same momentum grid as the finite lattice, for the ls_zone formulas: p_x = 2 pi j/Lx, p_y = 2 pi j/Ly, p_z midpoint."""
    px = torch.tensor([2 * PI * j / Lx for j in range(Lx)])
    py = torch.tensor([2 * PI * j / Ly for j in range(Ly)]) if Ly > 1 else torch.tensor(pz_grid(ny), dtype=torch.float64)
    pz = torch.tensor(pz_grid(nz), dtype=torch.float64)
    X, Y, Zz = torch.meshgrid(px, py, pz, indexing="ij")
    P = torch.stack([X.reshape(-1), Y.reshape(-1), Zz.reshape(-1)], 1)
    W = torch.full((P.shape[0],), 1.0 / P.shape[0])
    return P, W


def fourier(f, lat, K):
    ph = np.exp(-1j * (K[0] * lat.X + K[1] * lat.Y))
    return complex(np.sum(ph * f) / lat.ns)


# ------------------------------------------------------------------ mode two
def mode_two(lat, nz, q, vh, cN, A):
    out = {}
    def energy(amp):
        tot = 0.0
        arg = q[0] * lat.X + q[1] * lat.Y
        N = 1 + cN * amp * np.cos(arg)
        v = [vh[i] * amp * np.cos(arg) for i in range(3)]
        for pz in lat.blocks(nz):
            H, _, _ = lat.hm(pz, N, v)
            e = torch.linalg.eigvalsh(H)
            tot += float(e[e < 0].sum())
        return tot / (lat.ns * len(lat.blocks(nz)))
    E0, Ep, Em = energy(0.0), energy(A), energy(-A)
    exact = (Ep - 2 * E0 + Em) / A ** 2
    P, W = zone_discrete(lat.Lx, lat.Ly, nz, lat.ny)
    formula = 0.0
    n0, e0 = Z.nvec(P, lat.m)
    for sg in (1.0, -1.0):
        qq = sg * torch.tensor([q[0], q[1], 0.0], dtype=torch.float64)
        n1, e1 = Z.nvec(P + qq, lat.m)
        O = 0.5 * ((n1 - n0) ** 2).sum(1)
        mv = 0.5 * ((Z.Qs(P) + Z.Qs(P + qq)) * torch.tensor(vh, dtype=torch.float64)).sum(1)
        M = cN * 0.5 * (e1 - e0) + mv
        formula += 0.5 * float((-W * M * M * O / (e0 + e1)).sum())
    out.update({"q": q, "vh": vh, "cN": cN, "A": A, "E2_exact": exact, "E2_formula": formula,
                "rel": abs(exact - formula) / max(abs(formula), 1e-300)})
    return out


# ------------------------------------------------------------------ mode three
def mode_three(lat, nz, q1, q2, v1, v2, A):
    K = [q1[0] + q2[0], q1[1] + q2[1]]
    arg1 = q1[0] * lat.X + q1[1] * lat.Y; arg2 = q2[0] * lat.X + q2[1] * lat.Y
    rho = {}
    for s1 in (1, -1):
        for s2 in (1, -1):
            v = [s1 * A * v1[i] * np.cos(arg1) + s2 * A * v2[i] * np.cos(arg2) for i in range(3)]
            acc = np.zeros(lat.ns)
            for pz in lat.blocks(nz):
                H, h0, _ = lat.hm(pz, np.ones(lat.ns), v)
                e, U, occ = sea(H)
                acc += site_dens(h0, U[:, occ]).numpy()
            rho[(s1, s2)] = acc / len(lat.blocks(nz))
    cross = (rho[(1, 1)] - rho[(1, -1)] - rho[(-1, 1)] + rho[(-1, -1)]) / (4 * A * A)
    exact = 4 * fourier(cross, lat, K)         # cos waves: amplitudes A/2 at +-q each, so the K component carries 1/4
    P, W = zone_discrete(lat.Lx, lat.Ly, nz, lat.ny)
    formula = Z._three_point_chunk(P, W, lat.m, torch.tensor([q1[0], q1[1], 0.0]), torch.tensor([q2[0], q2[1], 0.0]),
                                   torch.tensor(v1, dtype=torch.float64), torch.tensor(v2, dtype=torch.float64))
    return {"q1": q1, "q2": q2, "v1": v1, "v2": v2, "A": A, "exact_re": exact.real, "exact_im": exact.imag, "formula": formula,
            "rel": abs(exact.real - formula) / max(abs(formula), 1e-300)}


# ------------------------------------------------------------------ adiabatic first-order densities
def rho1_and_J1(lat, nz, fields_at, dt):
    """rho1_x = Tr(C1 e_x), J1_x,i = Tr(C1 p_x,i) with C1_ab = i Pdot_ab/(eps_a - eps_b) (a, b in different bands) at t = 0,
    Pdot = (P(dt) - P(-dt))/(2 dt), P(t) the negative projector of h[u(t)]; fields_at(t) -> (N, v)."""
    rho = np.zeros(lat.ns); J = np.zeros((3, lat.ns))
    for pz in lat.blocks(nz):
        N0, v0 = fields_at(0.0)
        H, h0, Qs = lat.hm(pz, N0, v0)
        e, U, occ = sea(H)
        Pp = []
        for s in (1, -1):
            Ns, vs = fields_at(s * dt)
            Hs, _, _ = lat.hm(pz, Ns, vs)
            es, Us, occs = sea(Hs)
            Uo = Us[:, occs]
            Pp.append(Uo @ Uo.conj().T)
        Pdot = (Pp[0] - Pp[1]) / (2 * dt)
        Pd = U.conj().T @ Pdot @ U
        den = e[:, None] - e[None, :]
        inter = occ[:, None] ^ occ[None, :]
        C1 = torch.zeros_like(Pd)
        C1[inter] = 1j * Pd[inter] / den[inter]
        C1x = U @ C1 @ U.conj().T                       # in the site basis
        # Tr(C1 {Pi_x, A}/2) = Re sum over the site's components of (A C1)_xx (C1 Hermitian)
        def dens(A):
            M = A @ C1x
            return torch.real(torch.diagonal(M)).reshape(-1, 4).sum(1).numpy()
        rho += dens(h0)
        for i in range(3):
            J[i] += dens(Qs[i])
    nb = len(lat.blocks(nz))
    return rho / nb, J / nb


def mode_rate1(lat, nz, K, vh, B, dt):
    arg = K[0] * lat.X + K[1] * lat.Y
    def fields_at(t):
        return np.ones(lat.ns), [vh[i] * B * t * np.cos(arg) for i in range(3)]
    rho, J = rho1_and_J1(lat, nz, fields_at, dt)
    # v = B t cos(Kx) vh: d_t v amplitude B/2 at +-K; rho1(K) = i R(K) * (B/2) per the formula's definition
    exact = fourier(rho, lat, K) / (B / 2)
    P, W = zone_discrete(lat.Lx, lat.Ly, nz, lat.ny)
    Kt = torch.tensor([K[0], K[1], 0.0]); vt = torch.tensor(vh, dtype=torch.float64)
    n0, e0 = Z.nvec(P, lat.m); n1, e1 = Z.nvec(P + Kt, lat.m)
    O = 0.5 * ((n1 - n0) ** 2).sum(1)
    mv = 0.5 * ((Z.Qs(P) + Z.Qs(P + Kt)) * vt).sum(1)
    R = float((W * (e0 - e1) * O * mv / (e0 + e1) ** 2).sum())
    # the current's response to d_t N on the same wave
    def fields_N(t):
        return 1 + B * t * np.cos(arg), [np.zeros(lat.ns)] * 3
    rhoN, JN = rho1_and_J1(lat, nz, fields_N, dt)
    exJ = [fourier(JN[i], lat, K) / (B / 2) for i in range(3)]
    return {"K": K, "vh": vh, "rho1_K_exact": [exact.real, exact.imag], "i_R_formula": [0.0, R],
            "rel_im": abs(exact.imag - R) / max(abs(R), 1e-300),
            "J1_K_exact_from_Ndot": [[x.real, x.imag] for x in exJ], "rho_from_Ndot_K": [fourier(rhoN, lat, K).real, fourier(rhoN, lat, K).imag]}


def harmonics(f, lat, q):
    """(mean, cos 2qx, sin 2qx, cos qx, sin qx) components of a site function on a 1D modulation along x."""
    X = lat.X
    return {"h0": float(np.mean(f)), "c2": float(2 * np.mean(f * np.cos(2 * q * X))), "s2": float(2 * np.mean(f * np.sin(2 * q * X))),
            "c1": float(2 * np.mean(f * np.cos(q * X))), "s1": float(2 * np.mean(f * np.sin(q * X)))}


def mode_rate2(lat, nz, q, A, B, dt):
    """second order in the amplitude A, first in the rate B, 1D modulation along x (Lx sites, Ly = 1 allowed)."""
    X = lat.X
    c, s = np.cos(q * X), np.sin(q * X)
    one = np.ones(lat.ns); zero = np.zeros(lat.ns)
    res = {}
    def run(fields_of_sign):
        r = {}
        for sg in (1, -1):
            rho, J = rho1_and_J1(lat, nz, fields_of_sign(sg), dt)
            r[sg] = (rho, J)
        rho = (r[1][0] - r[-1][0]) / (2 * A * B); J = (r[1][1] - r[-1][1]) / (2 * A * B)
        return rho, J
    # (n, udot): N = 1 + sg A cos qx, v_x = B t sin qx
    rho, J = run(lambda sg: (lambda t: (one + sg * A * c, [B * t * s, zero, zero])))
    H = harmonics(rho, lat, q)
    # n' udot = -q sin * sin = -q (1 - cos2)/2 ; n udot' = cos * q cos = q (1 + cos2)/2
    M = np.array([[-q / 2, q / 2], [q / 2, q / 2]])           # rows: h0, c2 ; cols: alpha (n' udot), gamma (n udot')
    al_ga = np.linalg.solve(M, [H["h0"], H["c2"]])
    res["n_udot"] = {"harm": H, "coef_nprime_udot": al_ga[0], "coef_n_udotprime": al_ga[1]}
    # (u, ndot): v_x = sg A sin qx, N = 1 + B t cos qx: ndot u' = cos * q cos = q(1+cos2)/2 ; u ndot' = sin * (-q sin) = -q(1-cos2)/2
    rho, J = run(lambda sg: (lambda t: (one + B * t * c, [sg * A * s, zero, zero])))
    H = harmonics(rho, lat, q)
    M = np.array([[q / 2, -q / 2], [q / 2, q / 2]])           # cols: beta (ndot u'), delta (u ndot')
    be_de = np.linalg.solve(M, [H["h0"], H["c2"]])
    res["u_ndot"] = {"harm": H, "coef_ndot_uprime": be_de[0], "coef_u_ndotprime": be_de[1]}
    # current J1_x at second order: (n, ndot): N = 1 + sg A cos qx + B t sin qx: forms ndot n' = sin * (-q sin) = -q(1-cos2)/2,
    # n ndot' = cos * q cos = q(1+cos2)/2  (the second only from expanding a linear term, whose coefficient is zero)
    rho, J = run(lambda sg: (lambda t: (one + sg * A * c + B * t * s, [zero, zero, zero])))
    H = harmonics(J[0], lat, q)
    M = np.array([[-q / 2, q / 2], [q / 2, q / 2]])
    x = np.linalg.solve(M, [H["h0"], H["c2"]])
    res["J_n_ndot"] = {"harm": H, "coef_ndot_nprime": x[0], "coef_n_ndotprime": x[1]}
    # (u, udot): v_x = sg A sin qx + B t cos qx: udot u' = cos * q cos = q(1+cos2)/2 ; u udot' = sin * (-q sin) = -q(1-cos2)/2
    rho, J = run(lambda sg: (lambda t: (one, [sg * A * s + B * t * c, zero, zero])))
    H = harmonics(J[0], lat, q)
    M = np.array([[q / 2, -q / 2], [q / 2, q / 2]])
    x = np.linalg.solve(M, [H["h0"], H["c2"]])
    res["J_u_udot"] = {"harm": H, "coef_udot_uprime": x[0], "coef_u_udotprime": x[1]}
    return res


def mode_evolve(lat, nz, K, vh, B0, tau, T, dt):
    """exact unitary evolution of the sea (all blocks) under v(x, t) = B0 f(t) cos(K x) vh, f(t) = t - tau (1 - exp(-t/tau))
    (rate switched on smoothly, then constant B0): the density's departure from the instantaneous sea, Tr((C - P(t)) e_x), at the
    transfer K, against the adiabatic first-order prediction i R(K) B0 / 2 (rate_linear's integrand on this momentum grid)."""
    arg = K[0] * lat.X + K[1] * lat.Y
    nsteps = int(round(T / dt))
    rec_t = np.linspace(0.5 * T, T, 6)
    rec_i = set(int(round(t / dt)) for t in rec_t)
    dev = {i: np.zeros(lat.ns) for i in rec_i}
    f = lambda t: t - tau * (1 - math.exp(-t / tau))
    for pyz in lat.blocks(nz):
        H0, h0, _ = lat.hm(pyz, np.ones(lat.ns), [np.zeros(lat.ns)] * 3)
        e, U = torch.linalg.eigh(H0)
        Phi = U[:, e < 0]
        for i in range(1, nsteps + 1):
            tm = (i - 0.5) * dt
            Hm, _, _ = lat.hm(pyz, np.ones(lat.ns), [vh[c] * B0 * f(tm) * np.cos(arg) for c in range(3)])
            ee, UU = torch.linalg.eigh(Hm)
            Phi = UU @ (torch.exp(-1j * ee * dt)[:, None] * (UU.conj().T @ Phi))
            if i in rec_i:
                Ht, _, _ = lat.hm(pyz, np.ones(lat.ns), [vh[c] * B0 * f(i * dt) * np.cos(arg) for c in range(3)])
                et, Ut = torch.linalg.eigh(Ht)
                dev[i] += site_dens(h0, Phi).numpy() - site_dens(h0, Ut[:, et < 0]).numpy()
    nb = len(lat.blocks(nz))
    P, W = zone_discrete(lat.Lx, lat.Ly, nz, lat.ny)
    Kt = torch.tensor([K[0], K[1], 0.0]); vt = torch.tensor(vh, dtype=torch.float64)
    n0, e0 = Z.nvec(P, lat.m); n1, e1 = Z.nvec(P + Kt, lat.m)
    O = 0.5 * ((n1 - n0) ** 2).sum(1)
    mv = 0.5 * ((Z.Qs(P) + Z.Qs(P + Kt)) * vt).sum(1)
    R = float((W * (e0 - e1) * O * mv / (e0 + e1) ** 2).sum())
    out = []
    for i in sorted(rec_i):
        fk = fourier(dev[i] / nb, lat, K)
        rate = 1 - math.exp(-i * dt / tau)
        out.append({"t": i * dt, "dev_K": [fk.real, fk.imag], "pred_iR_B0rate_over2": [0.0, R * B0 * rate / 2],
                    "ratio_im": fk.imag / (R * B0 * rate / 2) if R != 0 else None})
    return {"K": K, "vh": vh, "B0": B0, "tau": tau, "T": T, "dt": dt, "R": R, "records": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["two", "three", "rate1", "rate2", "evolve"])
    ap.add_argument("--Lx", type=int, default=12)
    ap.add_argument("--Ly", type=int, default=12)
    ap.add_argument("--nz", type=int, default=8)
    ap.add_argument("--ny", type=int, default=8)
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--A", type=float, default=1e-3)
    ap.add_argument("--B", type=float, default=1e-3)
    ap.add_argument("--dt", type=float, default=1e-2)
    ap.add_argument("--j", default="1")
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--T", type=float, default=80.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", default="8")
    a = ap.parse_args()
    lat = Lat(a.Lx, a.Ly, a.m, a.ny)
    t0 = time.time()
    res = {"mode": a.mode, "Lx": a.Lx, "Ly": a.Ly, "nz": a.nz, "ny": a.ny, "m": a.m, "A": a.A, "B": a.B, "dt": a.dt, "script": "ls_exact_v1.py"}
    gx, gy = 2 * PI / a.Lx, 2 * PI / a.Ly
    if a.mode == "two":
        r = []
        for (q, vh, cN) in (([gx, 0], [1, 0, 0], 0.0), ([gx, 0], [0, 1, 0], 0.0), ([gx, 0], [0, 0, 1], 0.0), ([gx, gy], [0.6, -0.8, 0], 0.0),
                            ([gx, gy], [0.3, 0.4, 0.866], 0.0), ([gx, 0], [0, 0, 0], 1.0), ([gx, gy], [0.6, 0.0, 0.8], 0.5)):
            r.append(mode_two(lat, a.nz, q, vh, cN, a.A))
            print(json.dumps(r[-1]), "%.1fs" % (time.time() - t0), flush=True)
        res["two"] = r
    elif a.mode == "three":
        rng = np.random.default_rng(11)
        r = []
        for (q1, q2) in (([gx, 0], [0, gy]), ([gx, gy], [gx, -gy]), ([2 * gx, 0], [gx, gy]), ([gx, 0], [gx, gy])):
            v1 = rng.normal(size=3).tolist(); v2 = rng.normal(size=3).tolist()
            r.append(mode_three(lat, a.nz, q1, q2, v1, v2, a.A))
            print(json.dumps(r[-1]), "%.1fs" % (time.time() - t0), flush=True)
        res["three"] = r
    elif a.mode == "rate1":
        r = []
        for (K, vh) in (([gx, 0], [1, 0, 0]), ([gx, gy], [0.6, 0.8, 0]), ([gx, 0], [0, 1, 0])):
            r.append(mode_rate1(lat, a.nz, K, vh, a.B, a.dt))
            print(json.dumps(r[-1]), "%.1fs" % (time.time() - t0), flush=True)
        res["rate1"] = r
    elif a.mode == "rate2":
        js = [int(x) for x in a.j.split(",")]
        r = {}
        for j in js:
            q = 2 * PI * j / a.Lx
            r[str(j)] = {"q": q, **mode_rate2(lat, a.nz, q, a.A, a.B, a.dt)}
            print(j, json.dumps(r[str(j)]), "%.1fs" % (time.time() - t0), flush=True)
        res["rate2"] = r
    elif a.mode == "evolve":
        r = []
        for (K, vh) in (([gx, 0], [1, 0, 0]), ([gx, 0], [0, 1, 0])):
            r.append(mode_evolve(lat, a.nz, K, vh, a.B, a.tau, a.T, a.dt))
            print(json.dumps(r[-1]), "%.1fs" % (time.time() - t0), flush=True)
        res["evolve"] = r
    res["seconds"] = time.time() - t0
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "ls_exact_v1_%s.json" % a.tag), "w") as f:
        json.dump(res, f, indent=1)
    print("DONE %.1fs" % res["seconds"])


if __name__ == "__main__":
    main()
