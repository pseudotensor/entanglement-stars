#!/usr/bin/env python3
"""obs_plateau_v2.py (builder analyst; job J-PL of SPEC_final4b_observables.md sec. 9.1; version 2 of obs_plateau_v1.py).

Version 2: (1) profile pgs, the exact Painleve-Gullstrand exterior with an ANALYTIC continuation inside: v = -sqrt(r_s/rho(R)),
rho(R) = R_1 + w log(1 + e^((R - R_1)/w)) (rho = R up to w e^(-(R - R_1)/w) outside, rho -> R_1 inside: the inner lead at v = -sqrt(r_s/R_1));
the surface rho(R_h) = r_s and kappa = rho'(R_h)/(2 r_s) are computed from the continued profile, and printed against 1/(2 r_s).  Why (builder
finding, labelled): version 1's pg profile blended to the inner lead with the C-infinity step e^(-1/t), which is not analytic; its Fourier
content at the lattice scale (Delta k = pi/2 between the two outward interior branches at k = +-pi/4, the empty one alpha and the occupied
one beta) mixed them at the 1e-2 level, so theta moved by 4 to 10 percent with R_in (0.96 at R_in = r_s/2, 1.05 at r_s/4, kappa a = 1/128,
r = 1/4) and Gamma by 1e-3: the inner boundary condition, as the spec's sentence anticipated.  The softplus is analytic in the strip
|Im R| < pi w, so the mixing is of order e^(-(pi/2)(pi w)) (1e-7 at w = 4), while the horizon's Taylor coefficients move by e^(-(r_s - R_1)/w);
both small needs r_s - R_1 >= 14 w with w >= 4: r_s >= 64 (kappa a <= 1/128).  At r_s = 16 and 32 no continuation satisfies both (the
interior is r_s sites long), and their rows carry the continuation ladder's spread as their error.  (2) The optional smooth outer saturation
(--outer-w, units of R_lead): rho -> R_lead - w_L log(1 + e^((R_lead - rho)/w_L)) beyond the lead point, so the outer lead joins without a kink
(default off: the kink and the R_lead ladder, as version 1).  (3) The summary pairs sizes by the nominal geometry class (version 1 keyed on
the realized R_lead, which differs per size, so no Richardson value was formed).  (4) The greybody table: Gamma per (kappa a, R_lead) and its
(kappa a)^2 extrapolation from the two smallest kappa a.
Version 1's docstring follows.

obs_plateau_v1.py (builder analyst; job J-PL of SPEC_final4b_observables.md sec. 9.1, the plateau's (kappa a)^2 correction by stationary
scattering on the 1 + 1 chain; also the greybody code checks of the same section).
The operator: the model's chain with the lapse hook, in obs_raymap_pg_v2.chain_hamiltonian_N's convention (on-site N_n (m sz + y sy), the
sine hop <n|h|n+1> = -(i/2)(N_n + N_{n+1})/2 sx, the range-two flow hop <n|h|n+2> = -(i/4)(v_n + v_{n+2})/2 1; symbol
N (m sz + y sy + sin k sx) + v sin k cos k; y = 0 here: the axial channel, P25's 1-dimensional problem). At a fixed Killing frequency omega
the chain is solved between two uniform semi-infinite leads (BUILDER: the wave-function matching method; the site-level modes of each lead
from the degree-4 matrix polynomial, so the doublers k and k + pi stay distinct channels; the leads' Bloch matrices on two-site cells give the
self-energies; one banded LU per omega, one right-hand side per incoming channel; the flux normalization by each mode's group velocity, so
the scattering matrix is unitary and its columns are checked to 1e-10). Channels are labelled by side, site momentum k, group velocity and
the free-fall frequency omega_ff = omega - v sin k cos k (the eigenvalue of the free part; omega_ff < 0 is a mode of the transported sea,
occupied in the in-state: the goal's transported incoming sea, never the instantaneous bands).
The Hawking channel H: the outgoing right-mover on the outer lead with |k| < pi/2. Its decomposition over the incoming channels:
|beta|^2 = sum over the occupied inner in-channels, |alpha|^2 = sum over the empty inner in-channels, R = sum over the outer in-channels
(the reflection); Gamma = |alpha|^2 + |beta|^2 = 1 - R (the transmission, the greybody factor of the code checks); the conversion
n = |beta|^2/Gamma and T_conv = omega/ln(1/n - 1) = omega/ln(|alpha|^2/|beta|^2); T_n = omega/ln(1/n_H - 1) from the full occupation
n_H = sum over all occupied in-channels (greybody-biased; printed beside). T_H = kappa/(2 pi), kappa = d(N + v)/dR at the surface N + v = 0.
Profiles: tanh (P25's v = -(5/4)[1 - tanh(5 z/6 + artanh(1/5))], z = kappa (x - x_h), the code's control: 0.99443 at kappa a = 1/8 and
0.99965 at 1/32, r = 1/4, m = 0; P25's exact lattice value 0.99965721953 at 1/32); pg (v = -sqrt(r_s/R), N = 1, kappa = 1/(2 r_s); the outer
lead at R_lead with the local v(R_lead), the lead ladder R_lead in {32, 64, 128, 256} r_s; BUILDER, labelled: the inner lead at R_in < r_s, the
profile blended to the constant v(R_in) over [R_in, R_in + W_in] by the C-infinity step, the R_in ladder printed). The solved exterior
(profile (i)) is version 2's, when the collapse's fields exist.
Output: per (profile, size, lead, r, mu): theta = T_conv/T_H, theta_n, Gamma, n, the unitarity error, the channel table; per profile the
per-size c = (1 - theta)/(kappa a)^2, the two-size Richardson c, and the fit c = c_0 + c_2 r^2 + c_m mu^2 with its errors (the spec's sign:
T/T_H - 1 = -(kappa a)^2 (c_0 + c_2 r^2 + c_m mu^2); P25: tanh (1/3, 5/18, 5/36), Painleve-Gullstrand (-5/8, -1, 3/4)).
Args: --profile (tanh|pg) --kappa-a (list) --r (list) --mu (list) --R-lead (list, units of r_s) --R-in (list, units of r_s) --W-in (units of
r_s) --z-span (tanh) --out-json --md --selftest."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
import json
import math
import time

import numpy as np
from scipy.linalg import eig, solve_banded
from scipy.optimize import brentq

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.array([[1, 0], [0, -1]], complex)
I2 = np.eye(2, dtype=complex)
Z2 = np.zeros((2, 2), complex)
BW = 5                                                     # half bandwidth of the site-component matrix (range-two hop: offset up to 5)


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    f = lambda s: np.where(s > 0, np.exp(-1.0 / np.maximum(s, 1e-300)), 0.0)
    a, b = f(t), f(1.0 - t)
    return a / (a + b)


# ---------------------------------------------------------------- the chain, banded
def band_matrix(v, N, m, y):
    """the chain's matrix in LAPACK band storage ab[BW + i - j, j] = h[i, j], dimension 2n."""
    n = len(v)
    dim = 2 * n
    ab = np.zeros((2 * BW + 1, dim), complex)

    def put(r, c, val):
        np.add.at(ab, (BW + r - c, c), val)
    idx = np.arange(n)
    on = N[:, None, None] * (m * SZ + y * SY)[None]
    for a_ in range(2):
        for b_ in range(2):
            put(2 * idx + a_, 2 * idx + b_, on[:, a_, b_])
    i1 = np.arange(n - 1)
    B = -0.5j * 0.5 * (N[:-1] + N[1:])
    for a_ in range(2):
        for b_ in range(2):
            if SX[a_, b_] != 0:
                put(2 * i1 + a_, 2 * i1 + 2 + b_, B * SX[a_, b_])
                put(2 * i1 + 2 + b_, 2 * i1 + a_, np.conj(B * SX[a_, b_]))
    i2 = np.arange(n - 2)
    C = -0.25j * 0.5 * (v[:-2] + v[2:])
    for a_ in range(2):
        put(2 * i2 + a_, 2 * i2 + 4 + a_, C)
        put(2 * i2 + 4 + a_, 2 * i2 + a_, np.conj(C))
    return ab


def add_block(ab, r0, M):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ab[BW + (r0 + i) - (r0 + j), r0 + j] += M[i, j]


# ---------------------------------------------------------------- a uniform lead
def lead_cells(v, N, m, y):
    on = N * (m * SZ + y * SY)
    S = -0.5j * N * SX
    F = -0.25j * v * I2
    H00 = np.block([[on, S], [S.conj().T, on]])
    H01 = np.block([[F, Z2], [S, F]])                     # <cell j| h |cell j+1>: site 0 -> 0 (flow), site 1 -> 0 (sine), 1 -> 1 (flow)
    return H00, H01


def site_modes(v, N, m, y, omega, tol=1e-7):
    """the 8 site-level modes psi_n = lam^n u of the uniform chain at omega: lam, u (|u| = 1), propagating flag, k, group velocity, omega_ff."""
    h0 = N * (m * SZ + y * SY)
    h1 = -0.5j * N * SX
    h2 = -0.25j * v * I2
    A = np.zeros((8, 8), complex)
    Bm = np.eye(8, dtype=complex)
    A[0:2, 2:4] = I2; A[2:4, 4:6] = I2; A[4:6, 6:8] = I2
    A[6:8, 0:2] = -h2.conj().T; A[6:8, 2:4] = -h1.conj().T; A[6:8, 4:6] = -(h0 - omega * I2); A[6:8, 6:8] = -h1
    Bm[6:8, 6:8] = h2
    w, vr = eig(A, Bm, homogeneous_eigvals=True)
    al, be = w
    out = []
    for j in range(8):
        if abs(be[j]) <= 1e-300 or abs(al[j]) / max(abs(be[j]), 1e-300) > 1e12:
            raise RuntimeError("infinite lead eigenvalue: v = %g too small for the site-level lead" % v)
        lam = al[j] / be[j]
        blocks = [vr[2 * q:2 * q + 2, j] for q in range(4)]
        q = int(np.argmax([np.linalg.norm(b) for b in blocks]))
        u = blocks[q] / lam ** q
        u = u / np.linalg.norm(u)
        P = h2.conj().T / lam ** 2 + h1.conj().T / lam + (h0 - omega * I2) + h1 * lam + h2 * lam ** 2
        res = float(np.linalg.norm(P @ u)) / max(1.0, abs(lam) ** 2, abs(lam) ** -2)
        prop = abs(abs(lam) - 1.0) < tol
        md = {"lam": lam, "u": u, "abs": abs(lam), "prop": prop, "res": res}
        if prop:
            k = float(np.angle(lam))
            dh = N * math.cos(k) * SX + v * math.cos(2 * k) * I2
            md["k"] = k
            md["vg"] = float(np.real(u.conj() @ dh @ u))
            md["wff"] = float(np.real(u.conj() @ (N * (m * SZ + y * SY + math.sin(k) * SX)) @ u))
        out.append(md)
    return out


def lead_setup(v, N, m, y, omega, side):
    """side 'R': the lead extends to +infinity; the retained set '+' (outgoing right-movers and right-decaying modes), the in-channels the
    propagating left-movers.  side 'L': mirror.  Returns the Bloch data, the self-energy and the mode lists."""
    modes = site_modes(v, N, m, y, omega)
    amb = [md for md in modes if 1e-9 < abs(md["abs"] - 1) < 1e-4]
    if side == "R":
        keep = [md for md in modes if (md["prop"] and md["vg"] > 0) or (not md["prop"] and md["abs"] < 1)]
        inc = [md for md in modes if md["prop"] and md["vg"] < 0]
    else:
        keep = [md for md in modes if (md["prop"] and md["vg"] < 0) or (not md["prop"] and md["abs"] > 1)]
        inc = [md for md in modes if md["prop"] and md["vg"] > 0]
    if len(keep) != 4:
        raise RuntimeError("lead %s at omega %g: %d retained modes (expected 4)" % (side, omega, len(keep)))
    U = np.array([np.concatenate([md["u"], md["lam"] * md["u"]]) for md in keep]).T
    lam2 = np.array([md["lam"] ** 2 for md in keep])
    H00, H01 = lead_cells(v, N, m, y)
    Ui = np.linalg.inv(U)
    if side == "R":
        Fp = U @ np.diag(lam2) @ Ui
        Sig = H01 @ Fp
        T = Fp
    else:
        Fmi = U @ np.diag(1.0 / lam2) @ Ui
        Sig = H01.conj().T @ Fmi
        T = Fmi
    return {"keep": keep, "inc": inc, "U": U, "Ui": Ui, "T": T, "Sig": Sig, "H01": H01, "cond": float(np.linalg.cond(U)),
            "ambiguous": len(amb), "res_max": max(md["res"] for md in modes)}


def scatter(v, N, m, y, omega):
    """the scattering matrix's Hawking row and the full unitarity check on the chain (v, N) whose first and last 4 sites are the leads'."""
    n = len(v)
    LL = lead_setup(v[0], N[0], m, y, omega, "L")
    LR = lead_setup(v[-1], N[-1], m, y, omega, "R")
    ab = band_matrix(v, N, m, y)
    ab[BW, :] -= omega
    add_block(ab, 0, LL["Sig"])
    add_block(ab, 2 * n - 4, LR["Sig"])
    chans = [("L", md) for md in LL["inc"]] + [("R", md) for md in LR["inc"]]
    rhs = np.zeros((2 * n, len(chans)), complex)
    for c, (side, md) in enumerate(chans):
        Ub = np.concatenate([md["u"], md["lam"] * md["u"]])
        if side == "L":
            rhs[0:4, c] = -(LL["H01"].conj().T @ (Ub / md["lam"] ** 2 - LL["T"] @ Ub))
        else:
            rhs[2 * n - 4:2 * n, c] = -(LR["H01"] @ (md["lam"] ** 2 * Ub - LR["T"] @ Ub))
    psi = solve_banded((BW, BW), ab, rhs, check_finite=False)
    outs = [("L", i, md) for i, md in enumerate(LL["keep"]) if md["prop"]] + [("R", i, md) for i, md in enumerate(LR["keep"]) if md["prop"]]
    S = np.zeros((len(outs), len(chans)), complex)
    for c, (side, md) in enumerate(chans):
        Ub = np.concatenate([md["u"], md["lam"] * md["u"]])
        cL = LL["Ui"] @ (psi[0:4, c] - (Ub if side == "L" else 0))
        cR = LR["Ui"] @ (psi[2 * n - 4:2 * n, c] - (Ub if side == "R" else 0))
        for o, (oside, i, mo) in enumerate(outs):
            amp = cL[i] if oside == "L" else cR[i]
            S[o, c] = amp * math.sqrt(abs(mo["vg"]) / abs(md["vg"]))
    unit = float(np.max(np.abs(np.sum(np.abs(S) ** 2, axis=0) - 1.0))) if S.size else None
    return {"S": S, "outs": outs, "chans": chans, "unitarity": unit, "cond": (LL["cond"], LR["cond"]), "ambiguous": LL["ambiguous"] + LR["ambiguous"],
            "res_max": max(LL["res_max"], LR["res_max"])}


def hawking_reading(sc, omega, TH):
    """the Hawking row: the outer lead's outgoing right-mover with |k| < pi/2."""
    Hs = [o for o, (side, i, mo) in enumerate(sc["outs"]) if side == "R" and abs(mo["k"]) < math.pi / 2]
    if len(Hs) != 1:
        return {"error": "%d long-wave outgoing channels on the outer lead" % len(Hs)}
    h = Hs[0]
    row = np.abs(sc["S"][h]) ** 2
    beta2 = alpha2 = refl = nH = 0.0
    table = []
    for c, (side, md) in enumerate(sc["chans"]):
        occ = md["wff"] < 0
        if side == "L":
            if occ:
                beta2 += row[c]
            else:
                alpha2 += row[c]
        else:
            refl += row[c]
        if occ:
            nH += row[c]
        table.append({"side": side, "k": md["k"], "vg": md["vg"], "wff": md["wff"], "occupied": bool(occ), "weight": float(row[c])})
    Gam = alpha2 + beta2
    out = {"alpha2": alpha2, "beta2": beta2, "R": refl, "Gamma": Gam, "n_H": nH, "channels": table, "k_H": sc["outs"][h][2]["k"]}
    if beta2 > 0 and alpha2 > beta2:
        Tc = omega / math.log(alpha2 / beta2)
        out.update({"n": beta2 / Gam, "T_conv": Tc, "theta": Tc / TH})
    if 0 < nH < 0.5:
        Tn = omega / math.log(1.0 / nH - 1.0)
        out.update({"T_n": Tn, "theta_n": Tn / TH})
    return out


# ---------------------------------------------------------------- profiles
def profile_tanh(kappa, z_span, v_out_min):
    """P25's tanh profile on sites x_n = n, z = kappa (x - x_h) in [-z_span, z_out], z_out where |v| falls to v_out_min (the site-level
    outer lead needs v != 0; |v| = (5/2)/(1 + e^(2u)) with u = 5 z/6 + artanh(1/5)); the first and last 4 sites set to the end values."""
    z_out = (6.0 / 5.0) * (0.5 * math.log(2.5 / v_out_min - 1.0) - math.atanh(0.2))
    nsites = int(math.ceil((z_span + z_out) / kappa)) + 8
    nsites += nsites % 2
    x = np.arange(nsites, dtype=float)
    xh = z_span / kappa + 4
    z = kappa * (x - xh)
    v = -1.25 * (1.0 - np.tanh(5.0 * z / 6.0 + math.atanh(0.2)))
    v[:4] = v[4]; v[-4:] = v[-5]
    return x, v, np.ones_like(v), {"x_h": xh, "kappa": kappa, "z_out": float(z[-1]), "v_in": float(v[0]), "v_out": float(v[-1]), "nsites": nsites}


def profile_pg(rs, R_lead, R_in, W_in, R_off=0.0):
    """v = -sqrt(r_s/R) on sites R_n (integers + R_off) from R_in - 8 to R_lead; inside R_in blended to the constant v(R_in) over
    [R_in, R_in + W_in]; the first and last 4 sites the leads' values."""
    R0 = math.floor(R_in) - 8 + R_off
    nsites = int(math.ceil(R_lead - R0)) + 1
    nsites += nsites % 2
    R = R0 + np.arange(nsites, dtype=float)
    vpg = -np.sqrt(rs / np.maximum(R, 1e-12))
    v_in = -math.sqrt(rs / R_in)
    s = smoothstep((R - R_in) / max(W_in, 1e-12))
    v = v_in + (vpg - v_in) * s
    v[:4] = v[4]; v[-4:] = v[-5]
    return R, v, np.ones_like(v), {"r_s": rs, "kappa": 1.0 / (2 * rs), "R_in": R_in, "W_in": W_in, "R_lead": float(R[-1]), "R_lead_nominal": R_lead / rs, "v_in": float(v[0]),
                                    "v_out": float(v[-1]), "nsites": nsites}


def profile_pgs(rs, R_lead, R1, w, outer_w=None):
    """version 2: v = -sqrt(r_s/rho(R)), rho = R_1 + w softplus((R - R_1)/w) (analytic; the inner lead at -sqrt(r_s/R_1)); optionally the
    outer saturation rho -> R_lead - w_L softplus((R_lead - rho)/w_L) with w_L = outer_w R_lead (then the chain runs to R_lead + 40 w_L)."""
    R0 = R1 - 40.0 * w
    wL = outer_w * R_lead if outer_w else None
    Rend = R_lead + (40.0 * wL if wL else 0.0)
    nsites = int(math.ceil(Rend - R0)) + 1
    nsites += nsites % 2
    R = R0 + np.arange(nsites, dtype=float)

    def rho_of(x):
        r_ = R1 + w * np.logaddexp(0.0, (x - R1) / w)
        if wL:
            r_ = R_lead - wL * np.logaddexp(0.0, (R_lead - r_) / wL)
        return r_
    v = -np.sqrt(rs / rho_of(R))
    v[:4] = v[4]; v[-4:] = v[-5]
    Rh = brentq(lambda x: float(rho_of(np.array([x]))[0]) - rs, R1, rs + 50 * w + 1.0, xtol=1e-14)
    hR = 1e-4
    drho = float((rho_of(np.array([Rh + hR])) - rho_of(np.array([Rh - hR])))[0]) / (2 * hR)
    kappa = drho / (2.0 * rs)
    return R, v, np.ones_like(v), {"r_s": rs, "kappa": kappa, "kappa_2rs": kappa * 2 * rs, "R_h": Rh, "R1": R1, "w": w, "outer_w": outer_w,
                                    "R_lead": R_lead, "R_lead_nominal": R_lead / rs, "R_in": R1, "v_in": float(v[0]), "v_out": float(v[-1]), "nsites": nsites}


# ---------------------------------------------------------------- driver
def run_point(v, N, kappa, r, mu):
    omega, m = r * kappa, mu * kappa
    TH = kappa / (2 * math.pi)
    t0 = time.time()
    sc = scatter(v, N, m, 0.0, omega)
    rd = hawking_reading(sc, omega, TH)
    rd.update({"r": r, "mu": mu, "omega": omega, "m": m, "unitarity": sc["unitarity"], "lead_cond": sc["cond"], "ambiguous_modes": sc["ambiguous"],
               "mode_residual_max": sc["res_max"], "n_in": len(sc["chans"]), "n_out": len(sc["outs"]), "seconds": time.time() - t0})
    return rd


def selftest():
    rows = []
    for vv, mm in ((-0.5, 0.0), (-0.3, 0.05), (-1.6, 0.0), (-2.0, 0.05)):
        v = np.full(64, vv); N = np.ones(64)
        sc = scatter(v, N, mm, 0.0, 0.08)
        S = sc["S"]
        # a uniform chain: every incoming channel leaves on its own branch with unit modulus
        rows.append("uniform v %g m %g: in %d out %d unitarity %.2e max|S| %.6f lead residual %.1e" % (vv, mm, len(sc["chans"]), len(sc["outs"]),
                    sc["unitarity"], float(np.max(np.abs(S))), sc["res_max"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["tanh", "pg", "pgs"], default="tanh")
    ap.add_argument("--cont", nargs="+", default=["0.125:4"], help="pgs: inner continuations R_1/r_s:w (w in sites)")
    ap.add_argument("--outer-w", type=float, default=None, help="pgs: the smooth outer saturation width, units of R_lead")
    ap.add_argument("--kappa-a", type=float, nargs="+", default=[0.125, 0.03125])
    ap.add_argument("--r", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--mu", type=float, nargs="+", default=[0.0, 0.2, 0.4])
    ap.add_argument("--R-lead", type=float, nargs="+", default=[32.0, 64.0, 128.0, 256.0], help="units of r_s (pg)")
    ap.add_argument("--R-in", type=float, nargs="+", default=[0.5, 0.25], help="units of r_s (pg)")
    ap.add_argument("--W-in", type=float, default=0.25, help="units of r_s (pg)")
    ap.add_argument("--z-span", type=float, default=17.0, help="tanh: the inner z range")
    ap.add_argument("--v-out-min", type=float, nargs="+", default=[1e-6], help="tanh: |v| at the outer lead (the ladder is the truncation control)")
    ap.add_argument("--extra", default="", help="extra (r,mu) points, e.g. 0.6:0.4")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    if a.selftest:
        for line in selftest():
            print(line, flush=True)
        return
    pts = [(r, mu) for mu in a.mu for r in a.r if r > mu]
    if a.extra:
        pts += [tuple(float(q) for q in p.split(":")) for p in a.extra.split(",")]
    res = []
    for ka in a.kappa_a:
        if a.profile == "tanh":
            geoms = [profile_tanh(ka, a.z_span, vo) for vo in a.v_out_min]
        elif a.profile == "pg":
            rs = 1.0 / (2 * ka)
            geoms = [profile_pg(rs, RL * rs, Ri * rs, a.W_in * rs) for RL in a.R_lead for Ri in a.R_in]
        else:
            rs = 1.0 / (2 * ka)
            geoms = [profile_pgs(rs, RL * rs, float(c.split(":")[0]) * rs, float(c.split(":")[1]), a.outer_w) for RL in a.R_lead for c in a.cont]
        for (x, v, N, info) in geoms:
            for (r, mu) in pts:
                try:
                    rd = run_point(v, N, info["kappa"], r, mu)
                except RuntimeError as e:
                    rd = {"r": r, "mu": mu, "error": str(e)}
                rd.update({"kappa_a": ka, "geometry": info})
                res.append(rd)
                print(json.dumps({k: rd.get(k) for k in ("kappa_a", "r", "mu", "theta", "theta_n", "Gamma", "n", "unitarity", "error")}),
                      "class %s" % (class_key(rd),), "kappa 2 r_s %s" % info.get("kappa_2rs", "-"), flush=True)
                if a.out_json:
                    json.dump({"script": "obs_plateau_v2.py", "args": vars(a), "rows": res}, open(a.out_json, "w"), indent=1, default=float)
    summ = summarize(res, a)
    gb = greybody_table(res)
    if a.out_json:
        json.dump({"script": "obs_plateau_v2.py", "args": vars(a), "rows": res, "summary": summ, "greybody": gb}, open(a.out_json, "w"), indent=1, default=float)
    if a.md:
        write_md(res, summ, gb, a)


def summarize(res, a):
    """per geometry class (R_lead, R_in for pg): c per size, the two-size Richardson c for each (r, mu), the fit (c_0, c_2, c_m)."""
    keyf = class_key
    out = []
    classes = sorted({keyf(rd) for rd in res}, key=lambda t: tuple(str(x) for x in t))
    for cl in classes:
        rows = [rd for rd in res if keyf(rd) == cl and "theta" in rd]
        pts = sorted({(rd["r"], rd["mu"]) for rd in rows})
        sizes = sorted({rd["kappa_a"] for rd in rows}, reverse=True)
        tab = []
        for (r, mu) in pts:
            cs = {}
            for ka in sizes:
                th = [rd["theta"] for rd in rows if rd["r"] == r and rd["mu"] == mu and rd["kappa_a"] == ka]
                if th:
                    cs[ka] = (1.0 - th[0]) / ka ** 2
            rich = None
            if len(cs) >= 2:
                k1, k2 = sorted(cs, reverse=True)[-2:]
                rich = (cs[k2] * k1 ** 2 - cs[k1] * k2 ** 2) / (k1 ** 2 - k2 ** 2)
            tab.append({"r": r, "mu": mu, "c_per_size": {str(k): c for k, c in cs.items()}, "c_richardson": rich})
        fit = None
        use = [t for t in tab if t["c_richardson"] is not None]
        if len(use) >= 3:
            Am = np.array([[1.0, t["r"] ** 2, t["mu"] ** 2] for t in use])
            bv = np.array([t["c_richardson"] for t in use])
            coef, resid, rank, sv = np.linalg.lstsq(Am, bv, rcond=None)
            dof = max(len(use) - 3, 1)
            s2 = float(np.sum((Am @ coef - bv) ** 2)) / dof
            cov = s2 * np.linalg.pinv(Am.T @ Am)
            fit = {"c0": float(coef[0]), "c2": float(coef[1]), "cm": float(coef[2]), "err": [float(math.sqrt(max(cov[i, i], 0))) for i in range(3)],
                   "rms_residual": math.sqrt(s2) if len(use) > 3 else None, "n_points": len(use), "rank": int(rank)}
        out.append({"class": list(cl), "table": tab, "fit": fit})
    return out


def class_key(rd):
    """the nominal geometry class: (profile tag, R_lead/r_s, the inner continuation, the outer saturation, the tanh truncation)."""
    g = rd["geometry"]
    if "r_s" not in g:
        return ("tanh", None, None, None, round(math.log10(abs(g["v_out"])), 1))
    if "R1" in g:
        return ("pgs", g["R_lead_nominal"], "R1/r_s %g w %g" % (g["R1"] / g["r_s"], g["w"]), g.get("outer_w"), None)
    return ("pg", g["R_lead_nominal"], "R_in/r_s %g W_in/r_s %g" % (g["R_in"] / g["r_s"], g["W_in"] / g["r_s"]), None, None)


def greybody_table(res):
    """Gamma per (r, mu, class) against kappa a, and the (kappa a)^2 extrapolation from the two smallest kappa a."""
    out = []
    for cl in sorted({class_key(rd) for rd in res}, key=lambda t: tuple(str(x) for x in t)):
        for (r, mu) in sorted({(rd["r"], rd["mu"]) for rd in res if class_key(rd) == cl and rd.get("Gamma") is not None and rd["mu"] > 0}):
            G = {rd["kappa_a"]: rd["Gamma"] for rd in res if class_key(rd) == cl and rd["r"] == r and rd["mu"] == mu and rd.get("Gamma") is not None}
            ext = None
            if len(G) >= 2:
                k1, k2 = sorted(G)[:2]
                ext = (G[k1] * k2 ** 2 - G[k2] * k1 ** 2) / (k2 ** 2 - k1 ** 2)
            out.append({"class": list(cl), "r": r, "mu": mu, "Gamma": {str(k): g for k, g in sorted(G.items())}, "extrapolated": ext})
    return out


def write_md(res, summ, gb, a):
    L = ["## J-PL (obs_plateau_v2.py): the plateau's (kappa a)^2 correction by stationary scattering on the 1 + 1 chain, profile %s" % a.profile, "",
         "| kappa a | class | kappa 2 r_s | r | mu | theta = T_conv/T_H | theta_n (from n_H) | Gamma | |beta|^2 | |alpha|^2 | R | unitarity | k_H |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for rd in res:
        g = rd["geometry"]
        cl = " ".join(str(x) for x in class_key(rd) if x is not None)
        k2 = ("%.9f" % g["kappa_2rs"]) if "kappa_2rs" in g else "-"
        if "error" in rd:
            L.append("| %g | %s | %s | %g | %g | %s | | | | | | | |" % (rd["kappa_a"], cl, k2, rd["r"], rd["mu"], rd["error"]))
            continue
        f = lambda key, fmt: (fmt % rd[key]) if rd.get(key) is not None else "-"
        L.append("| %g | %s | %s | %g | %g | %s | %s | %s | %s | %s | %s | %s | %s |" % (rd["kappa_a"], cl, k2, rd["r"], rd["mu"], f("theta", "%.9f"),
                 f("theta_n", "%.9f"), f("Gamma", "%.9f"), f("beta2", "%.6e"), f("alpha2", "%.6e"), f("R", "%.3e"), f("unitarity", "%.1e"), f("k_H", "%.5f")))
    L += ["", "### c = (1 - theta)/(kappa a)^2 per size, the two-size Richardson value, and the fit c = c_0 + c_2 r^2 + c_m mu^2", ""]
    for s in summ:
        L.append("Geometry class %s:" % " ".join(str(x) for x in s["class"] if x is not None))
        L.append("")
        L.append("| r | mu | c per size | c (Richardson, two smallest kappa a) |")
        L.append("|---|---|---|---|")
        for t in s["table"]:
            L.append("| %g | %g | %s | %s |" % (t["r"], t["mu"], ", ".join("%s: %.6f" % (k, c) for k, c in t["c_per_size"].items()),
                     ("%.6f" % t["c_richardson"]) if t["c_richardson"] is not None else "-"))
        if s["fit"]:
            ft = s["fit"]
            L.append("")
            L.append("fit: c_0 = %.5f +- %.5f, c_2 = %.5f +- %.5f, c_m = %.5f +- %.5f (%d points; rms residual %s)" % (ft["c0"], ft["err"][0], ft["c2"], ft["err"][1],
                     ft["cm"], ft["err"][2], ft["n_points"], ("%.2e" % ft["rms_residual"]) if ft["rms_residual"] is not None else "-"))
        L.append("")
    if gb:
        L += ["### Greybody: Gamma against kappa a per class, and the (kappa a)^2 extrapolation from the two smallest kappa a (P25: 0.88523533 at (mu, r) = (0.2, 0.25), 0.98975910 at (0.4, 0.6))", "",
              "| class | r | mu | Gamma per kappa a | extrapolated |", "|---|---|---|---|---|"]
        for t in gb:
            L.append("| %s | %g | %g | %s | %s |" % (" ".join(str(x) for x in t["class"] if x is not None), t["r"], t["mu"],
                     ", ".join("%s: %.9f" % (k, g) for k, g in t["Gamma"].items()), ("%.9f" % t["extrapolated"]) if t["extrapolated"] is not None else "-"))
        L.append("")
    L += ["- every number above is printed by the script; P25's coefficients for comparison: tanh (1/3, 5/18, 5/36), Painleve-Gullstrand (-5/8, -1, 3/4);",
          "  the tanh control must give 0.99443 at kappa a = 1/8 and 0.99965 at 1/32 (r = 1/4, m = 0)."]
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
