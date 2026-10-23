#!/usr/bin/env python
"""Strand S diagnostic D-a1 (SPEC_strandS_graded_entropy.md secs. 9.1 and 9.3; the lead's draft completed by the builder analyst): the
density of states of the sea at fixed boundary charge near the surface, on the static chart's locked background L (strand7_sea_page_v5's
LockedBackground: psi = 1 + A_0/max(r, s), chi = 1 - A_0/max(r, s), the shell radius s(T) the test-particle characteristic; the boundary
charge A_0 fixed; r_h = 4 A_0 = 2 G E_bdy; T_H = 1/(4 pi r_h)).

Per channel (j, kappa): the lane's radial polar operator A' (strand5_w0fast_v1.channel_matrix's construction, A' = m psi^-1 + psi^-3 D_j with
the integrating factor exp(kappa log(r_int/r_half)) and the per-channel cut, copied here with kappa = -(j + 1/2) as the lane's and +(j + 1/2)
as the second sign, BUILDER), on the int and half nodes of a uniform grid h; its polar factors from the SVD A' = W S V^T: U = W V^T,
E_0 = V S V^T; B_0 = E_0^(1/2) diag(chi) E_0^(1/2) (the clock allocation L = N psi = chi); eigh(B_0) = (eps_r, z_r); the pair modes
(z_r; +-U z_r)/sqrt2 with external energies +-eps_r.  Observables (the draft's channel()): N(T) = #{eps_r < T}, the near-surface N_ns with
the weights w_r = (1/2)(sum_{r_i < R_*} |z_r|^2 + sum_{r_i < R_*} |U z_r|^2) for R_* = 2 A_0 and R_* = s, the thermal S_th = 2 sum s_F(eps/T)
and E_th = 2 sum eps/(e^(eps/T) + 1) (particles and holes) with their near-surface shares, the Gaussian correction -(1/2) log(2 pi T^2 dE/dT)
(dE/dT by the central difference at T(1 +- 1e-3)); each summed over j <= j_max = floor(psi_s^2 s/h - 1/2) with the weight 2j + 1 per kappa
sign, the partial sums at j_max/4 and j_max/2 kept.  T = T_H (provisional; task 3's T_ray replaces it in sec. 9.2).
Runs: the sizes by --E0 (r_s0 = 2 E0; A_0 = --A0-over-rs0 r_s0), the markers N_s = 1/2, 4h/r_h, h T_H (the external times by root finding on
N_s(T) of L), the packet masses --masses (the massless packet and the light rectangle's at aM = 0.1: 0, 0.1/sqrt 2, 0.1), the box R_max =
--Rmax-over-rs r_s0 (BUILDER: 20 r_s0; the control at 40).  One JSON per run.  Torch on --device (cuda:N or cpu)."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
S7 = os.environ.get("STRAND7_CODE", os.path.join(HERE, "..", "strand7"))
sys.path.insert(0, S7)
from strand7_sea_page_v5 import LockedBackground  # noqa: E402


def channel_matrix(r_int, r_half, psi_int, psi_half, m, j, h, sign=-1, cut=4.0):
    """strand5_w0fast_v1.channel_matrix with the kappa sign as a parameter (kappa = sign (j + 1/2)); returns (A', k0)."""
    n = len(r_int)
    K = j + 0.5
    kap = sign * K
    k0 = max(1, int(math.floor(K / cut))) - 1
    mm = n - k0
    ri, rh = r_int[k0:], r_half[k0:]
    pi_, ph_ = psi_int[k0:], psi_half[k0:]
    D = np.zeros((mm, mm))
    idx = np.arange(mm)
    D[idx, idx] = np.exp(kap * np.log(ri / rh)) / h
    D[idx[1:], idx[:-1]] = -np.exp(kap * np.log(ri[:-1] / rh[1:])) / h
    Ap = np.zeros((2 * mm, 2 * mm))
    Ap[idx, idx] = m / pi_
    Ap[mm + idx, mm + idx] = m / ph_
    Ap[:mm, mm:] = (pi_ ** -3)[:, None] * D.T
    Ap[mm:, :mm] = -(ph_ ** -3)[:, None] * D
    return Ap, k0


def sF(x):
    x = np.abs(np.asarray(x, float))
    p = np.exp(-x)
    return np.log1p(p) + x * p / (1 + p)


def channel_modes(bg, T, j, sign, m, h, n, dev):
    import torch
    r_int = h * np.arange(1, n + 1, dtype=float)
    r_half = h * (np.arange(n, dtype=float) + 0.5)
    psi_i, chi_i, A, B = bg.at(T, r_int)
    psi_h, chi_h, _, _ = bg.at(T, r_half)
    Ap, k0 = channel_matrix(r_int, r_half, psi_i, psi_h, m, j, h, sign)
    mm = n - k0
    rdom = np.concatenate([r_int[k0:], r_half[k0:]])
    chi = np.concatenate([chi_i[k0:], chi_h[k0:]])
    X = torch.as_tensor(Ap, dtype=torch.float64, device=dev)
    W, S, Vh = torch.linalg.svd(X)
    U = W @ Vh
    Gp = (Vh.T * S ** 0.5) @ Vh
    L = torch.as_tensor(chi, dtype=torch.float64, device=dev)
    B0 = Gp @ (L[:, None] * Gp)
    B0 = 0.5 * (B0 + B0.T)
    eps, Z = torch.linalg.eigh(B0)
    UZ = U @ Z
    return eps.cpu().numpy(), (Z ** 2).cpu().numpy(), (UZ ** 2).cpu().numpy(), rdom, float(S.min().item())


def observables(eps, Z2, UZ2, rdom, Rstars, T):
    out = {}
    x = eps / T
    s = 2 * sF(x)
    e = 2 * eps / (np.exp(np.minimum(x, 700)) + 1)
    below = eps < T
    out.update(N=int(below.sum()), S=float(s.sum()), E=float(e.sum()))
    for name, Rs in Rstars.items():
        ins = rdom < Rs
        w = 0.5 * (Z2[ins].sum(0) + UZ2[ins].sum(0))
        out["N_ns_" + name] = float(w[below].sum())
        out["S_ns_" + name] = float((w * s).sum())
        out["E_ns_" + name] = float((w * e).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E0", type=float, required=True)
    ap.add_argument("--A0-over-rs0", type=float, default=0.234465)
    ap.add_argument("--R0-over-rs", type=float, default=4.0)
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--Rmax-over-rs", type=float, default=20.0)
    ap.add_argument("--T-end-over-rs", type=float, default=80.0)
    ap.add_argument("--masses", default="0,0.0707106781186548,0.1")
    ap.add_argument("--markers", default="half,lattice,TH")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    t0 = time.time()
    rs0 = 2 * a.E0
    A0 = a.A0_over_rs0 * rs0
    rh = 4 * A0
    TH = 1.0 / (4 * math.pi * rh)
    h = a.h
    n = int(round(a.Rmax_over_rs * rs0 / h))
    bg = LockedBackground(A0, rs0, a.R0_over_rs * rs0, a.T_end_over_rs * rs0)
    targets = {"half": 0.5, "lattice": 4 * h / rh, "TH": h * TH}
    out = {"meta": dict(vars(a), rs0=rs0, A0=A0, r_h=rh, T_H=TH, n=n, energy_control=bg.energy_control), "markers": {}}
    for mk in a.markers.split(","):
        Ns_t = targets[mk]
        f = lambda T: bg.N_s(T) - Ns_t
        Tm = brentq(f, 0.0, a.T_end_over_rs * rs0 * 0.999, xtol=1e-10)
        s = bg.s(Tm)
        psi_s = 1 + A0 / s
        jmax_idx = int(math.floor(psi_s ** 2 * s / h - 0.5))
        js = [k + 0.5 for k in range(0, max(jmax_idx, 0) + 1)]
        Rstars = {"2A0": 2 * A0, "s": s}
        rows = {}
        for m in [float(v) for v in a.masses.split(",")]:
            tot = {}
            tot_dT = {"E_plus": 0.0, "E_minus": 0.0}
            partial = {}
            per_j = []
            smin = np.inf
            for j in js:
                for sign in (-1, +1):
                    eps, Z2, UZ2, rdom, sm = channel_modes(bg, Tm, j, sign, m, h, n, a.device)
                    smin = min(smin, sm)
                    ob = observables(eps, Z2, UZ2, rdom, Rstars, TH)
                    wgt = 2 * j + 1
                    for k, v in ob.items():
                        tot[k] = tot.get(k, 0.0) + wgt * v
                    for fac, key in ((1 + 1e-3, "E_plus"), (1 - 1e-3, "E_minus")):
                        tot_dT[key] += wgt * observables(eps, Z2, UZ2, rdom, {}, TH * fac)["E"]
                    per_j.append({"j": j, "sign": sign, "N": ob["N"], "S": ob["S"], "S_ns_2A0": ob["S_ns_2A0"], "S_ns_s": ob["S_ns_s"],
                                  "eps_min": float(eps.min()), "eps_lowest5": [float(v) for v in np.sort(eps)[:5]]})
                if abs(j - (jmax_idx // 4 + 0.5)) < 1e-9:
                    partial["quarter"] = dict(tot)
                if abs(j - (jmax_idx // 2 + 0.5)) < 1e-9:
                    partial["half"] = dict(tot)
            dEdT = (tot_dT["E_plus"] - tot_dT["E_minus"]) / (2e-3 * TH)
            tot["dE_dT"] = dEdT
            tot["gauss_correction"] = -0.5 * math.log(2 * math.pi * TH ** 2 * dEdT) if dEdT > 0 else None
            rows["m=%g" % m] = {"total": tot, "partial": partial, "per_j": per_j, "smallest_singular_value": smin}
            print("E0 %g marker %s (T/rs0 %.3f, N_s %.4g, s/A0 %.4f, j_max %g) m %g: N %d, S %.5g, S_ns(2A0) %.5g, S_ns(s) %.5g, E %.4g (%.0f s)" % (
                a.E0, mk, Tm / rs0, bg.N_s(Tm), s / A0, js[-1], m, tot["N"], tot["S"], tot["S_ns_2A0"], tot["S_ns_s"], tot["E"], time.time() - t0), flush=True)
        out["markers"][mk] = {"T": Tm, "T_over_rs0": Tm / rs0, "N_s": bg.N_s(Tm), "s": s, "s_over_A0": s / A0, "psi_s": psi_s, "j_max": js[-1],
                              "R_star": Rstars, "rows": rows}
    out["seconds"] = time.time() - t0
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    print("wrote %s (%.0f s)" % (a.out_json, out["seconds"]))


if __name__ == "__main__":
    main()
