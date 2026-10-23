#!/usr/bin/env python
"""Task 2 post-processor (SPEC_task2_static_row_diagnostics.md secs. 1 and 2, the lead's draft completed by the builder analyst): the
static-row diagnostics from the stored S7a fields (s7a_<tag>_fields.npz: r, T, u, v at every 0.1 r_s output) and the run's record
(s7a_<tag>.json), on the shell code's own grid (strand7_shell_v4.UGrid: J1's finite-volume Laplacian with the Robin exterior; never retyped).

Per output: the sources rebuilt from the fields (I2), S1 = -psi K L u, S2 = chi S1/psi^2 - K L v; the clock weights w = S1/psi; A = sum w/(4 pi K),
B = sum(chi S1/psi^2 - S2)/(4 pi K) (the Gauss charges) beside the stored ones; log E_bdy = log(2A); D = A - B, D/E_bdy; B/A, C/A = -B/A,
M_clock/E = (1 + B/A)/2; the nodal stress ratio chi_kin = -psi S2/(2 N S1) (I3) and its S1/psi-weighted mean over the shell's nodes (S1 >
1e-6 max S1), the S1/psi-weighted shell lapse N_s, N_* = 1/(3 + 2 chi_kin); the thin-shell identity B/A - <N(1 + 2 chi_kin)>_{S1/psi} over
every node with S1 != 0; P07's corrected lapse N_p07 = psi^-1 - J_C^-T dE_0/dpsi with J = K L - diag(S1/psi^2) + diag(S2/(N psi)),
J_C = diag(psi) J (I1), its largest relative difference from N; N_min over the nodes and its node, the first node with chi <= 0; the
smallest singular values of J and J_C by inverse iteration on the sparse LU (the dense svdvals at --dense-every outputs as the check);
F_disc = K v^T L u + sum_x N_x S1_x (the nodal sum; K v^T L u contains the Robin boundary term 4 pi K A_n B_n/R_max, so the lead's draft's
separate + 4 pi K A B/R_max would count it twice: the builder's correction, labelled) and the markers' form K v^T L u + sum nu N W (the
record's sum_nu_H; part 2's F_disc).  Over the run: the barrier rate r(T) = -d log N_min/dT and its integral (I5); the throat values
interpolated at the recorded coordinate passage r_sh = A.  Writes <folder>/task2_<tag>.json.  Args: --folder --tags (comma list)
--dense-every --out-folder."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strand7_shell_v4 import K, UGrid  # noqa: E402


def sigma_min(M, iters=400, tol=1e-13, seed=1):
    """the smallest singular value of the sparse square M by inverse iteration on M^T M (two LU solves per step)."""
    lu = spla.splu(sp.csc_matrix(M))
    x = np.random.default_rng(seed).standard_normal(M.shape[0])
    x /= np.linalg.norm(x)
    lam = 0.0
    for it in range(iters):
        y = lu.solve(lu.solve(x), trans="T")
        nl = np.linalg.norm(y)
        x = y / nl
        if abs(nl - lam) <= tol * nl:
            lam = nl
            break
        lam = nl
    return 1.0 / math.sqrt(lam), it + 1


def diagnostics(G, u, v, sum_nu_H, dense=False):
    L = G.L
    psi, chi = 1 + u, 1 + v
    N = chi / psi
    Lu, Lv = L @ u, L @ v
    S1 = -psi * K * Lu
    S2 = chi * S1 / psi ** 2 - K * Lv
    w = S1 / psi
    A = w.sum() / (4 * np.pi * K)
    B = (chi * S1 / psi ** 2 - S2).sum() / (4 * np.pi * K)
    smax = np.max(np.abs(S1))
    nz = np.abs(S1) > 1e-14 * smax
    chik = np.full_like(S1, np.nan)
    chik[nz] = -psi[nz] * S2[nz] / (2 * N[nz] * S1[nz])
    sh = nz & (S1 > 1e-6 * smax)
    Ns = float((w[sh] * N[sh]).sum() / w[sh].sum())
    chis = float((w[sh] * chik[sh]).sum() / w[sh].sum())
    ident = B / A - float((w[nz] * N[nz] * (1 + 2 * chik[nz])).sum() / w.sum())
    J = (K * L + sp.diags(-S1 / psi ** 2 + S2 / (N * psi))).tocsc()
    JC = (sp.diags(psi) @ J).tocsc()
    dE0 = -S1 / psi ** 2 + S2 / (N * psi)
    N_p07 = 1 / psi - spla.spsolve(JC.T.tocsc(), dE0)
    sJ, itJ = sigma_min(J)
    sJC, itJC = sigma_min(JC)
    kmin = int(np.argmin(N))
    neg = np.nonzero(chi <= 0)[0]
    quad = float(K * v @ Lu)
    out = dict(A=float(A), B=float(B), logE=math.log(2 * A), A_n=float(G.Rmax * u[-1]), B_n=float(-G.Rmax * v[-1]), D=float(A - B),
               D_over_E=float((A - B) / (2 * A)), B_over_A=float(B / A), C_over_A=float(-B / A), Mclock_over_E=float((1 + B / A) / 2), N_s=Ns,
               chi_kin=chis, N_star=1 / (3 + 2 * chis), shell_identity=float(ident), shell_nodes=int(sh.sum()),
               p07_residual=float(np.max(np.abs(N - N_p07)) / np.max(np.abs(N))), N_min=float(N[kmin]), r_Nmin=float(G.r[kmin]),
               first_chi_nonpos_r=float(G.r[neg[0]]) if len(neg) else None, sigma_min_J=sJ, sigma_min_JC=sJC, sigma_iters=[itJ, itJC],
               F_disc_nodal=quad + float((N * S1).sum()), F_disc_markers=quad + float(sum_nu_H) if sum_nu_H is not None else None,
               quad_term=quad)
    if dense:
        Jd, JCd = J.toarray(), JC.toarray()
        out["sigma_min_J_dense"] = float(sla.svdvals(Jd).min())
        out["sigma_min_JC_dense"] = float(sla.svdvals(JCd).min())
    return out


def run(folder, tag, dense_every):
    rec = json.load(open(os.path.join(folder, "s7a_%s.json" % tag)))
    f = np.load(os.path.join(folder, "s7a_%s_fields.npz" % tag))
    meta = rec["meta"]
    h, n = float(meta["h"]), int(meta["n"])
    G = UGrid(n, h)
    assert np.allclose(G.r, f["r"]), "the stored grid differs from UGrid(n, h)"
    rs = float(meta["r_s"])
    T = np.asarray(f["T"], float)
    recs = rec["run"]["records"]
    rT = np.array([q["T"] for q in recs])
    out = []
    for i in range(len(T)):
        k = int(np.argmin(np.abs(rT - T[i])))
        snh = recs[k]["sum_nu_H"] if abs(rT[k] - T[i]) < 1e-6 * max(1.0, T[i]) else None
        d = diagnostics(G, f["u"][i], f["v"][i], snh, dense=(dense_every > 0 and i % dense_every == 0))
        d["T"] = float(T[i])
        d["T_over_rs"] = float(T[i] / rs)
        q = recs[k]
        d["rec_A"], d["rec_B"], d["rec_N_sh"], d["rec_r_sh_over_A"] = q["A"], q["B"], q["N_sh"], q["r_sh_over_A"]
        d["rec_residual"] = q.get("residual")
        out.append(d)
        if i % 50 == 0:
            print("%s output %d/%d T/r_s %.2f: A %.6g, B/A %.6g, N_s %.6g, chi_kin %.6g, P07 %.2e, sigma_min(J) %.3e, N_min %.4g" % (
                tag, i, len(T), d["T_over_rs"], d["A"], d["B_over_A"], d["N_s"], d["chi_kin"], d["p07_residual"], d["sigma_min_J"], d["N_min"]), flush=True)
    Nmin = np.array([o["N_min"] for o in out])
    r = -np.gradient(np.log(Nmin), T)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(T))])
    for o, ri, ci in zip(out, r, cum):
        o["barrier_rate"] = float(ri)
        o["barrier_int"] = float(ci)
    tc = rec["run"].get("throat_crossing")
    throat = None
    if tc:
        Tp = tc["T_over_rs"] * rs
        throat = {k: float(np.interp(Tp, T, [o[k] for o in out])) for k in ("N_s", "C_over_A", "Mclock_over_E", "chi_kin", "N_star", "B_over_A",
                                                                             "D_over_E", "shell_identity", "N_min", "sigma_min_J", "sigma_min_JC")}
        throat["T_over_rs"] = tc["T_over_rs"]
        throat["recorded"] = tc
    F0n, F0m = out[0]["F_disc_nodal"], out[0]["F_disc_markers"]
    for o in out:
        o["F_nodal_rel"] = o["F_disc_nodal"] / F0n - 1
        o["F_markers_rel"] = (o["F_disc_markers"] / F0m - 1) if (F0m and o["F_disc_markers"] is not None) else None
        o["F_nodal_defect"] = o["F_disc_nodal"] / (4 * np.pi * K * o["A_n"]) - 1
    return {"run": tag, "meta": {k: meta.get(k) for k in ("E0", "h", "n", "R_max", "r_s", "dt_over_rs", "Np", "wr", "R0_over_rs", "run", "tag")},
            "throat": throat, "series": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--dense-every", type=int, default=100)
    ap.add_argument("--out-folder", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_folder, exist_ok=True)
    for tag in a.tags.split(","):
        res = run(a.folder, tag, a.dense_every)
        json.dump(res, open(os.path.join(a.out_folder, "task2_%s.json" % tag), "w"), indent=1)
        print("wrote task2_%s.json" % tag, flush=True)


if __name__ == "__main__":
    main()
