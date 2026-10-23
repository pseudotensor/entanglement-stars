#!/usr/bin/env python
"""engine_l12_unit_v1.py (builder analyst; REVIEWA_REVIEW_web40e.md, SOLVE LANE S9 (b), the owner's binding guidance of 2026-10-16: "run the
L = 12 antiperiodic chain (ma 0.2, v = -2, bits (1,1,1,1,0,0,1,1,0,0,0,0) per D-sector) through the actual continuation routine at c_LZ
0.5, 1, 2 and report d(R, P_engine) and d(P_engine, P_spectral) against d(R, P) = 0.0096442277 (tau 4), 1.147e-3 (16), 2.609e-5 (64), min
gap 0.1540956128").

The chain is the lane's own construction of the review's operator test, written from its definition (the web file is an idea, not a
source): L sites, two components, the antiperiodic one-step shift T (T[j, j+1] = 1, T[L-1, 0] = -1), S = (T - T^+)/(2i), the two-step
current Q = (T^2 - T^+2)/(4i) times the identity on the components, H0 = S (x) sigma_x + m 1 (x) sigma_z with m = 0.2, the uniform flow
H1 = H0 + v Q with v = -2; the D-sectors d = (-1)^j (+1, -1) over (site, component), which every term preserves; the ramp
h(s) = H1 + r(s) DH, r(s) = s^3 (10 - 15 s + 6 s^2), DH_ij = (dv_i + dv_j) Q_ij / 2 with dv = -2 A f(j) on both components,
f(j) = cos(2 pi j/L) + 0.37 sin(4 pi j/L + 0.2), A = 0.3 (no lapse term). The initial state is the flat sea P0 (the negative-energy
projector of H0), which commutes with H1: its occupation in H1's eigenbasis is a fixed bit pattern per D-sector (printed and checked).
R(s): the 12 occupied orbitals propagated by i dY/ds = tau h(s) Y (scipy DOP853, rtol 1e-12, atol 1e-14). P_spectral(s): per sector the
eigenvectors of h(s) at the fixed bit positions in eigenvalue order. P_engine: engine_ops_v3.continue_sea_v3 (the production routine,
unchanged) applied at n_rec equal record steps of the ramp (the engine's time step tau / n_rec), its window kept as the engine keeps it
(the last accepted entry with ref, f, gap, weyl_gap, mask), with c_LZ in --clz. d(A, B) = ||A - B||_F^2 / 2. The reference's smallest
within-sector occupied-to-empty gap over s: a 4001-point grid refined by bounded minimization. Every number is printed by the script.
CPU (torch on the CPU, complex128). Args: --taus --nrecs --clz --md --json."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_ops_v3 as OPS3  # noqa: E402

L, M_A, V_FLOW, AMP = 12, 0.2, -2.0, 0.3
SX = np.array([[0, 1], [1, 0]], complex)
SZ = np.diag([1.0, -1.0]).astype(complex)


def build():
    T = np.zeros((L, L), complex)
    for j in range(L - 1):
        T[j, j + 1] = 1.0
    T[L - 1, 0] = -1.0
    S = (T - T.conj().T) / 2j
    Q1 = (T @ T - T.conj().T @ T.conj().T) / 4j
    H0 = np.kron(S, SX) + np.kron(np.eye(L), M_A * SZ)
    Q = np.kron(Q1, np.eye(2))
    H1 = H0 + V_FLOW * Q
    j = np.arange(L)
    f = np.cos(2 * np.pi * j / L) + 0.37 * np.sin(4 * np.pi * j / L + 0.2)
    dv = np.repeat(-2.0 * AMP * f, 2)
    DH = 0.5 * (dv[:, None] + dv[None, :]) * Q
    d = np.repeat((-1.0) ** j, 2) * np.tile([1.0, -1.0], L)
    sectors = [np.flatnonzero(d == s) for s in (1.0, -1.0)]
    for X in (H0, Q, DH):
        for a_ in sectors:
            for b_ in sectors:
                if a_ is not b_:
                    assert np.max(np.abs(X[np.ix_(a_, b_)])) < 1e-14, "a term mixes the D-sectors"
    return H0, Q, H1, DH, sectors


def ramp(s):
    return s ** 3 * (10.0 + s * (-15.0 + 6.0 * s))


def proj_neg(H):
    e, z = eigh(H)
    return z[:, e < 0], e


def d2(A, B):
    return 0.5 * float(np.linalg.norm(A - B, "fro") ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", default="4,16,64"); ap.add_argument("--nrecs", default="1,2,4,8,16,64"); ap.add_argument("--clz", default="0.5,1,2")
    ap.add_argument("--md", required=True); ap.add_argument("--json", required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    H0, Q, H1, DH, sectors = build()
    Y0, e0 = proj_neg(H0)
    P0 = Y0 @ Y0.conj().T
    n_sea = Y0.shape[1]
    occ = []
    for ids in sectors:
        e, z = eigh(H1[np.ix_(ids, ids)])
        w = np.real(np.diag(z.conj().T @ P0[np.ix_(ids, ids)] @ z))
        assert np.max(np.abs(w - np.round(w))) < 1e-10, "P0 does not commute with H1 in a sector"
        occ.append(w > 0.5)
    h = lambda s: H1 + ramp(s) * DH

    def p_spectral(s):
        P = np.zeros_like(H0); gap = math.inf
        for ids, oc in zip(sectors, occ):
            e, z = eigh(h(s)[np.ix_(ids, ids)])
            P[np.ix_(ids, ids)] = z[:, oc] @ z[:, oc].conj().T
            gap = min(gap, float(np.min(np.abs(e[oc][:, None] - e[~oc][None, :]))))
        return P, gap

    xs = np.linspace(0.0, 1.0, 4001)
    gs = np.array([p_spectral(x)[1] for x in xs])
    k0 = int(np.argmin(gs))
    lo, hi = xs[max(k0 - 1, 0)], xs[min(k0 + 1, len(xs) - 1)]
    res = minimize_scalar(lambda s: p_spectral(s)[1], bounds=(lo, hi), method="bounded", options={"xatol": 1e-14})
    gmin, smin = (float(res.fun), float(res.x)) if res.fun < gs[k0] else (float(gs[k0]), float(xs[k0]))

    def propagate(tau):
        n = Y0.size

        def rhs(s, y):
            Y = y.reshape(Y0.shape)
            return (-1j * tau * (h(s) @ Y)).ravel()
        sol = solve_ivp(rhs, (0.0, 1.0), Y0.ravel().astype(complex), method="DOP853", rtol=1e-12, atol=1e-14)
        assert sol.success, sol.message
        Y = sol.y[:, -1].reshape(Y0.shape)
        return Y @ Y.conj().T, float(np.linalg.norm(Y.conj().T @ Y - np.eye(n_sea), "fro")), int(sol.nfev)

    tt = lambda X: torch.as_tensor(X, dtype=torch.complex128)

    def eig_t(H):
        e, U = torch.linalg.eigh(tt(H))
        return e.real, U

    def eig_at(ch, f0, f1):
        return eig_t(H1 + float(np.asarray(f0).ravel()[0]) * DH)

    def h_of(ch, d0, d1):
        return sp.csr_matrix(float(np.asarray(d0).ravel()[0]) * DH)

    rows = []
    P_spec1, _ = p_spectral(1.0)
    for tau in [float(x) for x in a.taus.split(",")]:
        t0 = time.time()
        R, orth, nfev = propagate(tau)
        dRP = d2(R, P_spec1)
        for nrec in [int(x) for x in a.nrecs.split(",")]:
            for c in [float(x) for x in a.clz.split(",")]:
                e_init, U_init = eig_t(H1)
                ref0, info0, mask0 = OPS3.assign_max_overlap_v3(tt(Y0), e_init, U_init, n_sea)
                window = [{"ref": ref0, "f": (np.array([0.0]), np.array([0.0])), "gap": info0.get("gap_between_blocks"),
                           "weyl_gap": info0.get("weyl_gap", info0.get("gap_between_blocks")), "mask": mask0}]
                nsub_tot = 0; unverified = 0; ties = 0; partial = 0; pockets = 0; worst = 0.0; modes = set()
                for k in range(1, nrec + 1):
                    s_k = k / nrec
                    f_k = (np.array([ramp(s_k)]), np.array([0.0]))
                    e_k, U_k = eig_t(h(s_k))
                    ref, ci, mask = OPS3.continue_sea_v3(None, window, f_k, e_k, U_k, n_sea, tau / nrec, c, eig_at, h_of)
                    window.append({"ref": ref, "f": f_k, "gap": ci.get("gap_between_blocks"), "weyl_gap": ci.get("weyl_gap", ci.get("gap_between_blocks")),
                                   "mask": mask})
                    window = window[-1:]
                    Pe = (ref @ ref.conj().T).numpy()
                    worst = max(worst, d2(Pe, p_spectral(s_k)[0]))
                    nsub_tot += int(ci.get("nsub", 1)); unverified += int(not ci.get("verified", False)); ties += int(bool(ci.get("ambiguous_tie")))
                    partial += int(ci.get("partly_occupied_blocks", 0) > 0); pockets += int(ci.get("n_pocket", 0)); modes.add(ci.get("mode", "?").split(" (")[0])
                Pe1 = (window[-1]["ref"] @ window[-1]["ref"].conj().T).numpy()
                rows.append({"tau": tau, "n_rec": nrec, "c_LZ": c, "d_R_Pspectral": dRP, "d_R_Pengine": d2(R, Pe1), "d_Pengine_Pspectral": d2(Pe1, P_spec1),
                             "worst_step_d_Pengine_Pspectral": worst, "nsub_total": nsub_tot, "unverified_steps": unverified, "ambiguous_ties": ties,
                             "steps_with_partly_occupied_blocks": partial, "pocket_count_sum": pockets, "modes": sorted(modes), "orthonormality_R": orth, "nfev": nfev})
        print("tau %g done in %.1f s" % (tau, time.time() - t0), flush=True)
    out = {"script": "engine_l12_unit_v1.py", "L": L, "m": M_A, "v": V_FLOW, "A": AMP, "n_sea": n_sea,
           "bits_per_sector": [[int(b) for b in oc] for oc in occ], "min_gap": gmin, "s_at_min_gap": smin, "rows": rows}
    json.dump(out, open(a.json, "w"), indent=1)
    Lm = ["## The L = 12 chain through engine_ops_v3's continuation (engine_l12_unit_v1.py; REVIEWA_REVIEW_web40e.md S9 (b))", "",
          "- the chain: L %d, m a %g, v %g, ramp amplitude %g; n_sea %d; the flat sea's bits per D-sector in H1's eigenvalue order: %s; %s." % (
              L, M_A, V_FLOW, AMP, n_sea, "; ".join("(" + ",".join(str(int(b)) for b in oc) + ")" for oc in occ),
              "the review's (1,1,1,1,0,0,1,1,0,0,0,0) in both sectors" if all(list(map(int, oc)) == [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0] for oc in occ) else "NOT the review's pattern"),
          "- the reference's smallest within-sector occupied-to-empty gap over the ramp: %.10f at s = %.6f (the review: 0.1540956128)." % (gmin, smin),
          "- d(A, B) = ||A - B||_F^2 / 2; P_spectral the fixed-bit spectral reference; P_engine continue_sea_v3 at n_rec record steps (time step tau/n_rec); R the propagated state.", "",
          "| tau | n_rec | c_LZ | d(R, P_spectral) | d(R, P_engine) | d(P_engine, P_spectral) at s = 1 | largest d(P_engine, P_spectral) over the records | LZ sub-steps | unverified steps | ties | steps with partly occupied blocks | modes |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        Lm.append("| %g | %d | %g | %.10e | %.10e | %.3e | %.3e | %d | %d | %d | %d | %s |" % (r["tau"], r["n_rec"], r["c_LZ"], r["d_R_Pspectral"], r["d_R_Pengine"], r["d_Pengine_Pspectral"],
                                                                                     r["worst_step_d_Pengine_Pspectral"], r["nsub_total"], r["unverified_steps"], r["ambiguous_ties"],
                                                                                     r["steps_with_partly_occupied_blocks"], ", ".join(r["modes"])))
    Lm += ["", "- the review's d(R, P): 0.0096442277 (tau 4), 1.147e-3 (16), 2.609e-5 (64); the propagation's orthonormality error at the end below %.1e; every number above is printed by the script." % max(r["orthonormality_R"] for r in rows)]
    open(a.md, "w").write("\n".join(Lm) + "\n")
    print("\n".join(Lm))


if __name__ == "__main__":
    main()
