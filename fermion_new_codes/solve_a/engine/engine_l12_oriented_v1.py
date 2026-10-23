#!/usr/bin/env python
"""engine_l12_oriented_v1.py (builder analyst; engine spec 28.5, S4 (d) of REVIEWA_REVIEW_web40e: "(d) oriented-plane transport at isolated real
zero pairs in place of the maximum overlap, tested on the L = 12 chain at m a 0 and on vf9's channels 1 to 4"). The chain of engine_l12_unit_v1
at m a = 0 (h = i A with A real skew in the lane's basis: S4 (c)) with the ramp amplitude raised to --amp (1.3: each D-sector's lowest pair
of levels +-eps passes through zero inside the ramp, near s 0.65, and ends at |eps| about 0.04). Four continuations of the occupied set along n_rec
record steps, against the propagated state R(s) (DOP853 at rtol 1e-12):
  (1) the oriented-plane rule (P07's construction, re-derived here): per D-sector the pair of levels with the smallest |e| spans a real two-plane
      Q = B B^T (B = sqrt 2 (Re u, Im u) for the eigenvector u of +eps; the conjugate u* carries -eps); when the previous step held exactly one
      occupied line on the previous plane (the neutral case P + P* = Q), the line is carried by the direct rotation of the planes,
      phi = B O B_0^T phi_0 with O = polar(B^T B_0); the other levels by the largest weight on the previous occupied set;
  (2) the engine's convention, engine_ops_v3.continue_sea_v3 (the largest weight, with its blocks and LZ sub-steps), at c_LZ 1;
  (3) the fixed-bit spectral reference of engine_l12_unit_v1 (the occupation bits fixed in eigenvalue order per sector): it re-fills at the zero
      crossing, the alternative the continued sea excludes;
  (4) the largest weight alone (no plane rule) per sector, as the comparator of (1).
Printed per tau and n_rec: d(R, P) = ||R - P||_F^2/2 at s = 1 for each, the largest over the records, and the tracked line's signed energy
Tr(h P_line) through the crossing (continuous through zero for the transported line). Every number printed by the script. CPU.
Args: --amp --taus --nrecs --md --json."""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, polar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_l12_unit_v1 as U1  # noqa: E402
import engine_ops_v3 as OPS3  # noqa: E402


def d2(A, B):
    return 0.5 * float(np.linalg.norm(A - B, "fro") ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp", type=float, default=1.3); ap.add_argument("--taus", default="16,64,256,1024"); ap.add_argument("--nrecs", default="16,64,256")
    ap.add_argument("--md", required=True); ap.add_argument("--json", required=True)
    a = ap.parse_args()
    import torch
    import scipy.sparse as sp
    U1.M_A = 0.0; U1.AMP = a.amp
    H0, Q, H1, DH, sectors = U1.build()
    assert np.max(np.abs(H1 + H1.conj())) < 1e-14 and np.max(np.abs(DH + DH.conj())) < 1e-14, "not real skew at m 0"
    e0, z0 = eigh(H0); Y0 = z0[:, e0 < 0]; P0 = Y0 @ Y0.conj().T; n_sea = Y0.shape[1]
    h = lambda s: H1 + U1.ramp(s) * DH
    occ_bits = []
    for ids in sectors:
        e, z = eigh(H1[np.ix_(ids, ids)])
        w = np.real(np.diag(z.conj().T @ P0[np.ix_(ids, ids)] @ z)); occ_bits.append(w > 0.5)

    def p_spectral(s):
        P = np.zeros_like(H0)
        for ids, oc in zip(sectors, occ_bits):
            e, z = eigh(h(s)[np.ix_(ids, ids)])
            P[np.ix_(ids, ids)] = z[:, oc] @ z[:, oc].conj().T
        return P

    def propagate(tau, s_list):
        def rhs(s, y):
            return (-1j * tau * (h(s) @ y.reshape(Y0.shape))).ravel()
        sol = solve_ivp(rhs, (0.0, 1.0), Y0.ravel().astype(complex), method="DOP853", rtol=1e-12, atol=1e-14, t_eval=s_list)
        assert sol.success, sol.message
        return [sol.y[:, j].reshape(Y0.shape) for j in range(len(s_list))]

    def plane_basis(z, e, j):
        """the real orthonormal basis of the plane of the eigenvector z[:, j] (energy e[j]) and its conjugate."""
        u = z[:, j]; u = u / np.linalg.norm(u)
        x, y = np.sqrt(2) * u.real, np.sqrt(2) * u.imag
        B = np.stack([x, y], axis=1)
        q, _ = np.linalg.qr(B)
        return q

    def oriented_run(nrec):
        """the oriented-plane continuation per sector; returns the projectors at the records and the tracked lines' signed energies."""
        state = []
        for ids, oc in zip(sectors, occ_bits):
            e, z = eigh(H1[np.ix_(ids, ids)])
            state.append({"cols": z[:, oc], "plane": None, "line": None})
        Ps = []; lam = []
        for k in range(nrec + 1):
            s = k / nrec
            P = np.zeros_like(H0); lam_k = []
            for si, (ids, st) in enumerate(zip(sectors, state)):
                e, z = eigh(h(s)[np.ix_(ids, ids)])
                jmin = int(np.argmin(np.abs(e)))
                B = plane_basis(z, e, jmin)
                Qp = B @ B.T
                pair = [j for j in range(len(e)) if np.linalg.norm(Qp @ z[:, j]) > 0.5]
                rest = [j for j in range(len(e)) if j not in pair]
                prev = st["cols"]
                wprev = (np.abs(z.conj().T @ prev) ** 2).sum(axis=1)
                line = None
                if st["plane"] is not None and st["line"] is not None:
                    O, _ = polar(B.T @ st["plane"])
                    line = B @ (O @ (st["plane"].T @ st["line"]))
                    line = line / np.linalg.norm(line)
                n_line = 1 if line is not None else int(round(sum(wprev[j] for j in pair)))
                n_rest = prev.shape[1] - (1 if line is not None else n_line)
                order = sorted(rest, key=lambda j: -wprev[j])
                cols = [z[:, j] for j in order[:n_rest]]
                if line is not None:
                    cols.append(line)
                else:
                    if n_line == 1:
                        jo = max(pair, key=lambda j: wprev[j]); line = z[:, jo]; cols.append(line)
                    elif n_line == 2:
                        cols += [z[:, j] for j in pair]
                C = np.stack(cols, axis=1)
                st["cols"] = C
                if n_line == 1 or (st["line"] is not None):
                    st["plane"] = B; st["line"] = line
                    lam_k.append(float(np.real(line.conj() @ (h(s)[np.ix_(ids, ids)] @ line))))
                else:
                    st["plane"] = None; st["line"] = None; lam_k.append(None)
                P[np.ix_(ids, ids)] = C @ C.conj().T
            Ps.append(P); lam.append(lam_k)
        return Ps, lam

    def maxweight_run(nrec):
        state = []
        for ids, oc in zip(sectors, occ_bits):
            e, z = eigh(H1[np.ix_(ids, ids)])
            state.append(z[:, oc])
        Ps = []
        for k in range(nrec + 1):
            s = k / nrec
            P = np.zeros_like(H0)
            for si, ids in enumerate(sectors):
                e, z = eigh(h(s)[np.ix_(ids, ids)])
                w = (np.abs(z.conj().T @ state[si]) ** 2).sum(axis=1)
                sel = np.argsort(-w)[:state[si].shape[1]]
                state[si] = z[:, sel]
                P[np.ix_(ids, ids)] = state[si] @ state[si].conj().T
            Ps.append(P)
        return Ps

    tt = lambda X: torch.as_tensor(X, dtype=torch.complex128)

    def eig_t(Hm):
        e, Uv = torch.linalg.eigh(tt(Hm)); return e.real, Uv

    def engine_run(nrec, tau):
        e_i, U_i = eig_t(H1)
        ref0, info0, mask0 = OPS3.assign_max_overlap_v3(tt(Y0), e_i, U_i, n_sea)
        window = [{"ref": ref0, "f": (np.array([0.0]), np.array([0.0])), "gap": info0.get("gap_between_blocks"), "weyl_gap": info0.get("weyl_gap"), "mask": mask0}]
        Ps = [(ref0 @ ref0.conj().T).numpy()]; flags = {"unverified": 0, "ties": 0, "partly_occupied": 0}
        eig_at = lambda ch, f0, f1: eig_t(H1 + float(np.asarray(f0).ravel()[0]) * DH)
        h_of = lambda ch, d0, d1: sp.csr_matrix(float(np.asarray(d0).ravel()[0]) * DH)
        for k in range(1, nrec + 1):
            s = k / nrec; f_k = (np.array([U1.ramp(s)]), np.array([0.0]))
            e_k, U_k = eig_t(h(s))
            ref, ci, mask = OPS3.continue_sea_v3(None, window, f_k, e_k, U_k, n_sea, tau / nrec, 1.0, eig_at, h_of)
            window = [{"ref": ref, "f": f_k, "gap": ci.get("gap_between_blocks"), "weyl_gap": ci.get("weyl_gap", ci.get("gap_between_blocks")), "mask": mask}]
            flags["unverified"] += int(not ci.get("verified", False)); flags["ties"] += int(bool(ci.get("ambiguous_tie"))); flags["partly_occupied"] += int(ci.get("partly_occupied_blocks", 0) > 0)
            Ps.append((ref @ ref.conj().T).numpy())
        return Ps, flags

    ss_fine = np.linspace(0, 1, 4001)
    emin = [float(np.min(np.abs(eigh(h(s)[np.ix_(sectors[0], sectors[0])], eigvals_only=True)))) for s in ss_fine]
    k0 = int(np.argmin(emin))
    rows = []
    for nrec in [int(x) for x in a.nrecs.split(",")]:
        Po, lam = oriented_run(nrec)
        Pm = maxweight_run(nrec)
        s_list = [k / nrec for k in range(nrec + 1)]
        Pspec = [p_spectral(s) for s in s_list]
        for tau in [float(x) for x in a.taus.split(",")]:
            Ys = propagate(tau, s_list)
            Rs = [Y @ Y.conj().T for Y in Ys]
            Pe, fl = engine_run(nrec, tau)
            dist = {lab: [d2(R, P) for R, P in zip(Rs, PP)] for lab, PP in (("oriented", Po), ("engine_v3", Pe), ("spectral_fixed_bits", Pspec), ("max_weight", Pm))}
            rows.append({"n_rec": nrec, "tau": tau, "final": {k: v[-1] for k, v in dist.items()}, "largest": {k: max(v) for k, v in dist.items()},
                         "oriented_vs_engine_final": d2(Po[-1], Pe[-1]), "oriented_vs_maxweight_final": d2(Po[-1], Pm[-1]), "engine_flags": fl})
            print("n_rec %d tau %g done" % (nrec, tau), flush=True)
        lam_series = [[s_list[k], lam[k]] for k in range(0, nrec + 1, max(1, nrec // 16))]
        rows[-1]["line_energy_series"] = lam_series
    out = {"script": "engine_l12_oriented_v1.py", "amp": a.amp, "n_sea": n_sea, "bits": [[int(b) for b in oc] for oc in occ_bits], "crossing_s_grid": float(ss_fine[k0]),
           "crossing_min_abs_e": emin[k0], "rows": rows}
    json.dump(out, open(a.json, "w"), indent=1, default=float)
    L = ["## The L = 12 chain at m a 0 through a zero crossing: the oriented-plane rule against the engine's convention (engine_l12_oriented_v1.py; engine spec 28.5, S4 (d))", "",
         "- the chain of engine_l12_unit_v1 at m a 0 (h = i A, A real skew, asserted), the ramp amplitude %g; the bits per D-sector %s; each sector's lowest pair passes through zero near s %.4f (the smallest |e| on a 4001-point grid %.2e)." % (
             a.amp, "; ".join("(" + ",".join(str(int(b)) for b in oc) + ")" for oc in occ_bits), ss_fine[k0], emin[k0]),
         "- d(R, P) = ||R - P||_F^2 / 2 against the propagated state R; oriented: the direct rotation of the pair's real plane carrying the occupied line, the rest by the largest weight; engine_v3: continue_sea_v3 at c_LZ 1; spectral_fixed_bits: the bits fixed in eigenvalue order (re-fills at the crossing); max_weight: the largest weight alone per sector.", "",
         "| n_rec | tau | oriented (s = 1; largest) | engine_v3 (s = 1; largest) | spectral fixed bits (s = 1; largest) | max weight (s = 1; largest) | d(oriented, engine) at s = 1 | engine flags (unverified, ties, partly occupied) |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f, g = r["final"], r["largest"]
        L.append("| %d | %g | %.3e; %.3e | %.3e; %.3e | %.3e; %.3e | %.3e; %.3e | %.3e | %d, %d, %d |" % (
            r["n_rec"], r["tau"], f["oriented"], g["oriented"], f["engine_v3"], g["engine_v3"], f["spectral_fixed_bits"], g["spectral_fixed_bits"], f["max_weight"], g["max_weight"],
            r["oriented_vs_engine_final"], r["engine_flags"]["unverified"], r["engine_flags"]["ties"], r["engine_flags"]["partly_occupied"]))
    L += ["", "### The tracked line's signed energy Tr(h P_line) per sector (the oriented run at the largest n_rec)", "", "| s | sector 1 | sector 2 |", "|---|---|---|"]
    for s_, lk in rows[-1].get("line_energy_series", []):
        L.append("| %.4f | %s | %s |" % (s_, ("%+.5e" % lk[0]) if lk[0] is not None else "-", ("%+.5e" % lk[1]) if lk[1] is not None else "-"))
    L += ["", "- every number above is printed by the script; the propagated state follows the no-refilling branch through a protected zero crossing (the in-pair matrix element of any real-skew perturbation vanishes), so the rule that tracks it is the one whose d(R, P) falls with tau."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
