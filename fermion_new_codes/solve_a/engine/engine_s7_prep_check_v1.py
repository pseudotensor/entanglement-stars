#!/usr/bin/env python
"""engine_s7_prep_check_v1.py (builder analyst; engine spec 29.5, S7: "Two copies of one channel (kappa 1; kappa 2 the check) form C_0 (+)
C_0; C_+- = R_+- (C_0 (+) C_0) R_+-^dagger with R_+- = sum_x |x><x| (x) exp(+-i theta chi_X(x) sigma_y) in copy space; a uniform rotation
commutes with C_0 (+) C_0, so the preparations differ only across the window's transition, where chi_X varies within the hop range; the
copy-summed site energy and momentum densities, which alone source the spherical fields, are compared with C_0's to 1e-15 as 28.5 says,
the transition's shells printed, two windows printed (the transition at r_in/2 and at (r_in + R_t)/2 of the record at T/rs 20 of the r_s 32
tau 8 member) and theta pi/4 and pi/2"): the preparation check (part 1; the propagation on the member's history is part 2, not here). The
record: the member's record nearest T/rs 20 that holds both trapping zeros r_in and R_t (at T/rs 20 itself the tau 8 member has none; the
choice is the builder's, labelled). C_0 per channel: the negative-energy projector of the record's h_m at its applied fields (the
instantaneous sea stands for the state in this check of the construction; part 2 uses the propagated state); the window chi_X(r) =
(1/2) erfc((r - X)/w) with w = --width sites (default 2; the transition within the hop range of the two-step operator); per channel,
window and theta: C_+- = R_+- (I_2 (x) C_0) R_+-^dagger, the copy-summed densities eps_x = Tr[(I_2 (x) a_x(h_m)) C] and J_x = Tr[(I_2 (x)
a_x(Q)) C] (a_x(B) = {e_x, B}/2, e_x the site's two components) against I_2 (x) C_0's; the largest |difference| over the largest |density|
away from the transition (sites farther than 3 w from X) and the shells where the difference exceeds --tol (1e-15) printed. Every number
printed by the script. CPU, seconds. Args: --run dir/tag (a stored run with json and fields npz) --T-over-rs --kappas --thetas --width --tol
--md --out-json."""
import argparse
import json
import math
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.special import erfc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_collapse_v19 as E19  # noqa: E402


def site_density(B, C, n):
    """Tr[(a_x(B) (x) I_copies) C] summed over the two copies, for every site x: a_x(B) = {e_x, B}/2 on the site's two components."""
    # C is (2 copies x 2n) square; the density for copy-diagonal blocks: Re sum over components of (B C_cc + C_cc B)/2 restricted to site x
    out = np.zeros(n)
    for c in range(2):
        Ccc = C[c * 2 * n:(c + 1) * 2 * n, c * 2 * n:(c + 1) * 2 * n]
        BC = B @ Ccc
        d = 0.5 * (np.diag(BC) + np.diag(Ccc @ B.toarray() if sp.issparse(B) else Ccc @ B))
        out += np.real(d.reshape(n, 2).sum(axis=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--T-over-rs", dest="t", type=float, default=20.0)
    ap.add_argument("--kappas", default="1,2"); ap.add_argument("--thetas", default="0.7853981633974483,1.5707963267948966")
    ap.add_argument("--width", type=float, default=2.0); ap.add_argument("--tol", type=float, default=1e-15)
    ap.add_argument("--md", required=True); ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    E19.C5.backend(None)
    d, tag = os.path.split(a.run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag))); Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    ra = J["args"]; rs = float(ra["rs"]); n = int(ra["nsites"]); m = float(ra["m"])
    recs = [r for r in J["records"] if r.get("E") is not None and r.get("R_t") is not None and r.get("r_in") is not None]
    assert recs, "no record holds both trapping zeros"
    rec = min(recs, key=lambda r: abs(float(r["T"]) / rs - a.t))
    T = float(rec["T"]); Tz = np.asarray(Z["T"], float); kz = int(np.argmin(np.abs(Tz - T)))
    N = np.asarray(Z["N_applied"][kz], float); v = np.asarray(Z["v_applied"][kz], float)
    rin, Rt = float(rec["r_in"]), float(rec["R_t"])
    windows = [("r_in/2", 0.5 * rin), ("(r_in + R_t)/2", 0.5 * (rin + Rt))]
    thetas = [float(x) for x in a.thetas.split(",")]
    sy = np.array([[0.0, -1j], [1j, 0.0]])
    rows = []
    for kap in [int(k) for k in a.kappas.split(",")]:
        ch = E19.Chan(kap, m, n, bool(ra.get("half_offset", 0)))
        R = ch.R
        h = E19.clocked_host_sparse(ch, N, v).tocsr()
        Qm = ch.Qs.tocsr()
        e, U = np.linalg.eigh(h.toarray())
        P = U[:, e < 0] @ U[:, e < 0].conj().T
        C0 = np.kron(np.eye(2), P)
        eps0 = site_density(h, C0, n); J0 = site_density(Qm, C0, n)
        for wl, X in windows:
            chi = 0.5 * erfc((R - X) / a.width)
            far = np.abs(R - X) > 3.0 * a.width
            for th in thetas:
                for sgn in (+1, -1):
                    # R_+- = sum_x |x><x| (x) exp(i sgn theta chi(x) sigma_y) in copy space: per site a 2x2 copy rotation on both components
                    c = np.cos(th * chi); s_ = np.sin(th * chi)
                    # exp(i phi sigma_y) = cos(phi) I + i sin(phi) sigma_y = [[cos, sin], [-sin, cos]]
                    Rm = np.zeros((4 * n, 4 * n))
                    idx = np.arange(2 * n)
                    cc = np.repeat(c, 2); ss = np.repeat(s_, 2) * sgn
                    Rm[idx, idx] = cc; Rm[2 * n + idx, 2 * n + idx] = cc
                    Rm[idx, 2 * n + idx] = ss; Rm[2 * n + idx, idx] = -ss
                    Cpm = Rm @ C0 @ Rm.T
                    eps = site_density(h, Cpm, n); Jd = site_density(Qm, Cpm, n)
                    de = np.abs(eps - eps0); dJ = np.abs(Jd - J0)
                    scale_e = max(float(np.max(np.abs(eps0))), 1e-300); scale_J = max(float(np.max(np.abs(J0))), 1e-300)
                    bad = np.nonzero((de > a.tol * scale_e) | (dJ > a.tol * scale_J))[0]
                    rows.append({"kappa": kap, "window": wl, "X": X, "theta": th, "sign": sgn,
                                 "max_rel_eps_far": float(np.max(de[far]) / scale_e) if far.any() else None, "max_rel_J_far": float(np.max(dJ[far]) / scale_J) if far.any() else None,
                                 "max_rel_eps_all": float(np.max(de) / scale_e), "max_rel_J_all": float(np.max(dJ) / scale_J),
                                 "shells_above_tol": [float(R[i]) for i in bad][:40], "n_shells_above_tol": int(len(bad))})
        del P, C0, U
    L = ["## S7's preparation check, part 1 (engine_s7_prep_check_v1.py; engine spec 29.5): the copy-mixing preparations' copy-summed densities against C_0's", "",
         "- the record: %s at T/rs %.3f (the record nearest T/rs %g holding both trapping zeros; the builder's choice, labelled): r_in %.4f, R_t %.4f; C_0 per channel the negative-energy projector of the record's h_m at its applied fields (the instantaneous sea stands for the state in this check of the construction); the window chi_X(r) = (1/2) erfc((r - X)/w), w %g sites; the copy rotation exp(+-i theta chi_X sigma_y) per site on both components." % (tag, T / rs, a.t, rin, Rt, a.width), "",
         "| kappa | window (X) | theta | sign | max |d eps|/max|eps| away from the transition | max |d J|/max|J| away | max over all sites (eps, J) | shells above %g (count; first radii) |" % a.tol, "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| %d | %s (%.3f) | %.4f | %+d | %s | %s | %.2e, %.2e | %d; %s |" % (r["kappa"], r["window"], r["X"], r["theta"], r["sign"], ("%.2e" % r["max_rel_eps_far"]) if r["max_rel_eps_far"] is not None else "-",
                 ("%.2e" % r["max_rel_J_far"]) if r["max_rel_J_far"] is not None else "-", r["max_rel_eps_all"], r["max_rel_J_all"], r["n_shells_above_tol"], ", ".join("%.0f" % x for x in r["shells_above_tol"][:12])))
    L += ["", "- every number above is printed by the script; the propagation of both preparations on the member's history with the two-copy block and D_det (part 2) is not in this version."]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump({"script": "engine_s7_prep_check_v1.py", "run": a.run, "T_over_rs": T / rs, "r_in": rin, "R_t": Rt, "rows": rows}, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
