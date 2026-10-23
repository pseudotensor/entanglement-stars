#!/usr/bin/env python
"""engine_clocked_commutator_v2.py (builder analyst; version 2 = version 1 with the shift's declared counterpart: version 1 found the shift's
term of order one (1.2 of the largest main term on the (0.2, 1/4) member) and printed it without one, so its "all extras" compared the
whole shift term with the demanded force. Here D_shift = -v_f J_f - sum_{s<=n} v'_s J_s (J_s the shell momentum sum f Re <phi| (1/2){P_s, Q}
|phi>, J_f the mean of the face's two shells, v' the centred difference): the continuum -L_v of the densitized radial momentum that the
term (1/2){v, Q} of h_m generates (a classical H = v(x) p gives d_t J = -d_x(v J) - v' J; on flat slices the K S_i term of the
normal-observer law is inside it), the builder's operational definition like the other three; the extras and the declared balance use
it; f_req's two parts are printed apart. Version 1's docstring: engine spec 24.2: "then R27a's test on the regime member at ell 4 and 8: the
stationarity of the member's density matrix under the applied and the diagnostic fields ([h_m, D], the shell rates), and the full clocked
identity <i[h_m, Q_in]> against the normal-observer law, with the corrections the measured brackets demand, delta S_r = B_tt/(4 pi g N)
and delta S_t = (B_rr + B_tt)/(8 pi g N), compared nodewise with the clocked commutator's terms"). A converged static member (engine_star_v7's
record at its own ell) rebuilt as in engine_rows_static_v2 (the occupied columns at the member's applied site fields; its reference is the
ground state at those fields, so the pairs vanish and D is the dust's), per channel with the weight 16 kappa.
(1) Stationarity. Per channel h_m = (1/2){N, h} + (1/2){v, Q} under three site-field choices: (a) the member's applied fields (stored);
(b) the fields re-solved from the rebuilt shell totals (engine_fields_v14) and averaged onto the sites by the production's kernels
(site_fields_kernel_average: the applied fields' fixed point); (c) the same solution point-sampled at the sites (the dense N(r), v(r) the
rows' diagnostic uses). Printed: ||[h_m, D]||_F/||h_m D||_F (the largest over the channels) and the shell rates d<A_n>/dt = Tr[D i[h_m, A_n]]
for the charge P_n, the energy (1/2){P_n, h} and the momentum (1/2){P_n, Q}, each's largest magnitude inside 0.9 R_surf relative to the
largest shell value there; the momentum rate under (c), smoothed at ell (K_0 of the shell totals), is set beside the required force density
of (3).
(2) The clocked identity under (a). With Q_in^n = (1/2){P_<=n, Q} and the exact split i[(1/2){N, X}, Y] = (1/2){N, i[X, Y]} +
(1/2){i[N, Y], X}: d<Q_in^n>/dt = T_bond + T_site + T_grad + T_shift with T_bond = Tr[D (1/2){N, i[h_bond, Q_in^n]}] (the clocked face
flux), T_site = Tr[D (1/2){N, i[h_site, Q_in^n]}] (the clocked centrifugal force), T_grad = Tr[D (1/2){i[N, Q_in^n], h}] (the lapse
gradient's force) and T_shift = Tr[D i[(1/2){v, Q}, Q_in^n]] (the shift's), each by <a| i[X, (1/2){P, Y}] |b> = (i/2) sum_{s in P}
[(Xa)^* (Yb) + (YXa)^* b - (Ya)^* (Xb) - a^* (YXb)]_s (the two components of a site summed; a = N phi for the symmetrized products,
Tr[D (1/2){N, Z}] = sum f Re <N phi| Z |phi>); their sum against the direct Tr[D i[h_m, Q_in^n]] (the split's check; zero for a
stationary state to the eigen-residual). The declared stress's counterparts (the builder's operational definitions: the lane's dictionary
at unit clock and zero flow inserted with the member's site fields): D_bond = -N_f F_{n+1/2} (N_f the mean of the face's two sites, F the
Noether flux of engine_stress_v2), D_site = sum_{s<=n} N_s f_s, D_grad = -sum_{s<=n} N'_s eps_s (N' the centred difference of the site
lapse, eps the shell energy of h); the shift has no counterpart at zero flow (its term is printed as measured). Per shell the increments
(the force on the shell) and the extras e_X = clocked minus declared.
(3) The comparison. The rows (engine_rows_v2.brackets, h = ell/32, the stresses of engine_stress_v2 smoothed at the member's ell) at
nodes from 2 ell to 0.9 R_surf in steps of ell/4: the brackets B_rr, B_tt and the corrections they demand at fixed geometry and density
(the rows' sources are 4 pi g N (rho + S_r - 2 S_t) and 4 pi g N (rho - S_r)): delta S_r = B_tt/(4 pi g N), delta S_t = (B_rr + B_tt)/
(8 pi g N). The clocked extras as stresses: delta S_r^clk at the faces = -(T_bond - D_bond)/(4 pi R_f^2 N_f), at the shells the mean of
their faces (shell 0: its outer face), delta S_t^clk = e_site/(8 pi R N), smoothed by engine_stress_v2.smooth_stress; as force densities:
f_req = -(1/r^2) d(r^2 N delta S_r)/dr + 2 N delta S_t/r (np.gradient on the nodes) against K_0 @ (e_bond + e_site) and against the total
K_0 @ (e_bond + e_site + e_grad + e_shift). For each pair: the correlation coefficient, the ratio of the RMS norms (clocked over required)
and the RMS of the difference over the RMS of the required, over the nodes; a node table every --stride nodes. Args: --family json --NB
value --md; CPU (torch on the host for the stress parts)."""
import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_rows_static_v2 as RS2  # noqa: E402
import engine_star_v6 as S6  # noqa: E402
import engine_fields_v14 as F14  # noqa: E402
import engine_stress_v2 as STR2  # noqa: E402
import engine_rows_v2 as RW  # noqa: E402
from engine_ops_v1 import sources_static  # noqa: E402


def site_sum(X):
    return X.reshape(-1, 2, X.shape[-1]).sum(axis=1)


def site_terms(X, Y, a, b, f):
    """per site s: sum_j f_j <a_j| i[X, (1/2){P_s, Y}] |b_j> (Y None: the identity, i.e. i[X, P_s]); complex."""
    Xa = X @ a; Xb = X @ b
    if Y is None:
        Yb, YXa, Ya, YXb = b, Xa, a, Xb
    else:
        Yb = Y @ b; YXa = Y @ Xa; Ya = Y @ a; YXb = Y @ Xb
    t = 0.5j * (np.conj(Xa) * Yb + np.conj(YXa) * b - np.conj(Ya) * Xb - np.conj(a) * YXb)
    return site_sum(t) @ f


def comm_norm_rel(H, Phi, f):
    """||[H, D]||_F / ||H D||_F for D = Phi diag(f) Phi^dag (Gram form)."""
    Psi = H @ Phi
    B = np.concatenate([Psi, Phi], axis=1)
    G = B.conj().T @ B
    k = Phi.shape[1]
    F = np.diag(f.astype(complex))
    M = np.zeros((2 * k, 2 * k), complex); M[:k, k:] = F; M[k:, :k] = -F
    num = float(np.real(np.trace(M.conj().T @ G @ M @ G)))
    den = float(np.real(np.trace(F @ (Psi.conj().T @ Psi) @ F @ (Phi.conj().T @ Phi))))
    return math.sqrt(max(num, 0.0)) / max(math.sqrt(max(den, 0.0)), 1e-300)


def corr_stats(x, y):
    """correlation coefficient, RMS(x)/RMS(y), RMS(x - y)/RMS(y)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ry = math.sqrt(float(np.mean(y * y))); rx = math.sqrt(float(np.mean(x * x)))
    c = float(np.corrcoef(x, y)[0, 1]) if rx > 0 and ry > 0 else float("nan")
    return c, rx / max(ry, 1e-300), math.sqrt(float(np.mean((x - y) ** 2))) / max(ry, 1e-300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--NB", type=float, required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--stride", type=int, default=0, help="node table stride (0: about 60 lines)")
    a = ap.parse_args()
    import torch
    t0 = time.time()
    Jf = json.load(open(a.family))
    mem = [q for q in Jf["members"] if not q.get("failed") and abs(q["N_B"] - a.NB) < 1e-3][0]
    m = float(mem["m"]); g = float(mem["g"]); ell = float(mem["ell"]); nsites = int(mem["nsites"])
    Na = np.asarray(mem["N"], float); va = np.asarray(mem["v"], float)
    R, chans = RS2.orbitals(Na, va, m, nsites, mem["N_B"], bool(Jf["args"].get("half_offset", 0)))
    Rs = float(mem["R_surf"])
    eps = np.zeros(nsites); J = np.zeros(nsites)
    P = None
    for kappa, h, Q, U, f in chans:
        w = 16.0 * kappa
        e1, J1 = sources_static(h, Q, U, f, w)
        eps += e1; J += J1
        ch = SimpleNamespace(hs=h.tocsr(), Qs=Q.tocsr(), kappa=kappa, w=w, R=R)
        cs = STR2.ChanStress(ch, torch, torch.device("cpu"))
        p = STR2.stress_parts(cs, torch.as_tensor(np.asarray(U), dtype=torch.complex128), f)
        P = {k: w * np.asarray(x, float) for k, x in p.items()} if P is None else {k: P[k] + w * np.asarray(p[k], float) for k in P}
    SV = STR2.shell_values(P, R, m)
    es = np.asarray(mem["eps_tot"], float)
    d_e = float(np.max(np.abs(eps - es))) / float(np.max(np.abs(es)))
    _, _, _, info = F14.solve_constraints_ivp(eps, J, g, R, ell, collar_policy="flag")
    Nb, vb, _ = F14.site_fields_kernel_average(info, R, ell)
    Nc = np.asarray(info["N_dense"](R), float); vc = np.asarray(info["v_dense"](R), float)
    inside = R < 0.9 * Rs
    fields = {"(a) applied (stored)": (Na, va), "(b) re-solved, kernel-averaged": (Nb, vb), "(c) re-solved, point-sampled (the diagnostic's)": (Nc, vc)}
    L = ["## The clocked-commutator test on a static member (engine_clocked_commutator_v2.py)", "",
         "Member: %s, N_B %.3f, E %.6f, R_surf %.2f (m a %g, g %g, ell %g, %d sites, %d channels occupied: kappa %s); the rebuilt shell energies against the stored ones %.2e (relative); "
         "the site fields (b) and (c) against (a): max |N_b - N_a| %.2e, |v_b - v_a| %.2e, |N_c - N_a| %.2e, |v_c - v_a| %.2e." % (
             os.path.basename(a.family), mem["N_B"], mem["E"], Rs, m, g, ell, nsites, len(chans), ",".join(str(c[0]) for c in chans), d_e,
             float(np.max(np.abs(Nb - Na))), float(np.max(np.abs(vb - va))), float(np.max(np.abs(Nc - Na))), float(np.max(np.abs(vc - va)))), "",
         "### (1) Stationarity of D under the three field choices", "",
         "| fields | max over channels of the commutator norm ratio | charge rate / max shell charge | energy rate / max shell energy | momentum rate / max shell energy |", "|---|---|---|---|---|"]
    rates_c = None
    sh_charge = np.zeros(nsites); sh_energy = np.zeros(nsites)
    for kappa, h, Q, U, f in chans:
        w = 16.0 * kappa; Phi = np.asarray(U)
        sh_charge += w * site_sum((np.conj(Phi) * Phi).real) @ f
        sh_energy += w * site_sum((np.conj(Phi) * (h @ Phi)).real) @ f
    for lab, (Nf_, vf_) in fields.items():
        cmax = 0.0
        rq = np.zeros(nsites); re_ = np.zeros(nsites); rp = np.zeros(nsites)
        for kappa, h, Q, U, f in chans:
            w = 16.0 * kappa; Phi = np.asarray(U)
            H = S6.clocked_sparse(h, Q, Nf_, vf_).tocsr()
            cmax = max(cmax, comm_norm_rel(H, Phi, f))
            rq += w * np.real(site_terms(H, None, Phi, Phi, f))
            re_ += w * np.real(site_terms(H, h.tocsr(), Phi, Phi, f))
            rp += w * np.real(site_terms(H, Q.tocsr(), Phi, Phi, f))
        sc = float(np.max(np.abs(sh_charge[inside]))); se = float(np.max(np.abs(sh_energy[inside])))
        L.append("| %s | %.2e | %.2e | %.2e | %.2e |" % (lab, cmax, float(np.max(np.abs(rq[inside]))) / sc, float(np.max(np.abs(re_[inside]))) / se,
                                                    float(np.max(np.abs(rp[inside]))) / se))
        if lab.startswith("(c)"):
            rates_c = rp
    # (2) the clocked identity under (a)
    Tb = np.zeros(nsites, complex); Ts = np.zeros(nsites, complex); Tg = np.zeros(nsites, complex); Tv = np.zeros(nsites, complex); Tt = np.zeros(nsites, complex)
    for kappa, h, Q, U, f in chans:
        w = 16.0 * kappa; Phi = np.asarray(U)
        hsite, hbond = STR2.split_chain(h)
        Qc = Q.tocsr(); hc = h.tocsr()
        N2 = np.repeat(Na, 2)[:, None]; NPhi = N2 * Phi
        Nd = sp.diags(np.repeat(Na, 2)).tocsr(); Vd = sp.diags(np.repeat(va, 2)).tocsr()
        iNQ = 1j * (Nd @ Qc - Qc @ Nd)                                     # i[N, Q]
        Vgen = (0.5 * (Vd @ Qc + Qc @ Vd)).tocsr()
        Hm = S6.clocked_sparse(h, Q, Na, va).tocsr()
        Tb += w * np.cumsum(site_terms(hbond, Qc, NPhi, Phi, f))
        Ts += w * np.cumsum(site_terms(hsite, Qc, NPhi, Phi, f))
        # T_grad = sum f Re <h phi| i[N, Q_in^n] |phi>, i[N, Q_in^n] = (1/2){P_<=n, i[N, Q]}: per site (1/2)[(h phi)^* (iNQ phi) + (iNQ h phi)^* phi]
        hPhi = hc @ Phi
        tg = 0.5 * (np.conj(hPhi) * (iNQ @ Phi) + np.conj(iNQ @ hPhi) * Phi)
        Tg += w * np.cumsum(site_sum(tg) @ f)
        Tv += w * np.cumsum(site_terms(Vgen, Qc, Phi, Phi, f))
        Tt += w * np.cumsum(site_terms(Hm, Qc, Phi, Phi, f))
    Tb, Ts, Tg, Tv, Tt = (np.real(x) for x in (Tb, Ts, Tg, Tv, Tt))
    split_err = float(np.max(np.abs(Tb + Ts + Tg + Tv - Tt)))
    scaleT = max(float(np.max(np.abs(Tb[inside]))), float(np.max(np.abs(Ts[inside]))), float(np.max(np.abs(Tg[inside]))), 1e-300)
    F = SV["F_face"]
    Nf = np.concatenate([0.5 * (Na[:-1] + Na[1:]), [Na[-1]]])
    Rf = SV["R_face"]
    fsite = P["fc"]
    Np = np.gradient(Na, R)
    Db = -Nf * F
    Ds = np.cumsum(Na * fsite)
    Dg = -np.cumsum(Np * eps)
    Jsh = P["J"]
    Jf_ = np.concatenate([0.5 * (Jsh[:-1] + Jsh[1:]), [Jsh[-1]]])
    vf_face = np.concatenate([0.5 * (va[:-1] + va[1:]), [va[-1]]])
    vp = np.gradient(va, R)
    Dv = -vf_face * Jf_ - np.cumsum(vp * Jsh)
    inc = lambda X: np.diff(np.concatenate([[0.0], X]))
    e_b = inc(Tb - Db); e_s = inc(Ts) - Na * fsite; e_g = inc(Tg) + Np * eps; e_v = inc(Tv - Dv)
    L += ["", "### (2) The clocked identity under (a): d<Q_in^n>/dt split exactly", "",
          "- the split's check: max |T_bond + T_site + T_grad + T_shift - Tr[D i[h_m, Q_in^n]]| = %.2e; the total itself (stationarity), max over n inside 0.9 R_surf: %.2e; both relative to the largest of the three main terms there (%.4e): %.2e, %.2e." % (
              split_err, float(np.max(np.abs(Tt[inside]))), scaleT, split_err / scaleT, float(np.max(np.abs(Tt[inside]))) / scaleT),
          "- clocked against declared, the largest |difference| inside 0.9 R_surf relative to the largest declared term of its kind: bond %.2e, site %.2e, gradient %.2e, shift %.2e (the shift's term itself %.2e of the largest main term)." % (
              float(np.max(np.abs((Tb - Db)[inside]))) / max(float(np.max(np.abs(Db[inside]))), 1e-300),
              float(np.max(np.abs((Ts - Ds)[inside]))) / max(float(np.max(np.abs(Ds[inside]))), 1e-300),
              float(np.max(np.abs((Tg - Dg)[inside]))) / max(float(np.max(np.abs(Dg[inside]))), 1e-300),
              float(np.max(np.abs((Tv - Dv)[inside]))) / max(float(np.max(np.abs(Dv[inside]))), 1e-300),
              float(np.max(np.abs(Tv[inside]))) / scaleT),
          "- the declared balance (the lane's dictionary in the normal-observer law with the member's fields): max |D_bond + D_site + D_grad + D_shift| inside 0.9 R_surf relative to the largest term: %.2e (without D_shift: %.2e)." % (
              float(np.max(np.abs((Db + Ds + Dg + Dv)[inside]))) / scaleT, float(np.max(np.abs((Db + Ds + Dg)[inside]))) / scaleT)]
    # (3) the comparison with the rows' brackets
    rr = np.arange(2.0 * ell, 0.9 * Rs, ell / 4.0)
    Sr_l, St_l, trerr = STR2.smooth_stress(SV["S_r"], SV["S_t"], R, ell, rr)
    rho = np.asarray(info["rho_ell"](rr), float)
    b = RW.brackets(info["N_dense"], info["v_dense"], rho, Sr_l, St_l, g, rr, ell / 32.0)
    Nn = b["N"]
    dSr_req = b["rhs_tt"] / (4.0 * math.pi * g * Nn)
    dSt_req = (b["rhs_rr"] + b["rhs_tt"]) / (8.0 * math.pi * g * Nn)
    f_req_r = -np.gradient(rr ** 2 * Nn * dSr_req, rr) / rr ** 2; f_req_t = 2.0 * Nn * dSt_req / rr
    f_req = f_req_r + f_req_t
    dSr_face = -(Tb - Db) / (4.0 * math.pi * Rf ** 2 * Nf)
    dSr_sh = 0.5 * (np.concatenate([[dSr_face[0]], dSr_face[:-1]]) + dSr_face)
    dSt_sh = e_s / (8.0 * math.pi * R * Na)
    dSr_clk, dSt_clk, _ = STR2.smooth_stress(dSr_sh, dSt_sh, R, ell, rr)
    K0 = STR2.kernels_l(rr, R, ell, 0)
    f_bs = K0 @ (e_b + e_s); f_tot = K0 @ (e_b + e_s + e_g + e_v); f_rate = K0 @ rates_c if rates_c is not None else np.zeros_like(rr)
    rel_rr = float(np.max(np.abs(b["rhs_rr"]) / b["terms_rr"])); rel_tt = float(np.max(np.abs(b["rhs_tt"]) / b["terms_tt"]))
    L += ["", "### (3) The brackets' demanded corrections against the clocked extras (nodes 2 ell to 0.9 R_surf, step ell/4)", "",
          "- the rows here (stresses of engine_stress_v2): max |B_rr|/max term %.2e, max |B_tt|/max term %.2e; the trace identity of the smoothing %.1e." % (rel_rr, rel_tt, trerr),
          "- f_req's two parts over the nodes: RMS of -(1/r^2) d(r^2 N delta S_r)/dr %.3e, of 2 N delta S_t/r %.3e, of their sum f_req %.3e." % (
              math.sqrt(float(np.mean(f_req_r ** 2))), math.sqrt(float(np.mean(f_req_t ** 2))), math.sqrt(float(np.mean(f_req ** 2)))), "",
          "| pair (clocked, required) | correlation | RMS ratio (clocked/required) | RMS(difference)/RMS(required) |", "|---|---|---|---|"]
    for lab, x, y in (("delta S_r", dSr_clk, dSr_req), ("delta S_t", dSt_clk, dSt_req), ("force: bond + site extras against f_req", f_bs, f_req),
                      ("force: all four extras against f_req", f_tot, f_req), ("force: the momentum rate under (c) against f_req", f_rate, f_req)):
        c, rq_, rd = corr_stats(x, y)
        L.append("| %s | %+.3f | %.3e | %.3e |" % (lab, c, rq_, rd))
    st = a.stride if a.stride > 0 else max(1, len(rr) // 60)
    L += ["", "| r | B_rr | B_tt | delta S_r req | delta S_r clk | delta S_t req | delta S_t clk | f_req | bond + site extra | all extras | rate (c) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i in range(0, len(rr), st):
        L.append("| %.2f | %+.3e | %+.3e | %+.3e | %+.3e | %+.3e | %+.3e | %+.3e | %+.3e | %+.3e | %+.3e |" % (
            rr[i], b["rhs_rr"][i], b["rhs_tt"][i], dSr_req[i], dSr_clk[i], dSt_req[i], dSt_clk[i], f_req[i], f_bs[i], f_tot[i], f_rate[i]))
    L += ["", "- The declared counterparts are the builder's operational definitions (stated in the docstring) for the lead and the red team to check; every number above is printed by the script (%.0f s)." % (time.time() - t0)]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
