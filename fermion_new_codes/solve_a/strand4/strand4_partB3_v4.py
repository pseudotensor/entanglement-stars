#!/usr/bin/env python
"""Strand 4 Part B.3 to B.5, version 4 (builder analyst): version 3 with h = 1e-7 added (the 0.05 members, N_s = 1800, were not
converged at (1e-5, 1e-6): their D_mat steps agreed to 2e-3 against 2e-6 to 5e-5 for the other states); the reported values are the
(1e-6, 1e-7) Richardson pair, every pair kept.  Writes partB3_v4.json.  Version 3's description follows.
Strand 4 Part B.3 to B.5, version 3 (builder analyst): version 2 with the finite-difference steps h in {1e-3, 1e-4, 1e-5, 1e-6}
and Richardson over consecutive pairs (the reported values from (1e-5, 1e-6)).  Version 2's two steps disagreed by up to 50 percent in
D_mat at the 0.1 maximum because the dilation moves the shell at n by n h grid spacings (0.8 at h = 1e-3, N_s = 800), so the transported
orbitals' central differences carry an error of order (n h)^2 there; the fields-only part agreed to 1e-6.  Writes partB3_v3.json.
Version 2's description follows.
Strand 4 Part B.3 to B.5, version 2 (spec solve/specs/SPEC_strand4_controls.md with the lead's "Ruling on the Part B identity",
07:13 UTC; builder analyst): the discrete dilation family on the stored executed-star states and the pieces of the identity
D_lat = (A - B)/G_0 = Total - R_stat = [D_vac + S_inf] + D_mat - R_stat.  [paper, control]

States (the paper lane's B20-03c records, unmodified copies with md5s): the m_* a = 0.1 golden-section maximum (dev29); the dilute
member j = 0 of the 0.1 family; the members j = 9, 10, 11 of the 0.2 family and j = 1, 2, 3 of the 0.05 family (dev28: "the
m_* = 0.2 maximum +-1" and "the m_* = 0.05 maximum +-1" of the macro families; no golden-section maximum is stored for 0.2 or 0.05),
all in the (lamR, lam2) = (1, 1) combination.  The chain is built per family as dev28 builds it: make_chain(Ns, m_*, G, T_s, tables
p = -2 and response of that family, U'_B, U'_M of that family's flat matching, lamR = lam2 = 1), with the stored self-energy, and
evaluated at the stored (psi, chi, N_B).
Per state, B.1-type controls: the stored rows E, D against the re-evaluated ones (where stored); A = Ns (psi_Ns - 1), B = -Ns
(chi_Ns - 1), A/B - 1, (A - B)/G; E_exc,ret = F + T_s S.
Family (lam = 1 +- h, h in {1e-3, 1e-4}): as version 1 (n_lam(r_n) = n(lam r_n), s_lam(r_n) = lam^(1/2) s(lam r_n) by cubic splines
with the exact exterior and the even continuation; orbitals transported per component and orthonormalized per channel by QR).
Transport checks per lam: the nonzero spectrum of the transported X per channel (the eigenvalues of diag(f^(1/2)) Q^dag Q
diag(f^(1/2))) against the occupations f (largest difference), max |Q^dag Q - 1|, and the changes of N_B and S_occ.
E_mat = sum md sz + sum Qc Z + sum Wc ksy + E2 (as version 1).  Version 2 adds: E2 split into its two quartic channels by coupling,
E2 = E2(U'_M, 0) + E2(0, U'_B) (exact: E2 is linear in (U_M, U_B) at fixed covariance and fixed response table; the split's sum is
checked); the dressing tables' shares separately: mu_L frozen at 1 (enters the mass term), kappa_L frozen at 1 (kaL and kabL: the
angular and radial terms), and both.
Pieces (central differences in lam, Richardson (100 J(1e-4) - J(1e-3))/99):
  D_vac = d F_g(lam; 1)/dlam and the boundary term's share; D_mat by term (mass, radial, angular, E2 by channel) and the tables'
  shares; S_inf two ways: the closed form (k/2) 1^T K [(s - 1) + (chi - 1)] and the central difference d F_g(fields at 1; lam^(1/2))/dlam;
  R_stat = d/dlam [F_g(lam; 1) + E_mat(lam)] directly (and D_vac + D_mat beside it); Total = d/dlam [F_g(lam; lam^(1/2)) + E_mat(lam)]
  directly; the closures Total - (D_vac + D_mat + S_inf) and (Total - R_stat) - (A - B)/G.
Writes <out>/partB3_v2.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline

TERMS = ("mass", "radial", "angular", "E2_UM", "E2_UB")
STEPS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma list of state labels' prefixes to run (default all)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, os.path.join(a.inputs, "inputs"))
    sys.path.insert(0, a.inputs)
    import b20_03c_chain_v5 as BC
    t0 = time.time()
    FLAT = json.load(open(os.path.join(a.inputs, "dev", "dev05_stage0_flat.json")))
    TABZ = np.load(os.path.join(a.inputs, "dev", "dev05_stage0_tables.npz"))
    RESP = np.load(os.path.join(a.inputs, "dev", "dev10_resp_tables.npz"))
    Z = np.load(os.path.join(a.inputs, "inputs", "lattice_star_macro.npz"))
    G = float(Z["G"][23])
    Ts = float(Z["Ts"][23])
    MX = json.load(open(os.path.join(a.inputs, "dev", "dev29_stage2_max.json")))
    PMX = np.load(os.path.join(a.inputs, "dev", "dev29_stage2_max_profiles.npz"))
    S2 = json.load(open(os.path.join(a.inputs, "dev", "dev28_stage2.json")))
    PR = np.load(os.path.join(a.inputs, "dev", "dev28_stage2_profiles.npz"))
    states = []
    best = MX["best"]
    key = "Nc%.9f_" % best["Nc"]
    states.append(("maximum m*a 0.1", "0.1", {k[len(key):]: PMX[k] for k in PMX.files if k.startswith(key)}, best["rec"]["NB"], None))
    for fam, js, lab in (("0.1", (0,), "dilute member j = 0"), ("0.2", (9, 10, 11), None), ("0.05", (1, 2, 3), None)):
        for mem in S2["members"]:
            if mem["fam"] == fam and mem["j"] in js:
                k2 = "%s_%02d_1_1_" % (fam, mem["j"])
                P = {k[len(k2):]: PR[k] for k in PR.files if k.startswith(k2)}
                sig = {s: P["sig_" + s] for s in ("dm", "dQ", "dW")} if "sig_dm" in P else None
                label = lab or "m*a %s member j = %d (N_c %.4f)" % (fam, mem["j"], mem["Nc_target"])
                states.append((label, fam, P, mem["combos"]["1,1"]["NB"], sig))
    if a.only:
        pre = [s.strip() for s in a.only.split(",")]
        states = [s for s in states if any(s[0].startswith(p) for p in pre)]
    k = 1 / (2 * np.pi * G)
    res = {"meta": {"G": G, "Ts": Ts, "states": [s[0] for s in states]}, "states": []}
    for label, fam, P, NB, sig in states:
        ts = time.time()
        TAB = BC.Tables(TABZ["%s_p-2" % fam], RESP[fam])
        REC = FLAT["matching"][fam]
        psi, chi = np.asarray(P["psi"], float), np.asarray(P["chi"], float)
        Ns = len(psi)
        n = np.arange(1, Ns + 1, dtype=float)
        ch = BC.make_chain(Ns, float(fam), G, Ts, TAB, REC["UBp"], REC["UMp"], lamR=1.0, lam2=1.0)
        ch.sig = None if sig is None else {kk: vv.copy() for kk, vv in sig.items()}
        F, ev, parts = ch.free_energy(psi, chi, NB)
        Eexc_B1 = F + parts["TS"]
        rows = ev["rows"]
        lev, f = ev["lev"], np.asarray(ev["f"])
        occ = [(t, fj) for t, fj in zip(lev, f) if fj > 1e-15]
        A = Ns * (psi[-1] - 1)
        B = -Ns * (chi[-1] - 1)
        Gr = 1.0 / np.maximum(n[:, None], n[None, :])
        Kmat = 4 * np.pi * np.linalg.inv(Gr)
        out = {"state": label, "family": fam, "Ns": Ns, "NB": NB, "UBp": REC["UBp"], "UMp": REC["UMp"], "A": A, "B": B, "A_over_B_minus_1": A / B - 1,
               "A_minus_B_over_G": (A - B) / G, "E_exc_ret_B1": Eexc_B1, "n_occupied": len(occ), "steps": {}}
        if "E" in P and "D" in P:
            out["E_rows_max_abs_diff"] = float(np.max(np.abs(rows["E"] - P["E"])))
            out["D_rows_max_abs_diff"] = float(np.max(np.abs(rows["D"] - P["D"])))
            out["E_rows_max_abs"] = float(np.max(np.abs(P["E"])))
        xs_in = np.concatenate([-n[:3][::-1], n])
        ext = np.arange(Ns + 1, Ns + 6, dtype=float)

        def spline(vals, tail):
            x = np.concatenate([xs_in, ext])
            y = np.concatenate([vals[:3][::-1], vals, tail(ext)])
            return CubicSpline(x, y)
        sp_psi = spline(psi, lambda r: 1 + A / r)
        sp_chi = spline(chi, lambda r: 1 - B / r)
        xs_o = np.concatenate([[0.0], n, ext])
        orb = []
        for (e, kap, v), fj in occ:
            u = np.asarray(v).reshape(Ns, 2)
            sps = []
            for c in range(2):
                yr = np.concatenate([[0.0], u[:, c].real, np.zeros(len(ext))])
                yi = np.concatenate([[0.0], u[:, c].imag, np.zeros(len(ext))])
                sps.append((CubicSpline(xs_o, yr), CubicSpline(xs_o, yi)))
            orb.append((kap, fj, sps, u))

        def fields(lam):
            r = lam * n
            return np.sqrt(lam) * sp_psi(r), np.sqrt(lam) * sp_chi(r)

        def covariance(lam):
            r = lam * n
            by_k = {}
            for kap, fj, sps, u in orb:
                ul = u.copy() if lam == 1.0 else np.stack([sps[c][0](r) + 1j * sps[c][1](r) for c in range(2)], axis=1)
                by_k.setdefault(kap, []).append((fj, ul.reshape(-1)))
            sz = np.zeros(Ns)
            Zb = np.zeros(Ns - 1)
            ksy = np.zeros(Ns)
            NBl, Socc, spec_dev, orth_dev = 0.0, 0.0, 0.0, 0.0
            for kap, lst in by_k.items():
                M = np.array([vv for _, vv in lst]).T
                Q, Rq = np.linalg.qr(M)
                Q = Q * np.sign(np.real(np.diag(Rq)))[None, :]
                fs = np.array([fj for fj, _ in lst])
                Gm = Q.conj().T @ Q
                orth_dev = max(orth_dev, float(np.max(np.abs(Gm - np.eye(len(fs))))))
                sq = np.sqrt(fs)
                spec = np.sort(np.linalg.eigvalsh(0.5 * ((sq[:, None] * Gm * sq[None, :]) + (sq[:, None] * Gm * sq[None, :]).conj().T)))
                spec_dev = max(spec_dev, float(np.max(np.abs(spec - np.sort(fs)))))
                w = ch.w[kap]
                for (fj, _), col in zip(lst, Q.T):
                    uu = col.reshape(Ns, 2)
                    wf = w * fj
                    zz = -0.5j * (np.conj(uu[:-1, 0]) * uu[1:, 1] + np.conj(uu[:-1, 1]) * uu[1:, 0])
                    Zb += wf * 2.0 * zz.real
                    sz += wf * (np.abs(uu[:, 0]) ** 2 - np.abs(uu[:, 1]) ** 2)
                    ksy += wf * kap * 2.0 * (np.conj(uu[:, 0]) * uu[:, 1]).imag
                    NBl += wf
                    if 0 < fj < 1:
                        Socc += -w * (fj * np.log(fj) + (1 - fj) * np.log(1 - fj))
            return sz, Zb, ksy, NBl, Socc, spec_dev, orth_dev

        def e2_split(N, s, sz, Zb, ksy):
            UMp, UBp = ch.UMp, ch.UBp
            full = ch.e2_terms(N, s, sz, Zb, ksy)["E2"]
            try:
                ch.UBp = 0.0
                eM = ch.e2_terms(N, s, sz, Zb, ksy)["E2"]
                ch.UBp, ch.UMp = UBp, 0.0
                eB = ch.e2_terms(N, s, sz, Zb, ksy)["E2"]
            finally:
                ch.UMp, ch.UBp = UMp, UBp
            return float(full), float(eM), float(eB)

        def Emat_terms(N, s, cov, freeze=()):
            sz, Zb, ksy = cov[:3]
            c = ch.fac(N, s)
            muL = np.ones_like(s) if "mu" in freeze else c["muL"]
            kaL = np.ones_like(s) if "ka" in freeze else c["kaL"]
            kabL = np.ones(Ns - 1) if "ka" in freeze else c["kabL"]
            md = ch.mstar * N * muL
            Qc = c["Q"] * kabL
            Wc = c["W"] * kaL
            e2, eM, eB = e2_split(N, s, sz, Zb, ksy)
            return {"mass": float(np.sum(md * sz)), "radial": float(np.sum(Qc * Zb)), "angular": float(np.sum(Wc * ksy)),
                    "E2_UM": eM, "E2_UB": eB, "E2_split_closure": e2 - eM - eB}

        def Emat(T):
            return sum(T[t] for t in TERMS)

        def Fg(s, c, sigma):
            return -k * float((s - sigma) @ (Kmat @ (c - sigma)))

        def Bext(s, c):
            return -k * 4 * np.pi * Ns * (s[-1] - 1) * (c[-1] - 1)
        cov1 = covariance(1.0)
        s1, c1 = fields(1.0)
        N1 = c1 / s1
        T1 = Emat_terms(N1, s1, cov1)
        out.update({"E_mat_lam1": Emat(T1), "E_mat_lam1_terms": T1, "E_mat_minus_E_exc": Emat(T1) - Eexc_B1,
                    "fields_lam1_max_dev": float(max(np.max(np.abs(s1 - psi)), np.max(np.abs(c1 - chi)))),
                    "transport_lam1": {"spectrum_dev": cov1[5], "orth_dev": cov1[6]}})
        S_closed = 0.5 * k * float(np.sum(Kmat @ ((psi - 1) + (chi - 1))))
        out["S_inf_closed"] = S_closed
        for h in STEPS:
            vals = {}
            for sgn in (1, -1):
                lam = 1 + sgn * h
                s, c = fields(lam)
                N = c / s
                cov = covariance(lam)
                vals[sgn] = {"Fg1": Fg(s, c, 1.0), "Fgsig": Fg(s, c, np.sqrt(lam)), "Fg_sig_only": Fg(s1, c1, np.sqrt(lam)), "Bext": Bext(s, c),
                             "T": Emat_terms(N, s, cov), "Tmu": Emat_terms(N, s, cov, ("mu",)), "Tka": Emat_terms(N, s, cov, ("ka",)),
                             "Tboth": Emat_terms(N, s, cov, ("mu", "ka")), "Tcf": Emat_terms(N, s, cov1),
                             "NB": cov[3], "Socc": cov[4], "spec_dev": cov[5], "orth_dev": cov[6]}
            d = lambda f_: (f_(vals[1]) - f_(vals[-1])) / (2 * h)
            st = {"D_vac": d(lambda v: v["Fg1"]), "D_boundary_term": d(lambda v: v["Bext"]),
                  "D_mat_terms": {t: d(lambda v, t=t: v["T"][t]) for t in TERMS},
                  "share_mu_L": {t: d(lambda v, t=t: v["T"][t] - v["Tmu"][t]) for t in TERMS},
                  "share_kappa_L": {t: d(lambda v, t=t: v["T"][t] - v["Tka"][t]) for t in TERMS},
                  "share_tables_both": {t: d(lambda v, t=t: v["T"][t] - v["Tboth"][t]) for t in TERMS},
                  "D_mat_fields_only": {t: d(lambda v, t=t: v["Tcf"][t]) for t in TERMS},
                  "E2_split_closure_max": max(abs(vals[s_]["T"]["E2_split_closure"]) for s_ in (1, -1)),
                  "S_inf_fd": d(lambda v: v["Fg_sig_only"]),
                  "R_stat_direct": d(lambda v: v["Fg1"] + Emat(v["T"])),
                  "Total": d(lambda v: v["Fgsig"] + Emat(v["T"])),
                  "transport": {"NB_change": max(abs(vals[s_]["NB"] - cov1[3]) for s_ in (1, -1)),
                                "Socc_change": max(abs(vals[s_]["Socc"] - cov1[4]) for s_ in (1, -1)),
                                "spectrum_dev": max(vals[s_]["spec_dev"] for s_ in (1, -1)),
                                "orth_dev": max(vals[s_]["orth_dev"] for s_ in (1, -1))}}
            st["D_mat"] = sum(st["D_mat_terms"].values())
            st["R_stat_sum"] = st["D_vac"] + st["D_mat"]
            st["closure_Total_minus_pieces"] = st["Total"] - (st["D_vac"] + st["D_mat"] + S_closed)
            st["closure_Total_minus_R_stat_minus_Dlat"] = st["Total"] - st["R_stat_direct"] - (A - B) / G
            out["steps"][str(h)] = st
        rich = lambda a1, a2: (100 * a2 - a1) / 99

        def richardson(h1, h2):
            S1, S2_ = out["steps"][h1], out["steps"][h2]
            R = {kk: rich(S1[kk], S2_[kk]) for kk in ("D_vac", "D_boundary_term", "D_mat", "S_inf_fd", "R_stat_direct", "R_stat_sum", "Total")}
            for grp in ("D_mat_terms", "share_mu_L", "share_kappa_L", "share_tables_both", "D_mat_fields_only"):
                R[grp] = {t: rich(S1[grp][t], S2_[grp][t]) for t in TERMS}
            R["closure_Total_minus_pieces"] = R["Total"] - (R["D_vac"] + R["D_mat"] + S_closed)
            R["Total_minus_R_stat"] = R["Total"] - R["R_stat_direct"]
            R["closure_Total_minus_R_stat_minus_Dlat"] = R["Total_minus_R_stat"] - (A - B) / G
            R["D_vac_plus_S_inf"] = R["D_vac"] + S_closed
            R["tables_share_total"] = sum(R["share_tables_both"].values())
            R["step_agreement_rel"] = {kk: abs(S1[kk] - S2_[kk]) / max(abs(S2_[kk]), 1e-300) for kk in ("D_vac", "D_mat", "R_stat_direct", "Total")}
            R["step_pair"] = [h1, h2]
            return R
        keys = [str(h) for h in STEPS]
        out["richardson_pairs"] = {"%s,%s" % (keys[i], keys[i + 1]): richardson(keys[i], keys[i + 1]) for i in range(len(keys) - 1)}
        out["richardson"] = out["richardson_pairs"]["%s,%s" % (keys[-2], keys[-1])]
        out["seconds"] = time.time() - ts
        res["states"].append(out)
        print(json.dumps({kk: vv for kk, vv in out.items() if kk not in ("steps",)}, default=float), flush=True)
        with open(os.path.join(a.out, "partB3_v4.json"), "w") as fh:
            json.dump(res, fh, indent=1, default=float)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "partB3_v4.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
