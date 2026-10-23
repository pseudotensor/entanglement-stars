#!/usr/bin/env python
"""Strand 4 Part B.3 to B.5 (spec solve/specs/SPEC_strand4_controls.md; builder analyst): the discrete dilation family on the stored
executed-star states and the pieces of its identity.  [paper, control]

State: as strand4_partB1_v1 (the chain of b20_03c_chain_v5 evaluated at the stored (psi, chi, N_B); the occupied levels' vectors and
occupations f_j; channel weights w_kappa).
Family (lam = 1 +- h, h in {1e-3, 1e-4}): n_lam(r_n) = n(lam r_n), s_lam(r_n) = lam^(1/2) s(lam r_n) (so chi_lam = lam^(1/2) chi(lam r_n)),
with cubic splines of the shell values, the exact exterior s = 1 + A/r, chi = 1 - B/r beyond Ns and the even continuation at the
centre; every occupied orbital transported component-wise, u_lam(r_n) = u(lam r_n) (cubic splines, u(0) = 0, u = 0 beyond Ns), then
orthonormalized per channel by QR in the original order (so X_lam has the occupations of X: Tr X and S_occ unchanged; checked).
Energies: F_g(lam; sigma) = -k (s_lam - sigma 1)^T K (chi_lam - sigma 1), K = 4 pi Gr^-1, k = 1/(2 pi G) (n_inf = 1);
E_mat[X; n, s] = sum_n md sz + sum_b Qc Z + sum_n Wc ksy + E2 (the retained excess functional at fixed covariance: md = m_* N mu_L(psi),
Qc = (Nbar/psibar^2) kappa_L(psibar), Wc = (N/(psi^2 n)) kappa_L(psi); E2 = CorrChain.e2_terms; the covariance sums sz, Z, ksy from
the transported orbitals); at lam = 1 E_mat equals B.1's E_exc,ret (checked).
Pieces (central differences in lam at h, Richardson over the two h):
  D_vac = d F_g(lam; 1)/d lam; its boundary part d/dlam [-k 4 pi Ns (s_lam,Ns - 1)(chi_lam,Ns - 1)] beside it;
  D_mat = d E_mat/d lam, split into the mass term (and, inside it, the mu_L dressing table's share: the same with mu_L frozen at 1),
          the radial hopping (and kappa_L's share), the angular term (and kappa_L's share), and E2 (and its coupling/response share at
          fixed covariance);
  S_inf = (1/2) dF_g(1; sigma)/d sigma at sigma = 1 (the asymptotic ruler moves as lam^(1/2)), closed form (k/2) 1^T K [(s-1)+(chi-1)];
  R_stat = D_vac + D_mat (the first variation along the family at fixed sigma = 1);
  Total = d/dlam [F_g(lam; lam^(1/2)) + E_mat(lam)] computed directly, against R_stat + S_inf;
  and (A - B)/G from B.1.
Writes <out>/partB3.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, os.path.join(a.inputs, "inputs"))
    sys.path.insert(0, a.inputs)
    import b20_03c_chain_v5 as BC
    t0 = time.time()
    FLAT = json.load(open(os.path.join(a.inputs, "dev", "dev05_stage0_flat.json")))
    TABZ = np.load(os.path.join(a.inputs, "dev", "dev05_stage0_tables.npz"))
    RESP = np.load(os.path.join(a.inputs, "dev", "dev10_resp_tables.npz"))
    TAB = BC.Tables(TABZ["0.1_p-2"], RESP["0.1"])
    REC = FLAT["matching"]["0.1"]
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
    states.append(("maximum m*a 0.1", {k[len(key):]: PMX[k] for k in PMX.files if k.startswith(key)}, best["rec"]["NB"], None))
    for mem in S2["members"]:
        if mem["fam"] == "0.1" and mem["j"] == 0:
            k2 = "0.1_00_1_1_"
            P = {k[len(k2):]: PR[k] for k in PR.files if k.startswith(k2)}
            sig = {s: P["sig_" + s] for s in ("dm", "dQ", "dW")} if "sig_dm" in P else None
            states.append(("dilute member j = 0", P, mem["combos"]["1,1"]["NB"], sig))
    k = 1 / (2 * np.pi * G)
    res = {"meta": {"G": G, "Ts": Ts}, "states": []}
    for label, P, NB, sig in states:
        psi, chi = np.asarray(P["psi"], float), np.asarray(P["chi"], float)
        Ns = len(psi)
        n = np.arange(1, Ns + 1, dtype=float)
        ch = BC.make_chain(Ns, 0.1, G, Ts, TAB, REC["UBp"], REC["UMp"], lamR=1.0, lam2=1.0)
        ch.sig = None if sig is None else {kk: vv.copy() for kk, vv in sig.items()}
        F, ev, parts = ch.free_energy(psi, chi, NB)
        Eexc_B1 = F + parts["TS"]
        lev, f = ev["lev"], np.asarray(ev["f"])
        occ = [(t, fj) for t, fj in zip(lev, f) if fj > 1e-15]
        A = Ns * (psi[-1] - 1)
        B = -Ns * (chi[-1] - 1)
        Gr = 1.0 / np.maximum(n[:, None], n[None, :])
        Kmat = 4 * np.pi * np.linalg.inv(Gr)
        # field splines with the exact exterior and the even continuation
        xs_in = np.concatenate([-n[:3][::-1], n])
        ext = np.arange(Ns + 1, Ns + 6, dtype=float)

        def spline(vals, tail):
            x = np.concatenate([xs_in, ext])
            y = np.concatenate([vals[:3][::-1], vals, tail(ext)])
            return CubicSpline(x, y)
        sp_psi = spline(psi, lambda r: 1 + A / r)
        sp_chi = spline(chi, lambda r: 1 - B / r)
        # orbital splines (per occupied level, per component, real and imaginary parts), u(0) = 0, u = 0 beyond Ns
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
            s = np.sqrt(lam) * sp_psi(r)
            c = np.sqrt(lam) * sp_chi(r)
            return s, c

        def covariance(lam):
            r = lam * n
            by_k = {}
            for kap, fj, sps, u in orb:
                if lam == 1.0:
                    ul = u.copy()
                else:
                    ul = np.stack([sps[c][0](r) + 1j * sps[c][1](r) for c in range(2)], axis=1)
                by_k.setdefault(kap, []).append((fj, ul.reshape(-1)))
            sz = np.zeros(Ns)
            Zb = np.zeros(Ns - 1)
            ksy = np.zeros(Ns)
            NBl, Socc = 0.0, 0.0
            for kap, lst in by_k.items():
                M = np.array([vv for _, vv in lst]).T                  # (2 Ns, n_k)
                Q, Rq = np.linalg.qr(M)
                Q = Q * np.sign(np.real(np.diag(Rq)))[None, :]          # keep each column's orientation
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
            return sz, Zb, ksy, NBl, Socc

        def Emat_terms(N, s, cov, freeze_tables=False):
            sz, Zb, ksy = cov[:3]
            c = ch.fac(N, s)
            muL = np.ones_like(s) if freeze_tables else c["muL"]
            kaL = np.ones_like(s) if freeze_tables else c["kaL"]
            kabL = np.ones(Ns - 1) if freeze_tables else c["kabL"]
            md = ch.mstar * N * muL
            Qc = c["Q"] * kabL
            Wc = c["W"] * kaL
            e2 = ch.e2_terms(N, s, sz, Zb, ksy)["E2"]
            return {"mass": float(np.sum(md * sz)), "radial": float(np.sum(Qc * Zb)), "angular": float(np.sum(Wc * ksy)), "E2": float(e2)}

        def Fg(s, c, sigma):
            return -k * float((s - sigma) @ (Kmat @ (c - sigma)))

        def Bext(s, c):
            return -k * 4 * np.pi * Ns * (s[-1] - 1) * (c[-1] - 1)
        cov1 = covariance(1.0)
        s1, c1 = fields(1.0)
        N1 = c1 / s1
        T1 = Emat_terms(N1, s1, cov1)
        out = {"state": label, "Ns": Ns, "NB": NB, "E_exc_ret_B1": Eexc_B1, "E_mat_lam1": sum(T1.values()), "E_mat_lam1_terms": T1,
               "fields_lam1_max_dev": float(max(np.max(np.abs(s1 - psi)), np.max(np.abs(c1 - chi)))), "A": A, "B": B,
               "A_minus_B_over_G": (A - B) / G, "steps": {}}
        out["E_mat_minus_E_exc"] = out["E_mat_lam1"] - Eexc_B1
        sigma_piece = 0.5 * k * float(np.sum(Kmat @ ((psi - 1) + (chi - 1))))
        out["S_inf"] = sigma_piece
        for h in (1e-3, 1e-4):
            vals = {}
            for sgn in (1, -1):
                lam = 1 + sgn * h
                s, c = fields(lam)
                N = c / s
                cov = covariance(lam)
                T = Emat_terms(N, s, cov)
                Tfz = Emat_terms(N, s, cov, freeze_tables=True)
                Tcf = Emat_terms(N, s, cov1)                      # fields moved, covariance fixed
                vals[sgn] = {"Fg1": Fg(s, c, 1.0), "Fgsig": Fg(s, c, np.sqrt(lam)), "Bext": Bext(s, c), "T": T, "Tfz": Tfz, "Tcf": Tcf,
                             "NB": cov[3], "Socc": cov[4]}
            d = lambda f_: (f_(vals[1]) - f_(vals[-1])) / (2 * h)
            st = {"D_vac": d(lambda v: v["Fg1"]), "D_boundary_term": d(lambda v: v["Bext"]),
                  "D_mat_terms": {t: d(lambda v, t=t: v["T"][t]) for t in ("mass", "radial", "angular", "E2")},
                  "D_mat_tables_frozen": {t: d(lambda v, t=t: v["Tfz"][t]) for t in ("mass", "radial", "angular", "E2")},
                  "D_mat_fixed_covariance": {t: d(lambda v, t=t: v["Tcf"][t]) for t in ("mass", "radial", "angular", "E2")},
                  "Total": d(lambda v: v["Fgsig"] + sum(v["T"].values())),
                  "NB_change": max(abs(vals[s_]["NB"] - cov1[3]) for s_ in (1, -1)), "Socc_change": max(abs(vals[s_]["Socc"] - cov1[4]) for s_ in (1, -1))}
            st["D_mat"] = sum(st["D_mat_terms"].values())
            st["D_mat_dressing_tables"] = {t: st["D_mat_terms"][t] - st["D_mat_tables_frozen"][t] for t in ("mass", "radial", "angular")}
            st["R_stat"] = st["D_vac"] + st["D_mat"]
            st["Total_minus_R_stat_minus_S_inf"] = st["Total"] - st["R_stat"] - sigma_piece
            out["steps"][str(h)] = st
        h1, h2 = "0.001", "0.0001"
        rich = lambda a1, a2: (100 * a2 - a1) / 99
        out["richardson"] = {kk: rich(out["steps"][h1][kk], out["steps"][h2][kk]) for kk in ("D_vac", "D_boundary_term", "D_mat", "R_stat", "Total")}
        out["richardson"]["D_mat_terms"] = {t: rich(out["steps"][h1]["D_mat_terms"][t], out["steps"][h2]["D_mat_terms"][t]) for t in ("mass", "radial", "angular", "E2")}
        out["richardson"]["D_mat_dressing_tables"] = {t: rich(out["steps"][h1]["D_mat_dressing_tables"][t], out["steps"][h2]["D_mat_dressing_tables"][t])
                                                     for t in ("mass", "radial", "angular")}
        out["seconds"] = time.time() - t0
        res["states"].append(out)
        print(json.dumps({kk: vv for kk, vv in out.items() if kk not in ("steps",)}), flush=True)
    with open(os.path.join(a.out, "partB3.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
