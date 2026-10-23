#!/usr/bin/env python
"""Strand 4 Part B, the lead's audit after the reviewer B red team (SPEC_strand4_controls.md, "After the reviewer B red team" items 2 and 4;
builder analyst).  [paper, control]

Item 4, every stored state of partB3_v4 (the 0.1 maximum and dilute member, the 0.2 members j = 9, 10, 11, the 0.05 members j = 1, 2, 3),
the chain built as in strand4_partB3_v4: D_lat three ways (exterior (A - B)/G, source sums (M_A - M_B)/2 with M_A = sum E/psi and
M_B = sum (N E/psi + D), rows -(1/2) sum [(N - 1) E/psi + D]), the charge 1^T K u/(4 pi) against A, and E_ret/E_partial - 1 with
E_ret = F_g + E_exc,ret, F_g = -u^T K v/(2 pi G), E_partial = 2 A/G (u = psi - 1, v = chi - 1).
Item 2, the 0.1 maximum: (a) the paper's printed "dressing-residual dilation defect" is the expression of the paper lane's
dev30_tails_virial.py, 0.5 sum_n m_* N_n sz_n psi_n dmu_L(psi_n) (fixed covariance; dmu_L from the chain's fac); it is evaluated on the
reconstructed state and set beside the stored value of stage2_members.json at the same N_c; (b) the tables' replace-by-1 share of D_mat
split into the explicit part d/dlam [T_cf(lam) - T_cf,1(lam)] (fields dilated, covariance held at lam = 1) and the orbital part (the
share minus the explicit part), T_1 being E_mat with mu_L, kappa_L and kappa_L(psibar) replaced by 1; (c) the share with the tables
frozen at their stored spatial values, d/dlam [T(lam) - T_fs(lam)], T_fs with mu_L, kappa_L, kappa_L(psibar) held at their lam = 1
profiles; per term (mass, radial, angular) and in total, at h in {1e-4, 1e-5, 1e-6, 1e-7} with the Richardson value of each
consecutive pair.  The dilation family, the transport and E_mat are strand4_partB3_v4's.  Writes <out>/partB_audit.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline

TERMS = ("mass", "radial", "angular")
STEPS = (1e-4, 1e-5, 1e-6, 1e-7)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage2-members", required=True, help="the paper lane's B20-03c stage2_members.json (its tails_virial entries carry the printed dilation_defect)")
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
    SM = json.load(open(a.stage2_members))
    states = []
    best = MX["best"]
    key = "Nc%.9f_" % best["Nc"]
    states.append(("maximum m*a 0.1", "0.1", {k[len(key):]: PMX[k] for k in PMX.files if k.startswith(key)}, best["rec"]["NB"], None, best["Nc"]))
    for fam, js, lab in (("0.1", (0,), "dilute member j = 0"), ("0.2", (9, 10, 11), None), ("0.05", (1, 2, 3), None)):
        for mem in S2["members"]:
            if mem["fam"] == fam and mem["j"] in js:
                k2 = "%s_%02d_1_1_" % (fam, mem["j"])
                P = {k[len(k2):]: PR[k] for k in PR.files if k.startswith(k2)}
                sig = {s: P["sig_" + s] for s in ("dm", "dQ", "dW")} if "sig_dm" in P else None
                label = lab or "m*a %s member j = %d (N_c %.4f)" % (fam, mem["j"], mem["Nc_target"])
                states.append((label, fam, P, mem["combos"]["1,1"]["NB"], sig, None))
    k = 1 / (2 * np.pi * G)
    res = {"meta": {"G": G, "Ts": Ts}, "B1_all": [], "audit": None}
    for label, fam, P, NB, sig, Nc in states:
        TAB = BC.Tables(TABZ["%s_p-2" % fam], RESP[fam])
        REC = FLAT["matching"][fam]
        psi, chi = np.asarray(P["psi"], float), np.asarray(P["chi"], float)
        Ns = len(psi)
        n = np.arange(1, Ns + 1, dtype=float)
        ch = BC.make_chain(Ns, float(fam), G, Ts, TAB, REC["UBp"], REC["UMp"], lamR=1.0, lam2=1.0)
        ch.sig = None if sig is None else {kk: vv.copy() for kk, vv in sig.items()}
        F, ev, parts = ch.free_energy(psi, chi, NB)
        Eexc = F + parts["TS"]
        r = ev["rows"]
        N = chi / psi
        E, D = r["E"], r["D"]
        A = Ns * (psi[-1] - 1)
        B = -Ns * (chi[-1] - 1)
        Gr = 1.0 / np.maximum(n[:, None], n[None, :])
        Kmat = 4 * np.pi * np.linalg.inv(Gr)
        u, v = psi - 1, chi - 1
        Da = float(u @ (Kmat @ v))
        MA_s = float(np.sum(E / psi))
        MB_s = float(np.sum(N * E / psi + D))
        rec = {"state": label, "Ns": Ns, "Dlat_exterior": (A - B) / G, "Dlat_sums": (MA_s - MB_s) / 2,
               "Dlat_rows": -0.5 * float(np.sum((N - 1) * E / psi + D)), "A": A, "A_K_charge": float(np.sum(Kmat @ u)) / (4 * np.pi),
               "E_ret_over_E_partial_minus_1": (-Da / (2 * np.pi * G) + Eexc) / (2 * A / G) - 1}
        res["B1_all"].append(rec)
        print(json.dumps(rec), flush=True)
        if label != "maximum m*a 0.1":
            continue
        # ---- the one-state audit at the 0.1 maximum
        lev, f = ev["lev"], np.asarray(ev["f"])
        occ = [(t, fj) for t, fj in zip(lev, f) if fj > 1e-15]
        xs_in = np.concatenate([-n[:3][::-1], n])
        ext = np.arange(Ns + 1, Ns + 6, dtype=float)

        def spline(vals, tail):
            return CubicSpline(np.concatenate([xs_in, ext]), np.concatenate([vals[:3][::-1], vals, tail(ext)]))
        sp_psi = spline(psi, lambda rr: 1 + A / rr)
        sp_chi = spline(chi, lambda rr: 1 - B / rr)
        xs_o = np.concatenate([[0.0], n, ext])
        orb = []
        for (e, kap, vv), fj in occ:
            uu = np.asarray(vv).reshape(Ns, 2)
            sps = [(CubicSpline(xs_o, np.concatenate([[0.0], uu[:, c].real, np.zeros(len(ext))])),
                    CubicSpline(xs_o, np.concatenate([[0.0], uu[:, c].imag, np.zeros(len(ext))]))) for c in range(2)]
            orb.append((kap, fj, sps, uu))

        def fields(lam):
            rr = lam * n
            return np.sqrt(lam) * sp_psi(rr), np.sqrt(lam) * sp_chi(rr)

        def covariance(lam):
            rr = lam * n
            by_k = {}
            for kap, fj, sps, uu in orb:
                ul = uu.copy() if lam == 1.0 else np.stack([sps[c][0](rr) + 1j * sps[c][1](rr) for c in range(2)], axis=1)
                by_k.setdefault(kap, []).append((fj, ul.reshape(-1)))
            sz = np.zeros(Ns)
            Zb = np.zeros(Ns - 1)
            ksy = np.zeros(Ns)
            for kap, lst in by_k.items():
                M = np.array([vv for _, vv in lst]).T
                Q, Rq = np.linalg.qr(M)
                Q = Q * np.sign(np.real(np.diag(Rq)))[None, :]
                w = ch.w[kap]
                for (fj, _), col in zip(lst, Q.T):
                    uu = col.reshape(Ns, 2)
                    wf = w * fj
                    zz = -0.5j * (np.conj(uu[:-1, 0]) * uu[1:, 1] + np.conj(uu[:-1, 1]) * uu[1:, 0])
                    Zb += wf * 2.0 * zz.real
                    sz += wf * (np.abs(uu[:, 0]) ** 2 - np.abs(uu[:, 1]) ** 2)
                    ksy += wf * kap * 2.0 * (np.conj(uu[:, 0]) * uu[:, 1]).imag
            return sz, Zb, ksy
        s1, c1 = fields(1.0)
        N1 = c1 / s1
        fac1 = ch.fac(N1, s1)

        def terms(Nl, s, cov, tables="live"):
            sz, Zb, ksy = cov
            c = ch.fac(Nl, s)
            if tables == "one":
                muL, kaL, kabL = np.ones_like(s), np.ones_like(s), np.ones(Ns - 1)
            elif tables == "stored":
                muL, kaL, kabL = fac1["muL"], fac1["kaL"], fac1["kabL"]
            else:
                muL, kaL, kabL = c["muL"], c["kaL"], c["kabL"]
            return {"mass": float(np.sum(ch.mstar * Nl * muL * sz)), "radial": float(np.sum(c["Q"] * kabL * Zb)),
                    "angular": float(np.sum(c["W"] * kaL * ksy))}
        cov1 = covariance(1.0)
        # (a) the paper's expression on the reconstructed state
        paper_expr = 0.5 * float(np.sum(ch.mstar * N1 * cov1[0] * s1 * fac1["dmuL"]))
        tv = [m_ for m_ in SM.get("tails_virial", []) if isinstance(m_, dict) and m_.get("fam") == "0.1" and m_.get("combo") == "1,1"
              and m_.get("Nc") is not None and "dilation_defect" in m_]
        tv.sort(key=lambda m_: abs(m_["Nc"] - Nc))
        stored_val = tv[0]["dilation_defect"] if tv else None
        stored_Nc = tv[0]["Nc"] if tv else None
        steps = {}
        for h in STEPS:
            vals = {}
            for sgn in (1, -1):
                lam = 1 + sgn * h
                s, c = fields(lam)
                Nl = c / s
                cov = covariance(lam)
                vals[sgn] = {"live": terms(Nl, s, cov), "one": terms(Nl, s, cov, "one"), "stored": terms(Nl, s, cov, "stored"),
                             "cf_live": terms(Nl, s, cov1), "cf_one": terms(Nl, s, cov1, "one")}
            d = lambda g_: (g_(vals[1]) - g_(vals[-1])) / (2 * h)
            st = {"share_replace_by_1": {t: d(lambda v_, t=t: v_["live"][t] - v_["one"][t]) for t in TERMS},
                  "explicit_fixed_covariance": {t: d(lambda v_, t=t: v_["cf_live"][t] - v_["cf_one"][t]) for t in TERMS},
                  "share_frozen_at_stored": {t: d(lambda v_, t=t: v_["live"][t] - v_["stored"][t]) for t in TERMS}}
            st["orbital_work"] = {t: st["share_replace_by_1"][t] - st["explicit_fixed_covariance"][t] for t in TERMS}
            for grp in list(st):
                st[grp]["total"] = sum(st[grp][t] for t in TERMS)
            steps["%g" % h] = st
        keys = ["%g" % h for h in STEPS]
        rich = {}
        for i in range(len(keys) - 1):
            h1, h2 = keys[i], keys[i + 1]
            rich["%s,%s" % (h1, h2)] = {grp: {t: (100 * steps[h2][grp][t] - steps[h1][grp][t]) / 99 for t in steps[h1][grp]} for grp in steps[h1]}
        res["audit"] = {"state": label, "Nc": Nc, "paper_dressing_residual_expression_on_reconstruction": paper_expr,
                        "stored_dilation_defect": stored_val, "stored_Nc": stored_Nc, "D_lat_exterior": (A - B) / G,
                        "steps": steps, "richardson_pairs": rich}
        print(json.dumps({"paper_expr": paper_expr, "stored": stored_val, "stored_Nc": stored_Nc,
                          "rich_last": rich["%s,%s" % (keys[-2], keys[-1])]}), flush=True)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "partB_audit.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
