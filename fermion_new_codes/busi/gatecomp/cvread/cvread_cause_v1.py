#!/usr/bin/env python
"""cvread_cause_v1.py (busi lane, inbox #426; ReviewA_cvvoid_v1.md sec. 3): the centre term's cause, from the stored records of X = CV2048
(16 r_s) and P = job (b) (32 r_s), no card. Read-only; the field module's ivp route (engine_fields_v16, QI2_FIELDS_ROUTE unset), which
reproduces both runs' stored nodes to <= 1.3e-10 (CVREAD_reads_v1).
  (a) the sub-step counts and decisions per channel on T 1 to 4, both sides (from cvread_ii_v1.json); the first-order clock-offset
      signature dN(0,T) / Ndot(0,T) over T 3 to 4 (Ndot central on P's records).
  (b) the end datum: X's stored sources at T 3.3125, 3.4375, 3.5 re-solved (1) as stored (N(r_collar_X) = 1), (2) with the lapse datum at
      X's last node taken from P (N_P at the same radius, the solver's normalisation N = exp(lnN - lnN(r_collar)) rescaled), (3) with X's
      sources on 1..2048 a and P's sources beyond (P's chain end, collar and charge at 32 r_s, X's interior): N(0) against X's and P's.
  (3.3125) the unconverged record: the profile of X - P against P's own iterate residual N_P - N_P,applied on r <= 64 a (shape and ratio).
  (c) the centre per record: rho_0 (the series coefficient), the regular flag, N(0), eps_tot and eps_pairs at site 1, both sides, T 0 to 5.
  python cvread_cause_v1.py X_DIR/X_TAG P_DIR/P_TAG --rs 128 --kits KITS --fm-dir FM --ii cvread_ii_v1.json --out OUT.json
"""
import argparse
import json
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("X"); ap.add_argument("P"); ap.add_argument("--rs", type=float, required=True)
    ap.add_argument("--kits", required=True); ap.add_argument("--fm-dir", dest="fm_dir", required=True); ap.add_argument("--ii", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.kits); sys.path.insert(0, a.fm_dir)
    import gate_common_v1 as GC
    import engine_fields_v16 as FM
    rs = a.rs
    JX, FX = GC.load_run(*GC.split_spec(a.X)); JP, FP = GC.load_run(*GC.split_spec(a.P))
    nX = int(JX["args"]["nsites"]); R = np.asarray(FX["R"], float); RP_ = np.asarray(FP["R"], float)
    g = float(FX["g"]); ell = float(FX["ell"])
    IX, IP = GC.rows_by_T(FX, rs), GC.rows_by_T(FP, rs); RX, RP = GC.recs_by_T(JX, rs), GC.recs_by_T(JP, rs)
    out = {}

    def solve(eps, J, Rg):
        Nf, vf, Wf, info = FM.solve_constraints_ivp(np.asarray(eps, float), np.asarray(J, float), g, Rg, ell, collar_policy="flag", support_rel=1e-14)
        if info.get("failed"):
            return None, info
        return (info["N_dense"], info["v_dense"]), info

    # ---------------- (a)
    D = json.load(open(a.ii))
    print("== (a) sub-step counts and decisions per channel, T 1 to 4 (X / P)")
    nrec = 0; diffs = []
    for r in D["rows"]:
        t = float(r["T"])
        if not (1.0 - 1e-9 <= t <= 4.0 + 1e-9):
            continue
        nrec += 1
        for c in r["channels"]:
            dd = {q: v for q, v in c["dec"].items() if v[0] != v[1]}
            if c["nsub"][0] != c["nsub"][1] or dd:
                diffs.append((t, c["ch"], c["nsub"], dd))
    print("   records %d; channel-records %d; differing entries:" % (nrec, 10 * nrec))
    for t, ch, ns, dd in diffs:
        print("   T %.4f ch %d: nsub %s/%s; %s" % (t, ch, ns[0], ns[1], ", ".join("%s %s/%s" % (q, v[0], v[1]) for q, v in dd.items()) or "decisions equal"))
    out["a_diffs"] = diffs
    # clock-offset signature
    print("-- (a) signature: dN(0,T) / Ndot(0,T), Ndot central on P (per r_s); a centre clock offset would make it constant")
    TP = np.asarray(FP["T"], float) / rs
    sig = []
    for t in sorted(IX):
        if not (3.0 - 1e-9 <= t <= 4.0 + 1e-9):
            continue
        ip = IP[t]
        if ip - 1 < 0 or ip + 1 >= len(TP):
            continue
        n0 = lambda F, i: float(np.asarray(F["N_nodes"][i], float)[0])
        dN = n0(FX, IX[t]) - n0(FP, ip)
        Nd = (n0(FP, ip + 1) - n0(FP, ip - 1)) / (TP[ip + 1] - TP[ip - 1])
        Nb = (n0(FP, ip) - n0(FP, ip - 1)) / (TP[ip] - TP[ip - 1])
        sig.append((t, dN, Nd, dN / Nd if Nd else None, Nb, dN / Nb if Nb else None))
        print("   %.4f dN(0) %+.3e  Ndot_c %+.3e  ratio %+.3e r_s | Ndot_back %+.3e ratio %+.3e r_s" % (t, dN, Nd, dN / Nd, Nb, dN / Nb))
    out["a_signature"] = sig

    # ---------------- (b)
    print("\n== (b) the end datum: X's stored sources re-solved with the 32 r_s end")
    bres = {}
    for t in (3.25, 3.3125, 3.4375, 3.5):
        ix, ip = IX[t], IP[t]
        eX, jX = np.asarray(FX["eps_tot"][ix], float), np.asarray(FX["J_tot"][ix], float)
        eP, jP = np.asarray(FP["eps_tot"][ip], float), np.asarray(FP["J_tot"][ip], float)
        SX, iX = solve(eX, jX, R)
        SP, iP = solve(eP, jP, RP_)
        hyb_e = eP.copy(); hyb_j = jP.copy(); hyb_e[:nX] = eX; hyb_j[:nX] = jX
        SH, iH = solve(hyb_e, hyb_j, RP_)
        NX0 = float(SX[0](np.array([0.0]))[0]); NP0 = float(SP[0](np.array([0.0]))[0]); NH0 = float(SH[0](np.array([0.0]))[0])
        rend = float(R[-1])
        NP_end = float(SP[0](np.array([rend]))[0]); NX_end = float(SX[0](np.array([rend]))[0])
        NX0_datum = NX0 * NP_end / NX_end
        EX = FM.charge(iX, g); EP = FM.charge(iP, g); EH = FM.charge(iH, g)
        row = {"NX0": NX0, "NP0": NP0, "dN0_stored": NX0 - NP0, "N_end_X": NX_end, "N_end_P_at_16rs": NP_end,
               "NX0_with_P_datum": NX0_datum, "shift_datum": NX0_datum - NX0, "dN0_after_datum": NX0_datum - NP0,
               "NH0_hybrid": NH0, "shift_hybrid": NH0 - NX0, "dN0_after_hybrid": NH0 - NP0,
               "charge_X": EX, "charge_P": EP, "charge_hybrid": EH, "r_collar": (iX.get("r_collar"), iP.get("r_collar"), iH.get("r_collar")),
               "centre_flag": (iX.get("centre_flag"), iP.get("centre_flag"))}
        # profile of the hybrid's change against the X - P difference
        rr = np.array([0.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 512.0, 1024.0, 2048.0])
        row["profile_r"] = rr.tolist()
        row["X_minus_P"] = (SX[0](rr) - SP[0](rr)).tolist()
        row["hybrid_minus_X"] = (SH[0](rr) - SX[0](rr)).tolist()
        bres[t] = row
        print(" T %.4f: N(0) X %.8f P %.8f, X - P %+.3e | N(16 r_s) X %.9f P %.9f | X with P's lapse datum: N(0) moves %+.3e (X - P then %+.3e) | "
              "X interior + P's sources beyond 16 r_s: N(0) moves %+.3e (then %+.3e) | charge X %.6f P %.6f hybrid %.6f | r_collar %s | centre %s"
              % (t, NX0, NP0, NX0 - NP0, NX_end, NP_end, NX0_datum - NX0, NX0_datum - NP0, NH0 - NX0, NH0 - NP0, EX, EP, EH, row["r_collar"], row["centre_flag"]))
        print("     r (a):        " + " ".join("%9.0f" % x for x in rr))
        print("     X - P:        " + " ".join("%+9.1e" % x for x in row["X_minus_P"]))
        print("     hybrid - X:   " + " ".join("%+9.1e" % x for x in row["hybrid_minus_X"]))
    out["b"] = bres

    # ---------------- 3.3125
    print("\n== the unconverged record 3.3125: X - P against P's own iterate residual (site rows, r <= 64 a)")
    t = 3.3125; ix, ip = IX[t], IP[t]
    sel = R <= 64.0
    dXP = np.asarray(FX["N"][ix], float)[:nX] - np.asarray(FP["N"][ip], float)[:nX]
    dXPa = np.asarray(FX["N_applied"][ix], float)[:nX] - np.asarray(FP["N_applied"][ip], float)[:nX]
    resP = np.asarray(FP["N"][ip], float)[:nX] - np.asarray(FP["N_applied"][ip], float)[:nX]
    resX = np.asarray(FX["N"][ix], float) - np.asarray(FX["N_applied"][ix], float)
    vXP = np.asarray(FX["v"][ix], float)[:nX] - np.asarray(FP["v"][ip], float)[:nX]
    vresP = np.asarray(FP["v"][ip], float)[:nX] - np.asarray(FP["v_applied"][ip], float)[:nX]
    cN = float(np.dot(dXP[sel], resP[sel]) / (np.linalg.norm(dXP[sel]) * np.linalg.norm(resP[sel])))
    kN = float(np.dot(dXP[sel], resP[sel]) / np.dot(resP[sel], resP[sel]))
    cv = float(np.dot(vXP[sel], vresP[sel]) / (np.linalg.norm(vXP[sel]) * np.linalg.norm(vresP[sel])))
    print("   |X - P| N max %.2e (applied %.2e); P residual |N - N_applied| max %.2e at %.0f a; X residual %.2e" % (
        np.abs(dXP[sel]).max(), np.abs(dXPa[sel]).max(), np.abs(resP[sel]).max(), R[sel][np.argmax(np.abs(resP[sel]))], np.abs(resX[sel]).max()))
    print("   cosine(X - P, P residual) on r <= 64 a: N %+.3f, v %+.3f; least-squares ratio (X - P) = k (N_P - N_P,applied): k %+.2f" % (cN, cv, kN))
    for q in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64):
        j = q - 1
        print("     r %3d a: X - P %+.3e  P resid %+.3e  X - P(applied) %+.3e" % (q, dXP[j], resP[j], dXPa[j]))
    out["u3125"] = {"cos_N": cN, "cos_v": cv, "k": kN, "max_dXP": float(np.abs(dXP[sel]).max()), "max_resP": float(np.abs(resP[sel]).max())}
    for t in (3.3125,):
        for nm, J in (("X", RX[t]), ("P", RP[t])):
            s = J.get("solve_v32") or {}
            print("   %s 3.3125: converged %s, iterations %s, kept residual %s, unconverged %s" % (nm, J.get("converged"), J.get("iterations"), s.get("kept_residual"), json.dumps(J.get("unconverged"))))

    # ---------------- (c)
    print("\n== (c) the centre per record, T 0 to 5: rho_0 (series), regular, N(0), eps_tot(1 a), eps_pairs(1 a), X / P")
    crow = []
    nirr = {"X": 0, "P": 0}; agree = 0; tot = 0
    for t in sorted(set(IX) & set(IP)):
        if t > 5.0 + 1e-9:
            continue
        ix, ip = IX[t], IP[t]
        vals = []
        for F, i, Rg in ((FX, ix, R), (FP, ip, RP_)):
            rn = FM.coarse_grid(Rg, ell)
            src = FM.SmoothedSources5(np.asarray(F["eps_tot"][i], float), np.asarray(F["J_tot"][i], float), Rg, ell, rn[-1], 1e-14)
            cc = FM.centre_coefficients(src, g, ell)
            vals.append((cc["rho0"], cc["regular"], float(np.asarray(F["N_nodes"][i], float)[0]), float(F["eps_tot"][i][0]), float(F["eps_pairs"][i][0])))
        (r0x, regx, nx, ex, epx), (r0p, regp, npp, ep, epp) = vals
        nirr["X"] += (not regx); nirr["P"] += (not regp); tot += 1; agree += (regx == regp)
        crow.append((t, r0x, r0p, regx, regp, nx, npp, ex, ep, epx, epp))
        print("   %.4f rho0 %+.4e / %+.4e (d %+.1e) | regular %s/%s | N(0) %.6f / %.6f (d %+.1e) | eps(1a) %+.4e / %+.4e | pairs(1a) %+.4e / %+.4e" % (
            t, r0x, r0p, r0x - r0p, regx, regp, nx, npp, nx - npp, ex, ep, epx, epp))
    print("   irregular centres (rho_0 <= 0): X %d, P %d of %d records; the flag agrees on %d" % (nirr["X"], nirr["P"], tot, agree))
    out["c"] = crow
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    print("written %s" % a.out)


if __name__ == "__main__":
    main()
