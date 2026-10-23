#!/usr/bin/env python
"""s22_audit_v1.py (busi, inbox #434: reviewer B's check of NOTCOMP 2.2 v1, ReviewB_nc22chk_v1). Box CPU, reads stored jsons only.
(1) the arithmetic audit reviewer B asks for: per window, at full precision, the raw per-chain fluences (actual, held), the multiplicity 16 kappa,
    the comparator integral B_kappa = int L_H(T_H) dt (the production summary's L_H_TH, which is already the channel's 16 kappa chains:
    its per-chain value is L_H_TH/(16 kappa)), the normalized values, and the combined ratio
        R(W) = sum_k 16k [F_act,k - F_hold,k] / sum_k 16k F_act,k     (raw fluences; no comparator needed);
(2) record-resolved delta L(t) = L_act - L_hold on the common 8 a grid: max |delta L|, int |delta L| dt beside the signed int, per window,
    and the first clock after T_ref at which |delta L| exceeds 1, 10 and 50 percent of max |L_act| over the window;
(3) the work by two routes: E_inv(T) - E_inv(T_ref) from the lean rows, and the production's record-resolved accumulator
    sum of W_acc_interval (int Tr[h-dot (C - P_-)] dt over each 2 r_s interval, per chain) over the window; and the collar fluence by two
    routes: the lean left sum against the production's collar_flow_acc difference.
Usage: python s22_audit_v1.py S22DIR OUT.json
"""
import json
import os
import sys

import numpy as np

RS = 64.0
PROD = "<internal>/v6o_B2_k%d_summary.json"


def lean(path):
    d = json.load(open(path))
    t = np.array([r["t"] for r in d["records"]], float); L = np.array([r["L_at_t"] for r in d["records"]], float)
    o = np.argsort(t); t, L = t[o], L[o]
    key = list(d["rows"][0]["per_channel"].keys())[0]
    rows = {round(r["t_over_rs"], 6): r["per_channel"][key]["E_inv"] for r in d["rows"]}
    return t, L, rows


def prod(K):
    d = json.load(open(PROD % K))
    out = {}
    for r in d["rows"]:
        pc = r["per_channel"][str(K)]
        out[round(r["t_over_rs"], 6)] = {"LH": r.get("L_H_TH"), "W_int": pc.get("W_acc_interval"), "collar": pc.get("collar_flow_acc"), "E_inv": pc.get("E_inv")}
    return out


def fl(t, L, a, b):
    dt = np.diff(np.append(t, t[-1] + 8.0)); s = (t >= a - 1e-9) & (t < b - 1e-9)
    return float(np.sum(L[s] * dt[s]))


def main():
    sd, outp = sys.argv[1], sys.argv[2]
    R = {}
    for K in (1, 2):
        R["act%d" % K] = lean(os.path.join(sd, "runs/act%d/act%d_k%d_summary.json" % (K, K, K)))
        for T in (60, 80):
            R["h%d_%d" % (T, K)] = lean(os.path.join(sd, "runs/h%ds_k%d/h%ds_k%d_k%d_summary.json" % (T, K, T, K, K)))
    P = {K: prod(K) for K in (1, 2)}
    res = {"windows": []}
    for T in (60, 80):
        edges = list(range(T, 101, 10)) + [None]
        wins = [(a, a + 10) for a in range(T, 100, 10)] + [(T, 100)]
        for (a, b) in wins:
            row = {"T_ref": T, "from": a, "to": b, "ch": {}}
            num = den = 0.0
            for K in (1, 2):
                ta, La, Ea = R["act%d" % K]; th, Lh, Eh = R["h%d_%d" % (T, K)]
                Fa = fl(ta, La, a * RS, b * RS); Fh = fl(th, Lh, a * RS, b * RS)
                m = 16.0 * K
                pk = P[K]; tg = sorted(x for x in pk if a - 1e-9 <= x <= b + 1e-9)
                lh = np.array([pk[x]["LH"] for x in tg], float)
                B = float(np.trapezoid(lh, np.array(tg) * RS)) if hasattr(np, "trapezoid") else float(np.trapz(lh, np.array(tg) * RS))
                # record-resolved difference on the common grid
                sa = (ta >= a * RS - 1e-9) & (ta < b * RS - 1e-9); sh = (th >= a * RS - 1e-9) & (th < b * RS - 1e-9)
                assert np.allclose(ta[sa], th[sh]), "grids differ"
                dL = La[sa] - Lh[sh]; tt = ta[sa]
                Lmax = float(np.max(np.abs(La[sa])))
                onset = {}
                for f in (0.01, 0.1, 0.5):
                    idx = np.nonzero(np.abs(dL) > f * Lmax)[0]
                    onset[str(f)] = float(tt[idx[0]] / RS) if len(idx) else None
                # work: two routes; collar: two routes
                W_e = Ea[round(float(b), 6)] - Ea[round(float(a), 6)]
                W_p = float(sum(pk[x]["W_int"] for x in tg if x > a + 1e-9))     # each interval's accumulator ends at its row
                C_p = float(pk[round(float(b), 6)]["collar"] - pk[round(float(a), 6)]["collar"])
                E_p = float(pk[round(float(b), 6)]["E_inv"] - pk[round(float(a), 6)]["E_inv"])
                ch = {"F_act": Fa, "F_hold": Fh, "dF": Fa - Fh, "mult": m, "B_LH_channel": B, "act_over_B": m * Fa / B, "hold_over_B": m * Fh / B,
                      "dF_over_B": m * (Fa - Fh) / B, "dF_over_Fact": (Fa - Fh) / Fa, "max_abs_dL": float(np.max(np.abs(dL))), "max_abs_Lact": Lmax,
                      "int_abs_dL": float(np.sum(np.abs(dL)) * 8.0), "int_dL": float(np.sum(dL) * 8.0), "onset_rs": onset,
                      "W_Einv_lean": W_e, "W_Einv_prod": E_p, "W_acc_prod": W_p, "collar_prod": C_p, "collar_lean_minus_prod": Fa - C_p}
                row["ch"][str(K)] = ch
                num += m * (Fa - Fh); den += m * Fa
            row["R_combined"] = num / den; row["num"] = num; row["den"] = den
            res["windows"].append(row)
            print("T_ref %d  window %d-%d  R_combined = %.6e / %.6e = %+.6f" % (T, a, b, num, den, num / den))
            for K, c in row["ch"].items():
                print("   k%s: F_act %.9e F_hold %.9e dF %+.9e (x16k %g) B %.6e -> act %.4f hold %.4f dF %+.4f L_H; dF/F %+.5f; max|dL| %.3e of max|L| %.3e, int|dL| %.3e vs int dL %+.3e; onset 1/10/50%% %s"
                      % (K, c["F_act"], c["F_hold"], c["dF"], c["mult"], c["B_LH_channel"], c["act_over_B"], c["hold_over_B"], c["dF_over_B"], c["dF_over_Fact"],
                         c["max_abs_dL"], c["max_abs_Lact"], c["int_abs_dL"], c["int_dL"], c["onset_rs"]))
                print("        work: E_inv lean %+.6e, E_inv prod %+.6e, prod accumulator %+.6e; collar: prod %.9e, lean - prod %+.3e" % (
                    c["W_Einv_lean"], c["W_Einv_prod"], c["W_acc_prod"], c["collar_prod"], c["collar_lean_minus_prod"]))
    json.dump(res, open(outp, "w"), indent=1)


if __name__ == "__main__":
    main()
