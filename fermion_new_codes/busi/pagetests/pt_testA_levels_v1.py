#!/usr/bin/env python
"""pt_testA_levels_v1.py (busi, inbox #428, Test A observable (ii), ReviewA_pc6_v1.md sec. 4): the interior block's spectrum at a dozen
clocks of the hold and of each recession, and the levels' drift against the Hellmann-Feynman rate
    omega_dot_n = <n|h_dot|n> = sum_x [<n|e_x|n> N_dot_x + <n|p_x|n> v_dot_x],  e_x = (1/2){Pi_x, h_0}, p_x = (1/2){Pi_x, Q}
(reviewer A's slide, sec. 1: omega_dot_n = sum_x J_n(x) v_dot_x in the shell, J_n(x) = <n|p_{x,r}|n>). The block: h[N, v] of the chain
(page_ops_v6 operators, the run's fields from pt_fieldsA_v1) restricted to the sites R <= R_b (Dirichlet cut), R_b = --Rb sites (128:
256 dimensions; 64 the check). Per clock: the eigenvalues, the band label s_n = sign <n|(1/2){N, h_0}|n> (the clock part: + upper band,
- lower band), inverted = band and Killing sign disagree (the rising band: lower band with omega > 0; the pocket: upper band with
omega < 0); the levels in the window |omega| < 3 T_H; the HF rate of every level (N and v parts apart) by a centred difference of the
fields over +-dt; the finite-difference drift of the levels tracked between consecutive clocks by maximal overlap; and the count of
levels whose tracked energy changed sign. Box CPU, seconds."""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "code_v6"))
import page_ops_v6 as OPS  # noqa: E402
import pt_fieldsA_v1 as FA  # noqa: E402


def block_ops(Rb, kappa):
    R = np.arange(1, Rb + 1, dtype=float)
    h0, Q = OPS.chain_sparse(R, 0.0, kappa)
    return R, h0, Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="P64_s4_B2n4096")
    ap.add_argument("--T-rec-rs", type=float, default=100.0)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--Leffs", default="0,1,3,10,30")
    ap.add_argument("--u-rs", default="0,100,200,300,400,600,800,1000,1250,1500,1750,2000")
    ap.add_argument("--Rb", type=int, default=128)
    ap.add_argument("--dt-hf", type=float, default=256.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = json.load(open(a.prep))
    H = FA.load_record(a.label, a.T_rec_rs)
    tz = np.load(a.prep.replace(".json", "_table.npz"))
    table = {"lam": tz["lam"], "N": tz["N"], "v": tz["v"], "E": tz["E"]}
    LH = P["L_H"]; g = H["g"]
    rsn = H["rs_nom"]
    R, h0, Q = block_ops(a.Rb, a.kappa)
    us = [float(x) for x in a.u_rs.split(",")]
    out = {"script": "pt_testA_levels_v1.py", "args": vars(a), "L_H": LH, "runs": {}}
    for Lf in [float(x) for x in a.Leffs.split(",")]:
        F = FA.FieldsA(H, a.Rb, "cpu", L_eff=Lf * LH, table=table if Lf > 0 else None)
        clocks = []
        prevV = prevE = None
        for u in us:
            t = H["T_rec"] + u * rsn
            N, v = F.numpy_fields(t)
            h = OPS.clocked_csr(h0, Q, N, v).toarray()
            hN = OPS.clocked_csr(h0, Q * 0, N, v * 0).toarray()
            e, V = np.linalg.eigh(h)
            TH = 1.0 / (8 * math.pi * g * F.charge(t))
            band = np.sign(np.real(np.einsum("in,ij,jn->n", V.conj(), hN, V)))
            inv = (band < 0) & (e > 0) | (band > 0) & (e < 0)
            # Hellmann-Feynman: h_dot = h[N_dot, v_dot]
            Np, vp = F.numpy_fields(t + a.dt_hf); Nm, vm = F.numpy_fields(max(t - a.dt_hf, H["T_rec"]))
            dtt = (t + a.dt_hf) - max(t - a.dt_hf, H["T_rec"])
            hdN = OPS.clocked_csr(h0, Q * 0, (Np - Nm) / dtt, 0 * vp).toarray()
            hdv = OPS.clocked_csr(h0 * 0, Q, 0 * Np, (vp - vm) / dtt).toarray()
            rN = np.real(np.einsum("in,ij,jn->n", V.conj(), hdN, V))
            rv = np.real(np.einsum("in,ij,jn->n", V.conj(), hdv, V))
            win = np.abs(e) < 3 * TH
            row = {"u_rs": u, "t": t, "E": F.charge(t), "T_H": TH, "n_levels": len(e), "n_inverted": int(inv.sum()),
                   "n_rising": int(((band < 0) & (e > 0)).sum()), "n_pocket": int(((band > 0) & (e < 0)).sum()),
                   "window_levels": [float(x) for x in e[win]], "window_inverted": [bool(x) for x in inv[win]],
                   "hf_rate_window_v": [float(x) for x in rv[win]], "hf_rate_window_N": [float(x) for x in rN[win]],
                   "hf_rate_inverted_mean_v_rising": float(np.mean(rv[(band < 0) & (e > 0)])) if ((band < 0) & (e > 0)).any() else None,
                   "hf_rate_inverted_mean_v_pocket": float(np.mean(rv[(band > 0) & (e < 0)])) if ((band > 0) & (e < 0)).any() else None,
                   "hf_rate_inverted_mean_N_rising": float(np.mean(rN[(band < 0) & (e > 0)])) if ((band < 0) & (e > 0)).any() else None,
                   "hf_rate_noninverted_mean_abs_v": float(np.mean(np.abs(rv[~inv]))) if (~inv).any() else None}
            if prevV is not None:
                O = np.abs(prevV.conj().T @ V) ** 2
                ri, ci = linear_sum_assignment(-O)
                de = e[ci] - prevE[ri]
                dt_ = t - clocks[-1]["t"]
                row["tracked_min_overlap"] = float(O[ri, ci].min())
                row["tracked_median_overlap"] = float(np.median(O[ri, ci]))
                row["sign_changes"] = int(np.sum(np.sign(e[ci]) != np.sign(prevE[ri])))
                # finite-difference drift against the HF rate averaged over the interval (levels in the window at the earlier clock)
                wprev = np.abs(prevE[ri]) < 3 * TH
                row["fd_drift_window"] = [float(x) for x in (de / dt_)[wprev]]
                row["hf_mean_window"] = [float(x) for x in (0.5 * (prev_rate[ri] + (rv + rN)[ci]))[wprev]]
                row["fd_drift_total_window_over_TH"] = float(np.mean(np.abs(de[wprev]))) / TH if wprev.any() else None
            prevV, prevE, prev_rate = V, e, rv + rN
            clocks.append(row)
            print("L_eff %g: u %6.0f r_s E %.5f T_H %.4e: levels %d, inverted %d (rising %d, pocket %d), window %d; HF v-rate rising mean %s pocket %s; %s" % (
                Lf, u, row["E"], TH, len(e), row["n_inverted"], row["n_rising"], row["n_pocket"], int(win.sum()),
                row["hf_rate_inverted_mean_v_rising"], row["hf_rate_inverted_mean_v_pocket"],
                ("sign changes %d, min overlap %.3f, window drift/T_H %s" % (row["sign_changes"], row["tracked_min_overlap"], row["fd_drift_total_window_over_TH"])) if "sign_changes" in row else ""), flush=True)
        out["runs"][str(Lf)] = clocks
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "testA_levels_k%g_Rb%d.json" % (a.kappa, a.Rb))
    json.dump(out, open(p, "w"), indent=1, default=float)
    print("wrote", p)


if __name__ == "__main__":
    main()
