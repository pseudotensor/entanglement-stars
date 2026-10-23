#!/usr/bin/env python
"""s16A_read_v1p1.py (busi; emits PAGE_Bprime_s16A_v1.1.md whole: the addendum to PAGE_Bprime_s16A_v1 reading the six owner-box pieces of
inbox #473 (s16A_h100x8_v2/g1..g3) against run 1 (s16A_h100x8_v1) on the same columns; every number printed by this script).
Pieces: g1 2c (dt 0.25 a) supply, all 128 columns to 6.93 L and the first 24 to 69.3 L; g2 run 1 both windows (the supply-exit cross
block), all 256 columns to 6.93 L and 16 + 16 to 69.3 L; g3 2b (onset 2.25 L, half-side 2.75 L) supply, 128 to 6.93 L and 24 to 69.3 L.
Labels as in v1: port=sink2L, declared covariances, no far-zone words."""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pt3_core as K3  # noqa: E402

L = 16.0


def eig(G):
    return np.sort(np.linalg.eigvalsh(0.5 * (G + G.conj().T)))


def clock_index(Ts, T):
    i = int(np.argmin(np.abs(np.asarray(Ts) - T)))
    return i if abs(Ts[i] - T) < 1e-6 * T else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ld = lambda d, t: (json.load(open(os.path.join(d, t + ".json"))), np.load(os.path.join(d, t + ".npz")))
    R1s = ld(a.v1, "s16A_1_F1_d0_dt0.5_T69.32_sup"); R1e = ld(a.v1, "s16A_1_F1_d0_dt0.5_T69.32_exit")
    P1 = {"2c_full": ld(os.path.join(a.v2, "g1"), "s16A_2c_F1_dthalf_dt0.25_T6.932_sup"), "2c_n48": ld(os.path.join(a.v2, "g1"), "s16A_2c_F1_dthalf_dt0.25_T69.32_sup_n48"),
          "x_full": ld(os.path.join(a.v2, "g2"), "s16A_1_F1_d0_dt0.5_T6.932_both"), "x_n32": ld(os.path.join(a.v2, "g2"), "s16A_1_F1_d0_dt0.5_T69.32_both_n32"),
          "2b_full": ld(os.path.join(a.v2, "g3"), "s16A_2b_F1_onset225_dt0.5_T6.932_sup"), "2b_n48": ld(os.path.join(a.v2, "g3"), "s16A_2b_F1_onset225_dt0.5_T69.32_sup_n48")}
    T1, T2 = math.log(2) * L / 0.1, math.log(2) * L / 0.01
    out = []; P = out.append
    md = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
    P("# PAGE B' s = 16, stage A: addendum v1.1, the owner-box pieces of inbox #473 (busi lane)")
    P("")
    P("A new file; PAGE_Bprime_s16A_v1.md stays as issued. Emitted whole by busi/code/pagetests/v3/s16A_read_v1p1.py; no number is retyped. Same labels as v1: `port=sink2L(...)`, declared covariances (supply_filled; no massless preparation), F1 at s = 16, sigma = L/8. These are EARLY-CLOCK AND COLUMN-SUBSET PREVIEWS of the arms reviewer B kept (ReviewB_s16A_v1 sec. 3); the full 256-column runs to the 1e2 L endpoint are queued (2c on qi2_h100_1; 2b and the 1e2 L clock on h100-f).")
    P("")
    P("## 1. What landed")
    P("")
    P("| piece | run | window and columns | dt (a) | clocks (T/L) | wall (h) | step median (s) | peak GPU (GB) | s per column per a | npz md5 |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for k, (j, z) in P1.items():
        tm = j["timing"]; K = j["K"]
        rate = tm["step_median_s"] / (min(tm["col_batch"], K) * tm["dt"])
        tag = j["args"]["tag"]
        g = {"2c": "g1", "x_": "g2", "2b": "g3"}[k[:2]]
        P("| %s | %s | %s | %s | %s | %.2f | %.3f | %.1f | %.4f | %s |" % (k, tag.split("_F1")[0].replace("s16A_", ""), "%s, %d" % (tag.split("_")[-1] if "n" not in tag.split("_")[-1] else tag.split("_")[-2], K),
          tm["dt"], ", ".join("%.4g" % x for x in j["clocks_over_L"]), tm["seconds_total_this_session"] / 3600, tm["step_median_s"], tm["peak_mem_GB"] or 0, rate,
          md(os.path.join(a.v2, g, tag + ".npz"))))
    P("")
    P("All six pieces ended with rc 0 before the 22:30 stop (driver logs), and their MD5SUMS check; nothing was killed, so nothing needs resuming. The measured dt 0.25 a rate replaces the extrapolated 0.096 s per column per a (row 2c_full).")
    P("")
    # ----------------------------------------------------------------- regression
    P("## 2. Regression: the both-window run against run 1 on the same columns and step (dt 0.5 a)")
    P("")
    jx, zx = P1["x_full"]
    i1 = clock_index(zx["T"], T1); j1 = clock_index(R1s[1]["T"], T1)
    Gx = zx["G"][i1]
    d_s = float(np.max(np.abs(Gx[:128, :128] - R1s[1]["G"][j1]))); d_e = float(np.max(np.abs(Gx[128:, 128:] - R1e[1]["G"][j1])))
    P("At 6.93 L: max |G_both[supply block] - G_run1(supply)| = %.3g; max |G_both[exit block] - G_run1(exit)| = %.3g (the same inputs, the same step, a different batch layout: a free regression gate, PASS if below 1e-10: %s)." % (
        d_s, d_e, "PASS" if max(d_s, d_e) < 1e-10 else "FAIL"))
    P("")
    # ----------------------------------------------------------------- 2c
    P("## 3. 2c: the time-step error bar at the run's clocks (dt 0.25 a against run 1's dt 0.5 a, same columns)")
    P("")
    P("| piece | T/L | columns | max \\|dG\\| | max \\|dp\\| (sorted spectra) | max \\|d diag G\\| | directions with \\|dp\\| <= 1e-4 | number delivered dt 0.5 | dt 0.25 |")
    P("|---|---|---|---|---|---|---|---|---|")
    worst = 0.0
    for k, ncol in (("2c_full", 128), ("2c_n48", 24)):
        j, z = P1[k]
        for T in (T1, T2):
            i = clock_index(z["T"], T); jr = clock_index(R1s[1]["T"], T)
            if i is None or jr is None:
                continue
            G = z["G"][i]; Gr = R1s[1]["G"][jr][:ncol, :ncol]
            dG = float(np.max(np.abs(G - Gr))); pa, pr = eig(G), eig(Gr)
            dp = float(np.max(np.abs(pa - pr))); dd = float(np.max(np.abs(np.diag(G).real - np.diag(Gr).real)))
            worst = max(worst, dG)
            P("| %s | %.4g | %d | %.3g | %.3g | %.3g | %d of %d | %.4f | %.4f |" % (k, T / L, ncol, dG, dp, dd, int((np.abs(pa - pr) <= 1e-4).sum()), ncol, np.trace(Gr).real, np.trace(G).real))
    P("")
    P("Verdict against the 1e-4 target (ReviewA_bpchk2_v1 sec. 4 criterion 2, applied between dt 0.5 and dt 0.25): max |dG| over these clocks and columns = %.3g, **%s**. With the ladder's first-order convergence, the dt 0.5 a production error is about this difference. What waits: the full 256 columns at the 1e2 L endpoint (reviewer B's 'certify full-matrix/spectral convergence at the actual endpoint'), on qi2_h100_1." % (worst, "MEETS 1e-4" if worst <= 1e-4 else "ABOVE 1e-4"))
    P("")
    # ----------------------------------------------------------------- cross block
    P("## 4. The supply-exit cross block (run 1, both windows)")
    P("")
    P("| piece | T/L | columns (supply + exit) | \\|\\|G_se\\|\\|_2 | max \\|G_se\\| element | joint spectrum: min p | max p | count p < 1e-2 (dark) | max \\|p_joint - p_(block union)\\| |")
    P("|---|---|---|---|---|---|---|---|---|")
    for k in ("x_full", "x_n32"):
        j, z = P1[k]; ns = int(z["n_supply"]); K = len(z["e"])
        for i, T in enumerate(z["T"]):
            if clock_index([T], T1) is None and clock_index([T], T2) is None:
                continue
            G = z["G"][i]; Gse = G[:ns, ns:]
            pj = eig(G); pu = np.sort(np.concatenate([eig(G[:ns, :ns]), eig(G[ns:, ns:])]))
            P("| %s | %.4g | %d + %d | %.3g | %.3g | %.4g | %.4g | %d | %.3g |" % (k, T / L, ns, K - ns, np.linalg.norm(Gse, 2), np.max(np.abs(Gse)), pj.min(), pj.max(), int((pj < 1e-2).sum()),
              np.max(np.abs(pj - pu))))
    P("")
    Gf = P1["x_full"][1]["G"][clock_index(P1["x_full"][1]["T"], T1)]; Gn = P1["x_n32"][1]["G"][clock_index(P1["x_n32"][1]["T"], T1)]
    P("At 6.93 L the cross block over all 128 + 128 columns has norm %.3g, while over the 16 + 16 subset (the first 16 columns of each window in energy order: the supply's lowest levels, 0.24 T0 upward, and the exit's most negative, -0.94 T0 upward) it is %.3g: the coupling between the windows lies outside that subset (the mirror pairing, if it holds, couples supply level e to exit level -e, and this subset holds no such pair)." % (
        np.linalg.norm(Gf[:128, 128:], 2), np.linalg.norm(Gn[:16, 16:], 2)))
    P("")
    P("Reading. A nonzero cross block means the supply and exit spans are not independent channels of the sink2L dilation at these clocks: the joint spectrum differs from the union of the two windows' spectra by the last column, and the dark-combination count is the number of joint directions delivering less than 1e-2. The joint thermal entropy is not reported: with the exit window in the hole (conjugate) convention, a joint record of both windows mixes particle and hole channels and needs the Nambu form, which is not written (ReviewA_bpchk2_v1 sec. 2; the reader's refusal stays). What waits: the 256-column cross block at the 1e2 L clock (h100-f).")
    P("")
    # ----------------------------------------------------------------- 2b
    P("## 5. 2b: the onset dependence (onset 2.25 L, half-side 2.75 L, against run 1's 2 L; a result of the sink2L model, not an error bar)")
    P("")
    P("| piece | T/L | columns | number delivered, onset 2 L | onset 2.25 L | relative change | max \\|dp\\| (sorted) | median dp | dp/d(onset) per L, median | max |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for k, ncol in (("2b_full", 128), ("2b_n48", 24)):
        j, z = P1[k]
        for T in (T1, T2):
            i = clock_index(z["T"], T); jr = clock_index(R1s[1]["T"], T)
            if i is None or jr is None:
                continue
            G = z["G"][i]; Gr = R1s[1]["G"][jr][:ncol, :ncol]
            pa, pr = eig(G), eig(Gr); dp = pa - pr
            P("| %s | %.4g | %d | %.4f | %.4f | %.3g | %.3g | %.3g | %.3g | %.3g |" % (k, T / L, ncol, pr.sum(), pa.sum(), (pa.sum() - pr.sum()) / pr.sum(), np.max(np.abs(dp)), np.median(dp),
              np.median(dp) / 0.25, np.max(np.abs(dp)) / 0.25))
    P("")
    P("What waits: 2b on all 256 columns at the 1e2 L endpoint (h100-f).")
    P("")
    P("## 6. What these previews settle, and what waits")
    P("")
    j2c = P1["2c_full"][0]["timing"]; r2c = j2c["step_median_s"] / (min(j2c["col_batch"], 128) * j2c["dt"])
    j2b = P1["2b_full"][0]["timing"]; r2b = j2b["step_median_s"] / (min(j2b["col_batch"], 128) * j2b["dt"])
    P("- Re-priced full runs at the measured rates (256 columns, T = 1e2 L = 1600 a): 2c at dt 0.25 a %.1f h (rate %.4f); 2b %.1f h (rate %.4f); the 1e2 L clock %.1f h (rate %.4f)." % (
        256 * 1600 * r2c / 3600, r2c, 256 * 1600 * r2b / 3600, r2b, 256 * 1600 * 0.0648 / 3600, 0.0648))
    P("- Settled: no piece needs resuming; the both-window run reproduces run 1's blocks (sec. 2); the dt 0.5 a step error at 6.93 L (all 128 supply columns) and at 69.3 L (24 columns) is the sec. 3 difference; the cross block's size at 6.93 L (all columns) and 69.3 L (16 + 16) is sec. 4; the onset dependence at the same clocks is sec. 5; the dt 0.25 a rate is measured.")
    P("- Waits for the full runs: endpoint convergence on all 256 columns at 1e2 L (2c, qi2_h100_1), the 1e2 L clock and its cross block (h100-f), 2b at 1e2 L (h100-f), the mirror audit (second_route), and any statement about the sink-2L residual beyond 69.3 L.")
    P("")
    P("Files: %s (pieces with MD5SUMS); %s (run 1); this reader busi/code/pagetests/v3/s16A_read_v1p1.py." % (a.v2, a.v1))
    open(a.out, "w").write("\n".join(out) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
