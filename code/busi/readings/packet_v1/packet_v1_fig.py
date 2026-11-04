#!/usr/bin/env python
"""packet_v1_fig.py (busi lane, reviewer A slot, inbox #354): the figure of the packet completeness reading from the stored outputs
(packet_v1_series.npz of packet_v1_propagate.py and packet_v1_bounds.npz of packet_v1_bounds.py); no solver is run here.
Panels: (a) <r> of the ray-like packets (engine rule E solid, interpolated rule I dotted) with the classical null rays dashed and the trapped
shell's roots shaded; (b) the packets' slice-frame energy <h0> against the radial band top and ||h0||, with the rays' ln eps on the right axis;
(c) the packets' share inside the inner root; (d) ||h_m|| per record against its bound, and the per-record maximum of |rho + v J| over the cubic cap.
Usage: python packet_v1_fig.py --series DIR/packet_v1_series.npz --bounds DIR/packet_v1_bounds.npz --out FIG.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True); ap.add_argument("--bounds", required=True); ap.add_argument("--summary", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--kappa", type=int, default=1)
    a = ap.parse_args()
    S = np.load(a.series, allow_pickle=True); B = np.load(a.bounds, allow_pickle=True)
    k = a.kappa
    x = S["T_over_rs"]; r_in = S["r_in"]; r_out = S["r_out"]
    labels = S["launch_label_k%d" % k]; r0s = S["launch_r0_k%d" % k]; k0s = S["launch_k0_k%d" % k]
    kr = 0.45
    sel = [i for i in range(len(r0s)) if abs(abs(k0s[i]) - kr) < 1e-9 and labels[i] in ("centre", "r_in", "mid", "r_out")]
    cols = {"centre": "black", "r_in": "tab:red", "mid": "tab:blue", "r_out": "tab:green"}
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    # (a)
    A = ax[0, 0]
    A.fill_between(x, np.nan_to_num(r_in, nan=0.0), np.nan_to_num(r_out, nan=0.0), color="0.85", label="trapped shell (roots of N = |v|)")
    def raykey(i, q):
        return "ray_k%d_%s_%s_%s" % (k, labels[i], "out" if k0s[i] > 0 else "in", q)
    drawn = set()
    for i in sel:
        c = cols[str(labels[i])]
        A.plot(x, S["E_k%d_rmean" % k][i], color=c, ls="-", lw=1.4, label="%s, k0 %+.2f (E)" % (labels[i], k0s[i]))
        A.plot(x, S["I_k%d_rmean" % k][i], color=c, ls=":", lw=1.2)
        key = raykey(i, "r")
        if key in S.files and key not in drawn:
            drawn.add(key)
            A.plot(x, S[key], color=c, ls="--", lw=1.0, alpha=0.9, label="null ray %s %s" % (labels[i], "out" if k0s[i] > 0 else "in"))
    A.set_ylim(0, 90); A.set_xlabel("T / r_s"); A.set_ylabel("<r> / a  (packet solid E, dotted I; ray dashed)"); A.set_title("(a) packets and the null rays launched with them")
    A.legend(fontsize=6.5, ncol=3, loc="upper left")
    # (b)
    Bx = ax[0, 1]
    for i in sel:
        c = cols[str(labels[i])]
        Bx.plot(x, S["E_k%d_E0" % k][i], color=c, lw=1.3)
    Bx.axhline(np.sqrt(0.04 + 1.0), color="black", ls="--", lw=1.0, label="radial band top sqrt(m^2 + a^-2)")
    Bx.axhline(float(B["normHm"].max()), color="0.4", ls="-.", lw=0.9, label="max ||h_m|| over the records")
    Bx.set_ylim(0, 1.5); Bx.set_xlabel("T / r_s"); Bx.set_ylabel("<h0> of the packet (slice-frame energy per tick)")
    B2 = Bx.twinx()
    drawn = set()
    for i in sel:
        key = raykey(i, "lneps")
        if key in S.files and key not in drawn:
            drawn.add(key)
            B2.plot(x, S[key] / np.log(10.0), color=cols[str(labels[i])], ls="--", lw=1.0, alpha=0.9)
    B2.set_ylabel("log10 eps of the null ray (dashed)"); B2.set_ylim(-1, 9)
    Bx.set_title("(b) bounded packet energy against the unbounded ray blueshift"); Bx.legend(fontsize=7, loc="center left")
    # (c)
    C = ax[1, 0]
    for i in sel:
        c = cols[str(labels[i])]
        C.plot(x, S["E_k%d_share_in" % k][i], color=c, lw=1.3, label="%s, k0 %+.2f" % (labels[i], k0s[i]))
        C.plot(x, S["E_k%d_share_core" % k][i], color=c, lw=0.8, ls=":")
    C.set_ylim(0, 1.02); C.set_xlabel("T / r_s"); C.set_ylabel("share inside the inner root (solid), inside r < 12a (dotted)")
    C.set_title("(c) the packets pass the inner root, where the rays end"); C.legend(fontsize=7, ncol=2)
    # (d)
    D = ax[1, 1]
    xb = B["T_over_rs"]
    ik = int(np.argmin(np.abs(B["kappas"] - k)))
    D.plot(xb, B["normHm"][:, ik], color="black", lw=1.2, label="||h_m|| kappa %d" % k)
    D.plot(xb, B["bound"][:, ik], color="black", ls="--", lw=1.0, label="its bound max N ||h0|| + max|v| ||Q||")
    D.set_xlabel("T / r_s"); D.set_ylabel("||h_m|| and its bound, kappa %d (left)" % k); D.set_ylim(0, 2.2)
    D2 = D.twinx()
    D2.semilogy(xb, B["ratio_cubic_plus"].max(axis=1), color="tab:red", lw=1.2, label="max_n |rho + v J| / cubic cap")
    D2.semilogy(xb, B["ratio_chain_plus"].max(axis=1), color="tab:blue", lw=1.0, label="max_n |rho + v J| / chain constant")
    D2.axhline(1.0, color="tab:red", ls=":", lw=0.8)
    D2.set_ylabel("source-bound ratio, all channels (right, log)"); D2.set_ylim(1e-7, 3)
    D.set_title("(d) the norm bound and the per-site source bound on every record")
    h1, l1 = D.get_legend_handles_labels(); h2, l2 = D2.get_legend_handles_labels(); D.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right")
    fig.suptitle("Packet completeness on the reference collapse (sigma/a 4, tau 8, r_s/a 64), channel kappa = %d, from the formation record" % k)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
