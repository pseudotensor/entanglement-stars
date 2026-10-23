#!/usr/bin/env python
"""Item 7's figure, version 3 (SPEC_strand7_page.md secs. 8, 12.7 and 13.8; builder analyst): loads the data script's .npz (version 3) and
writes .pdf and .png.  x: the external clock T/r_s0.  Left axis (nats), black and blue: the excess entanglement entropy (over the
instantaneous vacuum's) of the radiation region with its complement (black, solid; P's channel sum); the diagonal entropy of the exterior
eigenmodes at their occupations, a different object shown for comparison (black, dashed); the excess entanglement entropy across the
reference cut (blue, solid; scaled by a power of 10 when its maximum exceeds five times the first curve's, the factor in the legend); P's
j = 1/2 alone (black, solid, lw 0.8, "the same at j = 1/2 along L") with the comparison C at j = 1/2 (black, dotted, lw 1.0); C's channel sum
(black, dash-dot) only if C has three channels.  Right axis (logarithmic, 1e-3 to 2), red: the two-point correlation ratio across the
reference cut at d = h (solid) and 4h (dashed), relative to the initial vacuum, clipped at 1e-3.  Vertical dotted black lines at T_1 (lapse at
the shell below 0.5), T_2p (the band-edge marker) and T_2 (the declared lapse marker N_s = h T_H), tagged above the axes and explained in the
legend.  The legend sits below the axes.  Colors black, blue and red only (sec. 13.1's labels)."""
import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="output path without extension")
    a = ap.parse_args()
    z = np.load(a.data)
    T = z["T_over_rs0"]
    fig = plt.figure(figsize=(6.4, 5.6))
    ax = fig.add_axes([0.10, 0.37, 0.78, 0.56])
    l1, = ax.plot(T, z["dS_rad"], color="black", ls="-", lw=1.6,
                  label="excess entanglement entropy (over the instantaneous vacuum's) of the radiation region with its complement, along L")
    l2, = ax.plot(T, z["S_diag_rad"], color="black", ls="--", lw=1.2,
                  label="diagonal entropy of the exterior eigenmodes at their occupations: a different object, shown for comparison")
    m1 = np.nanmax(np.abs(z["dS_rad"])) if len(T) else 0.0
    m3 = np.nanmax(np.abs(z["dS_thr"])) if len(T) else 0.0
    p = 0
    if m1 > 0 and m3 > 5 * m1:
        p = int(math.ceil(math.log10(m3 / m1)))
    l3, = ax.plot(T, z["dS_thr"] / 10 ** p, color="blue", ls="-", lw=1.4,
                  label="excess entanglement entropy across the reference cut r = A_0" + (" (x 10^-%d)" % p if p else ""))
    lines = [l1, l2, l3]
    xmax = float(np.nanmax(T)) if len(T) else 40.0
    if "dS_rad_P_half" in z.files and len(z["dS_rad_P_half"]) == len(T):
        lh, = ax.plot(T, z["dS_rad_P_half"], color="black", ls="-", lw=0.8, label="the same at j = 1/2 along L")
        lines.append(lh)
    if "dS_rad_C_half" in z.files and len(z["dS_rad_C_half"]):
        lc, = ax.plot(z["T_over_rs0_C_half"], z["dS_rad_C_half"], color="black", ls=":", lw=1.0,
                      label="the same at j = 1/2 along the rows' own unlocked collapse C (an approximate numerical history)")
        lines.append(lc)
        xmax = max(xmax, float(np.nanmax(z["T_over_rs0_C_half"])))
    if "channels_C" in z.files and len(z["channels_C"]) == 3 and "dS_rad_C" in z.files:
        lc3, = ax.plot(z["T_over_rs0_C"], z["dS_rad_C"], color="black", ls="-.", lw=1.0,
                       label="the radiation region's excess entropy along C, channels 1/2, 3/2, 5/2")
        lines.append(lc3)
    ax.set_xlim(0.0, xmax)
    ax.set_xlabel("external clock T / r_s0")
    ax.set_ylabel("entropy (nats)")
    ax2 = ax.twinx()
    l4, = ax2.plot(T, np.clip(z["g_h"], 1e-3, 2), color="red", ls="-", lw=1.2,
                   label="two-point correlation ratio across the reference cut, d = h, relative to the initial vacuum (right axis)")
    l5, = ax2.plot(T, np.clip(z["g_4h"], 1e-3, 2), color="red", ls="--", lw=1.2, label="the same at d = 4h (both clipped at 1e-3)")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-3, 2)
    ax2.set_ylabel("correlation ratio")
    lines += [l4, l5]
    tags = []
    for key, tag, ha in (("T_1_over_rs0", "T_1", "center"), ("T_2p_over_rs0", "T_2p", "right"), ("T_2_over_rs0", "T_2", "left")):
        if key in z.files:
            v = float(z[key])
            if np.isfinite(v):
                ax.axvline(v, color="black", ls=":", lw=0.9)
                ax.annotate(tag, xy=(v, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 3), textcoords="offset points", ha=ha, va="bottom", fontsize=7)
                tags.append(tag)
    if tags:
        lines.append(Line2D([], [], color="black", ls=":", lw=0.9,
                            label="dotted verticals: T_1, the lapse at the shell below 0.5; T_2p, the band-edge marker; T_2, the declared lapse marker N_s = h T_H"))
    fig.legend(lines, [l_.get_label() for l_ in lines], loc="upper left", bbox_to_anchor=(0.02, 0.285), fontsize=6.2, frameon=False, ncol=1,
               handlelength=2.6)
    fig.savefig(a.out + ".pdf")
    fig.savefig(a.out + ".png", dpi=200)
    print("wrote %s.pdf and %s.png" % (a.out, a.out))


if __name__ == "__main__":
    main()
