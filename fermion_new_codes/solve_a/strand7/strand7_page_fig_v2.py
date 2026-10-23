#!/usr/bin/env python
"""Item 7's figure (SPEC_strand7_page.md sec. 8; builder analyst, from the spec): loads the data script's .npz and writes .pdf and .png.
x: the external clock T/r_s0.  Left axis (nats): the entanglement of the radiation zone with the rest (black, solid), the entropy of the
exterior modes if uncorrelated (black, dashed), the entanglement across the trapping surface (blue, solid; scaled by a power of 10 when its
maximum exceeds five times the first curve's, the factor printed in the legend).  Right axis (logarithmic, 1e-3 to 2): the two-point
correlation across the trapping surface at d = h (red, solid) and at d = 4h (red, dashed), relative to the initial vacuum.  Vertical dotted
black lines at T_1 and T_2.  Colors black, blue and red only.
Version 2 (SPEC_strand7_page.md sec. 12.7): if the .npz carries dS_rad_C (the comparison C), it is drawn on the left axis in black, dotted,
lw 1.0, labelled "the same along the quasi-static rows' own collapse (no lock)"; the legend notes that the right-axis curves are clipped at
1e-3."""
import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="output path without extension")
    a = ap.parse_args()
    z = np.load(a.data)
    T = z["T_over_rs0"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    l1, = ax.plot(T, z["dS_rad"], color="black", ls="-", lw=1.6, label="entanglement of the radiation zone (r > 2 r_s) with the rest")
    l2, = ax.plot(T, z["S_diag_rad"], color="black", ls="--", lw=1.2, label="entropy of the exterior modes if uncorrelated")
    m1 = np.nanmax(np.abs(z["dS_rad"])) if len(T) else 0.0
    m3 = np.nanmax(np.abs(z["dS_thr"])) if len(T) else 0.0
    p = 0
    if m1 > 0 and m3 > 5 * m1:
        p = int(math.ceil(math.log10(m3 / m1)))
    lab3 = "entanglement across the trapping surface (r = A)" + (" x 10^-%d" % p if p else "")
    l3, = ax.plot(T, z["dS_thr"] / 10 ** p, color="blue", ls="-", lw=1.4, label=lab3)
    extra = []
    if "dS_rad_C" in z.files:
        lc, = ax.plot(z["T_over_rs0_C"], z["dS_rad_C"], color="black", ls=":", lw=1.0, label="the same along the quasi-static rows' own collapse (no lock)")
        extra.append(lc)
    ax.set_xlabel("external clock T / r_s")
    ax.set_ylabel("entropy (nats)")
    ax2 = ax.twinx()
    l4, = ax2.plot(T, np.clip(z["g_h"], 1e-3, 2), color="red", ls="-", lw=1.2,
                   label="two-point correlation across the trapping surface, d = h (relative to the initial vacuum)")
    l5, = ax2.plot(T, np.clip(z["g_4h"], 1e-3, 2), color="red", ls="--", lw=1.2, label="the same at d = 4h (both clipped at 1e-3)")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-3, 2)
    ax2.set_ylabel("correlation ratio")
    lines = [l1, l2] + extra + [l3, l4, l5]
    for key, lab in (("T_1_over_rs0", "lapse at the surface below 0.5"), ("T_2_over_rs0", "below h T_H")):
        v = float(z[key])
        if np.isfinite(v):
            ax.axvline(v, color="black", ls=":", lw=1.0)
            ax.text(v, ax.get_ylim()[1], " " + lab, rotation=90, va="top", ha="left", fontsize=7)
    ax.legend(lines, [l_.get_label() for l_ in lines], fontsize=6.5, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(a.out + ".pdf")
    fig.savefig(a.out + ".png", dpi=200)
    print("wrote %s.pdf and %s.png" % (a.out, a.out))


if __name__ == "__main__":
    main()
