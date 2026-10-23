#!/usr/bin/env python
"""Item 5's figure (lead's draft, version 1; version 2 is the builder's layout completion: the Newtonian line drawn only below 1.5 M_0
so that the axes span the family (version 1's line reached 20 M_0 and pushed the family under the inset), explicit axis limits from the data,
the legend and the maximum's label above the curve, the inset in the empty lower left, the inset's upper limit 1e-2 so that the kernel
control's largest |D|/E_bdy (2.4e-3 at z_c = 3) is not clipped, smaller inset labels; content and colours unchanged). Loads the .npz of SPEC_strand5_figure.md
and writes <out>.pdf and <out>.png with matplotlib. No solver, no computation beyond the Newtonian reference line.
Colours: black, tab:blue, tab:red."""
import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True, help="output stem: writes <out>.pdf and <out>.png")
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=False)
    zc, M, R = d["ref_zc"], d["ref_M"], d["ref_R"]
    zmax, Mmax, Rmax, Cmax = (float(d[k]) for k in ("ref_zmax", "ref_Mmax", "ref_Rmax", "ref_Cmax"))
    EbM = float(d["ref_EbM_max"])
    low, high = zc <= zmax, zc >= zmax
    # the two branches joined through the maximum
    xs, ys = np.append(R[low], Rmax), np.append(M[low], Mmax)
    xh, yh = np.insert(R[high], 0, Rmax), np.insert(M[high], 0, Mmax)

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot(xs, ys, "k-", lw=1.5, label="covariant row, continuum (60 members)")
    ax.plot(xh, yh, "k--", lw=1.5, label="beyond the maximum (1 negative direction)")
    i0 = int(np.argmax(R))
    ylo, yhi = 0.7 * min(M.min(), d["lat_M"].min()), 1.5
    xlo, xhi = 0.83 * min(R.min(), d["lat_R"].min()), 1.25 * max(R.max(), d["lat_R"].max())
    Rn = np.geomspace(xlo, xhi, 200)
    Mn = M[i0] * (R[i0] / Rn) ** 3
    keep = Mn <= yhi
    ax.plot(Rn[keep], Mn[keep], "k:", lw=1.0, label=r"Newtonian law $M \propto R_*^{-3}$")
    st = d["lat_inertia"] == 0
    ax.plot(d["lat_R"][st], d["lat_M"][st], "o", ms=5, mfc="tab:blue", mec="tab:blue", ls="none",
            label=r"covariant row, $ma = 0.01$ (positive second variation)")
    ax.plot(d["lat_R"][~st], d["lat_M"][~st], "o", ms=5, mfc="white", mec="tab:blue", ls="none",
            label=r"$ma = 0.01$ (1 negative direction)")
    ax.plot([Rmax], [Mmax], "ko", ms=6.5, zorder=5)
    ax.annotate("$z_c = %.3f$\n$2GM/R_* = %.3f$\n$E_b/M = %.3f$" % (zmax, Cmax, EbM),
                xy=(Rmax, Mmax), xytext=(0.03, 0.97), textcoords="axes fraction", fontsize=7, va="top", ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.8, color="k"))
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$R_*/R_0$,  $R_0 = m_P/(\sqrt{8}\,m^2)$")
    ax.set_ylabel(r"$M/M_0$,  $M_0 = m_P^3/(\sqrt{8}\,m^2)$")
    ax.legend(loc="upper right", fontsize=7, frameon=False)

    ins = ax.inset_axes([0.15, 0.09, 0.40, 0.37])
    floor = 1e-16
    ins.plot(zc, np.maximum(np.abs(d["ref_lock"]), floor), "k.", ms=3)
    ins.plot(d["lat_zc"], np.maximum(np.abs(d["lat_lock"]), floor), "-", color="tab:blue", lw=1.2)
    ins.plot(d["lat_pred_zc"], np.abs(d["lat_pred"]), "s", ms=5, mfc="white", mec="tab:blue", ls="none")
    ins.plot(d["ker_zc"], np.maximum(np.abs(d["ker_lock"]), floor), "-", color="tab:red", lw=1.2)
    ins.plot(d["ker_pred_zc"], np.abs(d["ker_pred"]), "s", ms=5, mfc="white", mec="tab:red", ls="none")
    ins.axvline(zmax, color="k", ls=":", lw=0.8)
    ins.set_xscale("log")
    ins.set_yscale("log")
    ins.set_ylim(floor, 1e-2)
    ins.set_xlabel(r"$z_c$", fontsize=7, labelpad=0)
    ins.set_ylabel(r"$|D|/E_{\rm bdy}$", fontsize=7, labelpad=0)
    ins.tick_params(labelsize=6, pad=1)
    ins.set_yticks([1e-16, 1e-12, 1e-8, 1e-4])
    ins.text(0.03, 0.92, "lock audit", transform=ins.transAxes, fontsize=6.5)
    ins.text(0.03, 0.82, "red: frozen-kernel control", transform=ins.transAxes, fontsize=5.5, color="tab:red")
    ins.text(0.03, 0.72, r"blue: covariant row, $ma = 0.01$", transform=ins.transAxes, fontsize=5.5, color="tab:blue")

    fig.tight_layout()
    fig.savefig(a.out + ".pdf")
    fig.savefig(a.out + ".png", dpi=200)
    print("wrote", a.out + ".pdf", a.out + ".png")


if __name__ == "__main__":
    main()
