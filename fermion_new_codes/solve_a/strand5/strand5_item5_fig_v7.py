#!/usr/bin/env python
"""Item 5's figure (lead's draft, version 1; version 2 is the builder's layout completion: the Newtonian line drawn only below 1.5 M_0
so that the axes span the family (version 1's line reached 20 M_0 and pushed the family under the inset), explicit axis limits from the data,
the legend and the maximum's label above the curve, the inset in the empty lower left, the inset's upper limit 1e-2 so that the kernel
control's largest |D|/E_bdy (2.4e-3 at z_c = 3) is not clipped, smaller inset labels; content and colours unchanged; version 3: the legend on an
opaque frame so that the Newtonian line does not run through its text, the inset's blue key shortened so that it clears the blue curve; version 4: plain tick labels at 2, 3, 5, 10, 20 on the radius axis and
0.02, 0.05, 0.1, 0.2, 0.5, 1 on the mass axis; version 5: those tick settings placed after the log scales, which reset them; version 6, the lead's turn-6 hand-back: the dashed entry "beyond the
maximum", the blue entries "finite spacing, (ma, aH) = (0.01, 0.1), fixed-N_B Hessian positive" and "finite spacing, 1 negative direction",
the legend and the maximum's label at 8 pt, the inset's axis labels and keys at 7 pt and its tick labels at 6.5 pt, the inset keys "blue:
covariant row, finite spacing" and "red: frozen-kernel control"; version 7: at 8 pt the legend covered the maximum's label, so the first
entry drops "(60 members)" (the lead's shortening rule) and the label moves into the empty band below the dashed branch; the inset's keys move
to the empty band between the blue line and the black dots, where they clear every curve). Loads the .npz of SPEC_strand5_figure.md
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
    ax.plot(xs, ys, "k-", lw=1.5, label="covariant row, continuum")
    ax.plot(xh, yh, "k--", lw=1.5, label="beyond the maximum")
    i0 = int(np.argmax(R))
    ylo, yhi = 0.7 * min(M.min(), d["lat_M"].min()), 1.5
    xlo, xhi = 0.83 * min(R.min(), d["lat_R"].min()), 1.25 * max(R.max(), d["lat_R"].max())
    Rn = np.geomspace(xlo, xhi, 200)
    Mn = M[i0] * (R[i0] / Rn) ** 3
    keep = Mn <= yhi
    ax.plot(Rn[keep], Mn[keep], "k:", lw=1.0, label=r"Newtonian law $M \propto R_*^{-3}$")
    st = d["lat_inertia"] == 0
    ax.plot(d["lat_R"][st], d["lat_M"][st], "o", ms=5, mfc="tab:blue", mec="tab:blue", ls="none",
            label=r"finite spacing, $(ma, aH) = (0.01, 0.1)$, fixed-$N_B$ Hessian positive")
    ax.plot(d["lat_R"][~st], d["lat_M"][~st], "o", ms=5, mfc="white", mec="tab:blue", ls="none",
            label="finite spacing, 1 negative direction")
    ax.plot([Rmax], [Mmax], "ko", ms=6.5, zorder=5)
    ax.annotate("$z_c = %.3f$\n$2GM/R_* = %.3f$\n$E_b/M = %.3f$" % (zmax, Cmax, EbM),
                xy=(Rmax, Mmax), xytext=(0.15, 0.63), textcoords="axes fraction", fontsize=8, va="top", ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.8, color="k"))
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    xt = [t for t in (2, 3, 5, 10, 20) if xlo <= t <= xhi]
    yt = [t for t in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0) if ylo <= t <= yhi]
    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.xaxis.set_major_formatter(FixedFormatter(["%g" % t for t in xt]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_major_formatter(FixedFormatter(["%g" % t for t in yt]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"$R_*/R_0$,  $R_0 = m_P/(\sqrt{8}\,m^2)$")
    ax.set_ylabel(r"$M/M_0$,  $M_0 = m_P^3/(\sqrt{8}\,m^2)$")
    ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=1.0, edgecolor="none", facecolor="white")

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
    ins.tick_params(labelsize=6.5, pad=1)
    ins.set_yticks([1e-16, 1e-12, 1e-8, 1e-4])
    ins.text(0.03, 0.92, "lock audit", transform=ins.transAxes, fontsize=7)
    ins.text(0.36, 0.43, "red: frozen-kernel control", transform=ins.transAxes, fontsize=7, color="tab:red")
    ins.text(0.36, 0.35, "blue: covariant row, finite spacing", transform=ins.transAxes, fontsize=7, color="tab:blue")

    fig.tight_layout()
    fig.savefig(a.out + ".pdf")
    fig.savefig(a.out + ".png", dpi=200)
    print("wrote", a.out + ".pdf", a.out + ".png")


if __name__ == "__main__":
    main()
