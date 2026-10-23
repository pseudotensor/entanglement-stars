#!/usr/bin/env python
"""Item 5's figure (lead's draft, version 1; the builder completes and runs). Loads the .npz of SPEC_strand5_figure.md
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
    Rn = np.geomspace(0.9 * R.min(), 1.1 * R.max(), 50)
    ax.plot(Rn, M[i0] * (R[i0] / Rn) ** 3, "k:", lw=1.0, label=r"Newtonian law $M \propto R_*^{-3}$")
    st = d["lat_inertia"] == 0
    ax.plot(d["lat_R"][st], d["lat_M"][st], "o", ms=5, mfc="tab:blue", mec="tab:blue", ls="none",
            label=r"covariant row, $ma = 0.01$ (positive second variation)")
    ax.plot(d["lat_R"][~st], d["lat_M"][~st], "o", ms=5, mfc="white", mec="tab:blue", ls="none",
            label=r"$ma = 0.01$ (1 negative direction)")
    ax.plot([Rmax], [Mmax], "ko", ms=6.5, zorder=5)
    ax.annotate(r"$z_c = %.3f$, $2GM/R_* = %.3f$, $E_b/M = %.3f$" % (zmax, Cmax, EbM),
                xy=(Rmax, Mmax), xytext=(0.34, 0.68), textcoords="axes fraction", fontsize=8,
                arrowprops=dict(arrowstyle="-", lw=0.8, color="k"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$R_*/R_0$,  $R_0 = m_P/(\sqrt{8}\,m^2)$")
    ax.set_ylabel(r"$M/M_0$,  $M_0 = m_P^3/(\sqrt{8}\,m^2)$")
    ax.legend(loc="upper right", fontsize=7, frameon=False)

    ins = ax.inset_axes([0.09, 0.08, 0.46, 0.46])
    floor = 1e-16
    ins.plot(zc, np.maximum(np.abs(d["ref_lock"]), floor), "k.", ms=3)
    ins.plot(d["lat_zc"], np.maximum(np.abs(d["lat_lock"]), floor), "-", color="tab:blue", lw=1.2)
    ins.plot(d["lat_pred_zc"], np.abs(d["lat_pred"]), "s", ms=5, mfc="white", mec="tab:blue", ls="none")
    ins.plot(d["ker_zc"], np.maximum(np.abs(d["ker_lock"]), floor), "-", color="tab:red", lw=1.2)
    ins.plot(d["ker_pred_zc"], np.abs(d["ker_pred"]), "s", ms=5, mfc="white", mec="tab:red", ls="none")
    ins.axvline(zmax, color="k", ls=":", lw=0.8)
    ins.set_xscale("log")
    ins.set_yscale("log")
    ins.set_ylim(floor, 1e-3)
    ins.set_xlabel(r"$z_c$", fontsize=8, labelpad=1)
    ins.set_ylabel(r"$|D|/E_{\rm bdy}$", fontsize=8, labelpad=1)
    ins.tick_params(labelsize=7)
    ins.text(0.04, 0.9, "lock audit", transform=ins.transAxes, fontsize=7)
    ins.text(0.04, 0.78, "red: frozen-kernel control", transform=ins.transAxes, fontsize=6, color="tab:red")
    ins.text(0.04, 0.66, r"blue: covariant row, $ma = 0.01$", transform=ins.transAxes, fontsize=6, color="tab:blue")

    fig.tight_layout()
    fig.savefig(a.out + ".pdf")
    fig.savefig(a.out + ".png", dpi=200)
    print("wrote", a.out + ".pdf", a.out + ".png")


if __name__ == "__main__":
    main()
