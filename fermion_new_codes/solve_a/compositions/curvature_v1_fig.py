"""curvature_v1_fig.py: CURVATURE_readout_v1.png from the npz written by curvature_v1_data.py (no solver; figure only).
Usage: python curvature_v1_fig.py [indir] [out.png]"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else "<internal>/curvature_v1"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(D, "CURVATURE_readout_v1.png")
Z = {n: np.load(os.path.join(D, "curvature_v1_%s.npz" % n)) for n in ["B2", "P64_s2", "P64_s8", "P64_s4_tau4", "P64_s4_tau2"]}
STY = {"P64_s2": ("tab:red", "-", "sigma/a 2, tau 8"), "B2": ("black", "-", "sigma/a 4, tau 8 (B2, to 100)"),
       "P64_s8": ("tab:blue", "-", "sigma/a 8, tau 8"), "P64_s4_tau4": ("tab:green", "--", "sigma/a 4, tau 4"),
       "P64_s4_tau2": ("tab:purple", ":", "sigma/a 4, tau 2")}
fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))

# (1) max_r K_t^2 against the cap and the gate scales
a = ax[0, 0]
for n in ["P64_s2", "B2", "P64_s8", "P64_s4_tau4", "P64_s4_tau2"]:
    z = Z[n]; c, ls, lab = STY[n]
    a.plot(z["Trs"], z["Kt2_max"], color=c, ls=ls, lw=1.6, label=lab)
cap = float(Z["B2"]["cap"])
a.axhline(cap, color="0.3", lw=1.2); a.text(42, cap * 1.35, "exclusion cap (8/3) pi G rho_bar = %.1f / a^2" % cap, fontsize=8, color="0.2")
for n, s in (("P64_s2", 2), ("B2", 4), ("P64_s8", 8)):
    a.axhline((0.1 / s) ** 2, color=STY[n][0], lw=0.8, ls="-.", alpha=0.7)
a.text(42, (0.1 / 8) ** 2 * 0.45, "gate scales (0.1/sigma)^2 for sigma/a = 2, 4, 8 (dash-dot)", fontsize=8, color="0.2")
a.axvline(float(Z["B2"]["Trs"][int(Z["B2"]["kgate"])]), color="0.5", lw=0.8, ls=":")
a.axvline(float(Z["B2"]["Trs"][int(Z["B2"]["kform"])]), color="0.5", lw=0.8, ls="--")
a.set_yscale("log"); a.set_ylim(3e-6, 40); a.set_xlim(0, 100)
a.set_xlabel("T / r_s"); a.set_ylabel("max over r of K_t^2 = 2GM/r^3  [1/a^2]")
a.set_title("(a) spheres' curvature, max over r, every record (unbarred fields)", fontsize=10)
a.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.82), frameon=False)
a.text(14, 1.2e-5, "sigma/a 4: gate 11.75 (dotted), formation 12.125 (dashed)", fontsize=7.5, color="0.35")

# (2) radial profiles on the reference history
a = ax[0, 1]
z = Z["B2"]; rc = z["rc"]
cols = ["0.6", "tab:blue", "tab:red", "black", "tab:green", "tab:purple", "tab:brown"]
for c, t in zip(cols, [11.75, 12.125, 12.75, 13.25, 15.0, 40.0, 100.0]):
    j = int(np.argmin(np.abs(z["Trs"] - t)))
    a.plot(rc, z["Kt2_prof"][j], color=c, lw=1.5, label="T/r_s %g" % t)
    if z["nzeros"][j] > 0:
        for rr_ in (z["r_in_rec"][j], z["r_out_rec"][j]):
            a.plot([rr_], [np.interp(rr_, rc, z["Kt2_prof"][j])], "o", ms=5, color=c, mec="white", mew=1.0)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(0.25, 256); a.set_ylim(1e-6, 1)
a.set_xlabel("r / a"); a.set_ylabel("K_t^2(r)  [1/a^2]")
a.set_title("(a) radial profiles, sigma/a 4, tau 8 (B2); dots: trapping zeros", fontsize=10)
a.legend(fontsize=8, frameon=False, loc="lower left")

# (3) reviewer B's frame norm, cadences
a = ax[1, 0]
for n in ["P64_s2", "B2", "P64_s8"]:
    z = Z[n]; c, ls, lab = STY[n]
    a.plot(z["Trs"], z["norm_max_c1"], color=c, lw=1.6, label=lab + ", cadence 1")
z = Z["B2"]
a.plot(z["Trs"], z["norm_max_c4"], color="black", lw=1.0, ls="--", label="sigma/a 4, cadence 4 (every 4th record)")
a.axvline(float(z["Trs"][int(z["kgate"])]), color="0.5", lw=0.8, ls=":")
a.set_yscale("log"); a.set_xlim(0, 40); a.set_ylim(1e-5, 20)
a.set_xlabel("T / r_s"); a.set_ylabel("max over r of (sum_abcd R_abcd^2)^(1/2)  [1/a^2]")
a.set_title("(a) slice-frame orthonormal Riemann norm (reviewer B sec. 2), unbarred", fontsize=10)
a.legend(fontsize=8, frameon=False, loc="lower right")

# (4) null rays on the reference history
a = ax[1, 1]
z = Z["P64_s4_tau4"] if False else Z["B2"]
t = z["ray_c1_t"] / 64.0; R = z["ray_c1_r"]; s = z["ray_c1_s"]; r0 = z["ray_c1_r0"]
for i in range(R.shape[1]):
    if i % 2 and r0[i] not in (0.25,):
        continue
    a.plot(np.abs(R[:, i]), t, color="tab:red" if s[i] > 0 else "tab:blue", lw=0.8, alpha=0.8)
has = z["nzeros"] > 0
a.plot(z["r_in_rec"][has], z["Trs"][has], ".", color="black", ms=2, label="trapping zeros (run json)")
a.plot(z["r_out_rec"][has], z["Trs"][has], ".", color="black", ms=2)
a.plot([], [], color="tab:red", label="outgoing dr/dT = v + N"); a.plot([], [], color="tab:blue", label="ingoing dr/dT = v - N (reflected at r = 0)")
a.set_xlim(0, 120); a.set_ylim(12, 100)
a.set_xlabel("r / a"); a.set_ylabel("T / r_s")
a.set_title("(b) radial null rays from the formation record 12.125 (B2)", fontsize=10)
a.legend(fontsize=8, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print("wrote", OUT)
