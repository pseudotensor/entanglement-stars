"""Page figure, route 2 (figure script). Spec: WRITE_PAGE_SIGNED_v1.md sec. (3), parts A, C, H (followed as written).

Loads page_curve_amps_R2.npz (written by page_curve_amps_R2_data.py; no solver runs here) and writes
page_curve_amps_R2.pdf (7.0 x 2.4 in vector) and page_curve_amps_R2.csv (every plotted ordinate, long format).
Paths are drawn through the computed points with straight segments only (no fitting or smoothing).
"""
import csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

D = sys.argv[1] if len(sys.argv) > 1 else "<internal>/page_fig_r2"
d = np.load(f"{D}/page_curve_amps_R2.npz")
R, P, r0, zx = float(d["R"]), float(d["P"]), float(d["r0"]), float(d["zx"])
f_names = [str(x) for x in d["f_names"]]; f_vals = [float(x) for x in d["f_vals"]]
fv = dict(zip(f_names, f_vals))

BLUE, ORANGE, GREY = "#1f77b4", "#ff7f0e", "0.55"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5})


def tau(z):
    return 1.0 - np.maximum(0.0, 1.0 - np.asarray(z) / R) ** 1.5


def F(z, f):
    return np.minimum(z, f * (1.0 - np.asarray(z) / R))


def zP(f):
    return f * R / (R + f)


rows = []
def rec(panel, series, z, t, v):
    for zz, tt, vv in zip(np.atleast_1d(z), np.atleast_1d(t), np.atleast_1d(v)):
        rows.append((panel, series, f"{zz:.15g}", f"{tt:.15g}", f"{vv:.15g}"))


fig = plt.figure(figsize=tuple(d["fig_in"]))
# ---------------------------------------------------------------- (a) Conditional Page curve
ax = fig.add_axes(list(d["ax_a"]))
za = d["za"]; ta = tau(za)
Fm, Fp, F1, Fn = F(za, fv["f_-"]), F(za, fv["f_+"]), F(za, fv["1"]), F(za, fv["0.887"])
ax.fill_between(ta, Fm, Fp, color=BLUE, alpha=0.22, lw=0)
ax.plot(ta, F1, color="k", lw=1.1, solid_joinstyle="miter")
ax.plot(ta, Fn, color=BLUE, lw=1.1, ls=(0, (4, 2)))
for lab, ser in (("f_-", Fm), ("f_+", Fp), ("1", F1), ("0.887", Fn)):
    rec("a", f"F_{lab}", za, ta, ser)
for lab, col, dx, dy, ha in (("1", "k", 0.025, 0.01, "left"), ("0.887", BLUE, -0.025, 0.01, "right")):
    z_, t_ = zP(fv[lab]), float(tau(zP(fv[lab])))
    ax.axvline(t_, color=col, lw=0.7, ls=":", zorder=1)
    ax.plot([t_], [z_], "o", color=col, ms=3.6, zorder=4)
    ax.annotate(f"{t_:.4f}, {z_:.4f}", (t_, z_), xytext=(t_ + dx, z_ + dy), ha=ha, va="bottom", color=col)
    rec("a", f"PageDot_f={lab}", z_, t_, z_)
ax.plot([1.0], [0.0], "o", mfc="white", mec="k", ms=4, mew=0.8, zorder=5, clip_on=False)
ax.annotate("formal end", (1.0, 0.0), xytext=(0.955, 0.012), ha="right", va="bottom")
rec("a", "FormalEnd", R, 1.0, 0.0)
ax.set_xlim(0, 1); ax.set_ylim(0, 0.7)
ax.set_title("(a) Conditional Page curve")
ax.set_xlabel(r"Time at infinity, $t/t_*$"); ax.set_ylabel(r"Radiation entropy, $S_{\rm esc}/S_0$")

# ---------------------------------------------------------------- (b) Partner correlations
bx = fig.add_axes(list(d["ax_b"]))
zs = d["zs"]; tb = tau(zs)
cmin, cmax, cref = d["cmin"], d["cmax"], d["cref"]
tP1, tx = float(tau(P)), float(tau(zx))


def draw_b(a, lw):
    a.fill_between(tb, cmin, cmax, color=ORANGE, alpha=0.28, lw=0)
    a.fill_between(tb, 1 - cmax, 1 - cmin, color=BLUE, alpha=0.22, lw=0)
    a.axhline(0.5, color=GREY, lw=0.4, zorder=1)
    a.axvline(tP1, color="k", lw=0.7, ls=":", zorder=1)
    a.plot(tb, cref, color=ORANGE, lw=lw, solid_joinstyle="miter")
    a.plot(tb, 1 - cref, color=BLUE, lw=lw, solid_joinstyle="miter")
    a.plot([tx], [0.5], "o", color="k", ms=3.2, zorder=5)


draw_b(bx, 1.1)
for ser, v in (("c_ref_early", cref), ("c_ref_retained", 1 - cref), ("c_min_early", cmin), ("c_max_early", cmax),
               ("retained_min", 1 - cmax), ("retained_max", 1 - cmin)):
    rec("b", ser, zs, tb, v)
rec("b", "Crossing", zx, tx, 0.5)
rec("b", "PageGuide_f=1", P, tP1, np.nan)
bx.annotate("crossing", (tx, 0.5), xytext=(tx + 0.04, 0.33), ha="left", va="top",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="k", shrinkA=0.5, shrinkB=2.0))
bx.text(0.995, r0 + 0.012, f"{r0:.3f}", ha="right", va="bottom", color=ORANGE)
bx.text(0.995, 1 - r0 - 0.012, f"{1 - r0:.3f}", ha="right", va="top", color=BLUE)
bx.set_xlim(0, 1); bx.set_ylim(0, 1.05)
bx.set_title("(b) Partner correlations")
bx.set_xlabel(r"Time at infinity, $t/t_*$"); bx.set_ylabel("Share of summed response weight")
handles = [(Patch(color=ORANGE, alpha=0.28, lw=0), Line2D([], [], color=ORANGE, lw=1.1)),
           (Patch(color=BLUE, alpha=0.22, lw=0), Line2D([], [], color=BLUE, lw=1.1))]
leg = bx.legend(handles, ["earlier radiation", "retained system"], loc="center right", bbox_to_anchor=(1.0, 0.64),
                frameon=False, handlelength=1.6, borderaxespad=0.2)
bx.text(0.98, 0.52, "uniform reference;\nadmissible range", transform=bx.transAxes, ha="right", va="top", linespacing=1.1)

ins = bx.inset_axes(list(d["inset"]))
t_in0, t_in1 = (float(x) for x in d["inset_tau"])
draw_b(ins, 0.9)
ins.set_xlim(t_in0, t_in1); ins.set_ylim(0.35, 0.65)
# inset range (caption): t/t_* in [tau_P - 2(tau_x - tau_P), tau_P + 4(tau_x - tau_P)], share in [0.35, 0.65]; tick marks
# at 0.4, 0.5, 0.6 without labels, which would collide with the main axis labels or the transition
ins.set_xticks([]); ins.set_yticks([0.4, 0.5, 0.6]); ins.set_yticklabels([])
ins.tick_params(length=2)
ins.set_facecolor("white")
for sp in ins.spines.values():
    sp.set_linewidth(0.5)
span = t_in1 - t_in0
ins.annotate("Page time", (tP1, 0.65), xytext=(t_in0 + 0.2 * span, 0.675), ha="center", va="bottom", annotation_clip=False,
             arrowprops=dict(arrowstyle="-", lw=0.5, color="k", shrinkA=0.5, shrinkB=0))
ins.annotate("reference\ncrossover", (tx, 0.5), xytext=(t_in0 + 0.55 * span, 0.675), ha="left", va="bottom", annotation_clip=False,
             linespacing=1.0, arrowprops=dict(arrowstyle="-", lw=0.5, color="k", shrinkA=0.5, shrinkB=2.0))

fig.savefig(f"{D}/page_curve_amps_R2.pdf")
fig.savefig(f"{D}/page_curve_amps_R2_preview.png", dpi=300)
with open(f"{D}/page_curve_amps_R2.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["panel", "series", "z_SmargOverS0", "t_over_tstar", "value"])
    w.writerows(rows)
print("wrote", f"{D}/page_curve_amps_R2.pdf", len(rows), "csv rows")
