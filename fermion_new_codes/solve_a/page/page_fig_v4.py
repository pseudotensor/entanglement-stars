#!/usr/bin/env python
"""page_fig_v4.py. Changes against page_fig_v1.py (builder analyst; SPEC_final4b_engine.md 31.4 (9) and 31.6 BUILDER item 1; the GAP03 review
review_GAP03_page_curve.tex, R11, R4, R9): reads page_recursion_v4.py's <tag>_curve.npz (the figure never re-runs the solver); (R11) the
validation status from the npz ("validated" or "unresolved: ...") is printed on the figure, and an unresolved run carries the word
UNRESOLVED in red in its title; the title carries the run's actual mass metadata (m a and m/T_H from the npz; v1 hardcoded m/T_H = 0) and
the run's label ("on the exact exterior sequence", or "timing, not the pilot"); (R4) the left abscissa is the joined per-stage chart
retarded time at R_cut, labelled "not a global clock"; (R3 to R8) the left ordinate is labelled the entropy of the retained-channel
recursion's escaped sector (memory route), not a physical escaping entropy; (R9) the levels eta A/a^2 and A/(4G) are labelled supplied
levels for comparison; (R5, R8) checkpoints where the decrease from the maximum is established above the combined uncertainty are marked,
and the uncertainty band (S_esc minus and plus the accumulated bound) is drawn. Colors black, blue, red, grey (no yellow, no viridis).
page_fig_v1.py (lead reviewer A, GAP03 integration): the figure from <tag>_curve.npz (left: S_esc(t_infinity), the direct route, the stage
transitions, the levels, the stationary-channel comparator; right: the collected flux and the collar current against the continuum thermal
luminosity of the current charge).
Usage: python page_fig_v4.py <tag>_curve.npz out.pdf [--log-left]"""
import argparse
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _s(d, key, default=""):
    return str(d[key]) if key in d.files else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("out")
    ap.add_argument("--log-left", action="store_true")
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=True)
    rs1 = float(d["rs1"]); g = float(d["g"]); eta = float(d["eta"])
    t = d["t"] / rs1; tinf = d["t_inf"] / rs1
    S = d["S_esc_total"]; Sd = d["S_direct_total"]; stage = d["stage"]
    L = d["L_collected"]; Lc = d["L_collar"]; Lcont = d["L_cont"]
    m = float(d["m"]) if "m" in d.files else float("nan"); m_TH = float(d["m_over_TH"]) if "m_over_TH" in d.files else float("nan")
    status = _s(d, "validation_status", "no validation status in the npz (not a v4 run)")
    label = _s(d, "label_sequence", "on the exact exterior sequence")
    validated = status.strip() == "validated"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    ax1.plot(tinf, S, color="black", lw=1.6, label=r"$S_{\rm esc}$ of the recursion's escaped sector (memory route, $\mathrm{tr}\,s(M)$)")
    if "uncertainty_total" in d.files:
        U = d["uncertainty_total"]
        ok = np.isfinite(U)
        if ok.any():
            ax1.fill_between(tinf[ok], (S - U)[ok], (S + U)[ok], color="grey", alpha=0.3, lw=0, label="the accumulated truncation bound (D8, R5)")
    ok = np.isfinite(Sd)
    if ok.any():
        ax1.plot(tinf[ok], Sd[ok], color="blue", lw=1.0, ls="--", label="direct route (stage-1 covariance)")
    TH = float(d["TH"])
    Sdot = 1.6391 * 0.0052373 / rs1 ** 2 / TH
    ax1.plot(tinf, Sdot * d["t"], color="red", lw=0.8, ls=":", label=r"comparator $b\,L_{\rm cont}\,t/T_H$ (stationary channel; not an input)")
    if "turnover_established_run" in d.files and np.any(d["turnover_established_run"]):
        sel = d["turnover_established_run"].astype(bool)
        ax1.plot(tinf[sel], S[sel], "v", color="red", ms=4, label="decrease from the maximum above the combined uncertainty (R5, R8)")
    for k in np.unique(stage)[1:]:
        i = int(np.argmax(stage == k))
        ax1.axvline(tinf[i], color="grey", lw=0.5, alpha=0.6)
    for lev, lab in ((float(d["eta_A_over_a2_initial"]), r"$\eta A/a^2$, $\eta=%.4f$ (supplied level, R9)" % eta), (float(d["A_over_4G_initial"]), r"$A/(4G)$, $G/a^2=%g$ (supplied level, R9)" % g)):
        if a.log_left or lev < 3 * max(np.nanmax(S), 1e-9):
            ax1.axhline(lev, color="grey", lw=0.8, ls="-.")
            ax1.text(tinf[-1], lev, lab, ha="right", va="bottom", fontsize=7, color="grey")
    if a.log_left:
        ax1.set_yscale("log")
    ax1.set_xlabel(r"$t_\infty / r_{s,1}$: the joined per-stage chart retarded time at $R_{\rm cut}$ (not a global clock, R4)", fontsize=8)
    ax1.set_ylabel("entropy of the recursion's escaped sector [nats]", fontsize=8)
    head = (r"$r_s/a=%g$, $G/a^2=%g$, $m a=%g$, $m/T_H=%.3f$ (collapse: 160); %s" % (rs1, g, m, m_TH, label))
    if not validated:
        head = "UNRESOLVED. " + head
    fig.suptitle(head, fontsize=9, color=("black" if validated else "red"))
    ax1.legend(fontsize=6, loc="upper left")
    ax2.plot(t, L, color="black", lw=1.2, label="collected flux (memory route, Killing energy per unit time)")
    ax2.plot(t, Lc, color="blue", lw=0.8, label=r"collar current $\mathrm{tr}[\Delta(t) J_{\rm out}]$ (exact stage state)")
    ax2.plot(t, Lcont, color="red", lw=0.8, ls=":", label=r"continuum thermal $0.0052373/r_s^2$ (comparator)")
    ax2.set_yscale("symlog", linthresh=max(1e-12, 1e-3 * np.nanmax(np.abs(L)) if np.isfinite(L).any() else 1e-12))
    ax2.set_xlabel(r"$t / r_{s,1}$")
    ax2.set_ylabel("flux [lattice units]")
    ax2.legend(fontsize=7, loc="upper right")
    fig.text(0.01, 0.005, "\n".join(textwrap.wrap("validation status (8.3 thresholds): " + status, 190)), fontsize=6, color=("black" if validated else "red"), va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(a.out, dpi=200)
    print("wrote", a.out, "; validation status:", status)


if __name__ == "__main__":
    main()
